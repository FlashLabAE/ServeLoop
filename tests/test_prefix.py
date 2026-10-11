"""Prefix reuse is content-addressed, capacity-accounted, and completion-ordered."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_serveloop import _dense, _moe, _engine, _pool_spec, _spec, _trace, PlacementExecutor
from serveloop.contracts import (ServeLoopError, RequestSpec, SchedulerPolicy, RouterTrace,
                              RouterDecision, TraceProvenance, canonical_sha256)
from serveloop.placement import ServeLoopPlacement
from serveloop.prefix import block_keys


class PrefixTests(unittest.TestCase):
    simulator = None

    def test_moe_token_identity_alone_cannot_grant_prefix_hits(self):
        model = _moe()
        models = {model.model_id: model}
        with self.assertRaisesRegex(ServeLoopError, "routing identity"):
            ServeLoopPlacement(models=models, spec=replace(_spec(models), prefix_cache_bytes=65536))

    def test_moe_prefix_reuse_is_bound_to_routes_and_compiler_identity(self):
        class RecordedPrefixRouter(RouterTrace):
            def prefix_block_keys(self, *, request, model, block_tokens):
                parent, keys = model.digest, []
                for ordinal, tokens in enumerate(block_keys(request, model.digest, block_tokens)):
                    routes = [self.experts_for(request=request, token_index=token, layer=layer, model=model)
                              for token in range(ordinal * block_tokens, (ordinal + 1) * block_tokens)
                              for layer in range(model.num_layers)]
                    parent = canonical_sha256(dict(parent=parent, tokens=tokens, routes=routes))
                    keys.append(parent)
                return tuple(keys)

        model = _moe()
        models = {model.model_id: model}
        requests = tuple(RequestSpec(name, i * 1000, model.model_id, 33, 2, (1,) * 34)
                         for i, name in enumerate(('a', 'b', 'c')))
        router = RecordedPrefixRouter(TraceProvenance(kind='ci_fixture', source='recorded routes'),
            tuple(RouterDecision(request.request_id, token, 0, (0, 2) if request.request_id == 'c' else (0, 1))
                  for request in requests for token in range(34)))
        spec = replace(_pool_spec(model, 40), prefix_cache_bytes=65536)
        placement = ServeLoopPlacement(models=models, spec=spec, router=router)
        executor = PlacementExecutor(placement, latency_ns=10)
        policy = SchedulerPolicy(max_batch_requests=1, max_batch_tokens=16, prefill_chunk_tokens=16)
        result = _engine(models, _trace(*requests), executor, policy, router=router).run()
        self.assertEqual([r['prefix_hit_tokens'] for r in result['requests']], [0, 32, 0])
        self.assertEqual(placement.receipt()['prefix_router_sha256'], router.digest)
        changed = replace(router, provenance=TraceProvenance(kind='ci_fixture', source='different binding'))
        placement = ServeLoopPlacement(models=models, spec=spec, router=router)
        with self.assertRaisesRegex(ServeLoopError, 'differs from the prefix-cache router'):
            _engine(models, _trace(*requests), PlacementExecutor(placement, latency_ns=10), policy, router=changed).run()

    def simulate(self, requests, *, pool_blocks=40, cache_bytes=65536, ttl_ns=None, batch_tokens=16):
        model = _dense()
        spec = replace(_pool_spec(model, pool_blocks), prefix_cache_bytes=cache_bytes,
                       prefix_cache_ttl_ns=ttl_ns)
        placement = ServeLoopPlacement(models={model.model_id: model}, spec=spec)
        executor = PlacementExecutor(placement, latency_ns=10.0)
        result = _engine({model.model_id: model}, _trace(*requests), executor,
                         SchedulerPolicy(max_batch_requests=4, max_batch_tokens=batch_tokens,
                                         prefill_chunk_tokens=16)).run()
        self.assertTrue(placement.receipt()["final_invariants"]["all_request_kv_released"])
        self.assertTrue(placement.receipt()["final_invariants"]["kv_hot_allocations_accounted_for"])
        self.assertLessEqual(placement._prefix.bytes, cache_bytes)
        return result, placement

    def request(self, name, arrival, *, salt="", first_token=0, tokens=True):
        sequence = tuple([first_token] + [index % 8 for index in range(1, 34)])
        return RequestSpec(name, arrival, "dense", 33, 2, sequence if tokens else None, salt)

    def test_hit_skips_prefill_but_preserves_output_work(self):
        requests = (self.request("first", 0), self.request("reuse", 1000))
        cached, placement = self.simulate(requests)
        uncached, _ = self.simulate(requests, cache_bytes=0)
        self.assertEqual([row["prefix_hit_tokens"] for row in cached["requests"]], [0, 32])
        self.assertEqual(cached["summary"]["output_tokens"], uncached["summary"]["output_tokens"])
        self.assertLess(cached["summary"]["iterations"], uncached["summary"]["iterations"])
        self.assertEqual(placement._prefix.receipt()["hit_tokens"], 32)
        reuse = [item for batch in cached["batches"] for item in batch["schedule"]["slices"]
                 if item["request_id"] == "reuse"]
        self.assertEqual(reuse[0]["token_begin"], 32)
        self.assertEqual(reuse[0]["context_tokens_before"], 32)

    def test_parent_chain_salt_and_unidentified_requests_cannot_false_hit(self):
        result, _ = self.simulate((self.request("first", 0), self.request("different", 1000, first_token=7),
                                  self.request("isolated", 2000, salt="tenant-b"),
                                  self.request("length-only", 3000, tokens=False)))
        self.assertEqual([row["prefix_hit_tokens"] for row in result["requests"]], [0, 0, 0, 0])
        request = self.request("model-test", 0)
        self.assertNotEqual(block_keys(request, "model-a", 16), block_keys(request, "model-b", 16))

    def test_ttl_and_memory_pressure_evict_without_corrupting_active_blocks(self):
        expired, placement = self.simulate((self.request("first", 0), self.request("later", 1000)), ttl_ns=100)
        self.assertEqual(expired["requests"][1]["prefix_hit_tokens"], 0)
        self.assertGreater(placement._prefix.stats["expired_blocks"], 0)
        crowded, placement = self.simulate((self.request("first", 0), self.request("later", 1000),
                                           self.request("unrelated", 2000, salt="another")),
                                          pool_blocks=6, cache_bytes=16384)
        self.assertEqual(crowded["summary"]["output_tokens"], 6)
        self.assertEqual(placement._hot.free_blocks + len(placement._hot.references), 6)
        self.assertGreater(placement._prefix.stats["evicted_blocks"], 0)

    def test_shared_blocks_survive_release_and_cache_eviction(self):
        _, placement = self.simulate((self.request("warm", 0),))
        for identifier in ("active-a", "active-b"):
            self.assertEqual(placement.admit_request(self.request(identifier, 1000), now_ns=1000), 32)
        shared = tuple(block for layer in placement._kv["active-b"].blocks for block in layer)
        placement.release_request("active-a")
        for key in tuple(placement._prefix.entries):
            placement._prefix.evict(key)
        self.assertTrue(all(placement._hot.references[block] == 1 for block in shared))
        placement.release_request("active-b")
        self.assertEqual(placement._hot.free_blocks, placement._hot.capacity_blocks)

    def test_partial_blocks_are_private_but_same_batch_full_prefix_is_shared(self):
        request = RequestSpec("short", 0, "dense", 15, 1, tuple(index % 8 for index in range(15)))
        _, placement = self.simulate((request, replace(request, request_id="same", arrival_ns=1000)))
        self.assertEqual(placement._prefix.stats["inserted_blocks"], 0)
        result, _ = self.simulate((self.request("first", 0), self.request("simultaneous", 0)), batch_tokens=32)
        self.assertEqual([row["prefix_hit_tokens"] for row in result["requests"]], [0, 16])

    def test_real_simulator_feedback_reuses_only_committed_blocks(self):
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        from serveloop.contracts import RooflineTimingProvider
        from serveloop.hbfsim import HbfSimExecutor

        model = _dense()
        models = {model.model_id: model}
        trace = _trace(self.request("first", 0), self.request("reuse", 1e9))
        observations = []
        for capacity in (0, 65536):
            placement = ServeLoopPlacement(models=models, spec=replace(_spec(models), prefix_cache_bytes=capacity))
            executor = HbfSimExecutor(simulator_path=self.simulator,
                                     system_config_paths=(ROOT / "configs/4hbm-4hbf.cfg",), placement=placement)
            try:
                result = _engine(models, trace, executor,
                                 SchedulerPolicy(max_batch_requests=2, max_batch_tokens=16, prefill_chunk_tokens=16),
                                 timing=RooflineTimingProvider(peak_tflops=1.0, efficiency=0.5)).run()
            finally:
                executor.close()
            receipt = executor.final_receipt()
            self.assertTrue(receipt["placement"]["final_invariants"]["kv_hot_allocations_accounted_for"])
            observations.append(result)
        self.assertEqual([result["summary"]["prefix_hit_tokens"] for result in observations], [0, 32])
        self.assertLess(observations[1]["requests"][1]["ttft_ns"], observations[0]["requests"][1]["ttft_ns"])

    def test_generator_reuse_does_not_change_arrivals_or_work_sizes(self):
        from serveloop.synthetic import SyntheticRequestConfig, generate_requests

        config = SyntheticRequestConfig(4, 10, 65, 0, 2, 0, {"dense": 1}, 37,
                                        shared_prefix_tokens=64)
        trace = generate_requests(config)
        misses = generate_requests(replace(config, prefix_reuse_probability=0))
        self.assertEqual([(request.arrival_ns, request.prompt_tokens, request.output_tokens) for request in trace.requests],
                         [(request.arrival_ns, request.prompt_tokens, request.output_tokens) for request in misses.requests])
        self.assertEqual(trace.requests[0].token_ids[:64], trace.requests[1].token_ids[:64])
        self.assertNotEqual(misses.requests[0].token_ids[:64], misses.requests[1].token_ids[:64])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path)
    arguments, remaining = parser.parse_known_args()
    PrefixTests.simulator = arguments.simulator
    unittest.main(argv=[sys.argv[0], *remaining])

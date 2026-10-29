"""Aggregate KV producer causality; metadata-only, no native/model execution."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import unittest

from serveloop.catalog import load_model_any
from serveloop.compiler import ServeLoopCompiler
from serveloop.contracts import (
    BatchSlice, LinearTimingProvider, RequestSpec, RequestTrace, ScheduledBatch,
    TraceProvenance,
)
from serveloop.synthetic import HotsetZipfRouter


ROOT = Path(__file__).resolve().parents[1]


def compiler_for(model, *, depth=1):
    requests = tuple(RequestSpec(f"r{i}", 0.0, model.model_id, 5, 3)
                     for i in range(2))
    trace = RequestTrace(
        TraceProvenance("ci_fixture", "aggregate KV dependency regression"), requests)
    router = (HotsetZipfRouter(17, 2, .9, 1.2)
              if any(layer.is_moe for layer in model.layers) else None)
    timing = LinearTimingProvider(5, 2, 3, 4, moe_routing_fraction=.25)
    return ServeLoopCompiler(models={model.model_id: model}, request_trace=trace,
                             router=router, timing=timing, prefetch_depth=depth)


def schedule(model, batch_id, begin, count, phase, emits):
    return ScheduledBatch(batch_id, model.model_id, tuple(
        BatchSlice(f"r{i}", begin, count, begin, emits, phase) for i in range(2)), 0.)


def ancestors(batch, identifier):
    by_id = {op.id: op for op in batch.operations}
    found = set()
    pending = list(by_id[identifier].dependencies)
    while pending:
        parent = pending.pop()
        if parent not in found:
            found.add(parent)
            pending.extend(by_id[parent].dependencies)
    return found


class CoarseKvCausalityTests(unittest.TestCase):
    def assert_source_contract(self, compiler, batch):
        model = compiler.models[batch.schedule.model_id]
        roles = {op.role: op for op in batch.operations if op.is_barrier}
        seen = set()
        for op in batch.operations:
            # Independent topological-order check, including no dangling edges.
            self.assertTrue(set(op.dependencies) <= seen)
            seen.add(op.id)
        for layer_id in range(model.num_layers):
            source = roles[f"layer/{layer_id}/kv_source_ready"]
            prior = roles[f"layer/{layer_id - 1}/compute"] if layer_id else roles["embedding/complete"]
            weights = {op.id for op in batch.memory_operations
                       if op.role == "attention/weights" and batch.audit[op.id]["layer"] == layer_id}
            self.assertEqual(source.duration_ns, 0.)
            self.assertEqual(set(source.dependencies), weights | {prior.id})
            writes = [op for op in batch.memory_operations
                      if op.role == "attention/kv_write" and batch.audit[op.id]["layer"] == layer_id]
            self.assertEqual(len(writes), len(batch.schedule.slices))
            for write in writes:
                self.assertEqual(write.dependencies, (source.id,))
            # A producer cannot wait on consumers or its own whole layer.
            source_parents = ancestors(batch, source.id)
            forbidden = {op.id for op in writes}
            forbidden.update(roles[f"layer/{layer_id}/{name}"].id
                             for name in ("compute", "memory_complete"))
            for read in batch.memory_operations:
                if read.role == "attention/kv_read" and batch.audit[read.id]["layer"] == layer_id:
                    forbidden.add(read.id)
                    self.assertFalse({op.id for op in writes} & ancestors(batch, read.id))
            self.assertFalse(forbidden & source_parents)
        complete = batch.operations[-1]
        self.assertEqual(complete.role, "batch/complete")
        self.assertEqual(ancestors(batch, complete.id), seen - {complete.id})

    def test_delayed_attention_weights_block_current_kv_publication(self):
        # Semantic regression: make weight availability much later than input.
        # No simulator timing claim; only a DAG precedence witness.
        for fixture in ("ci-dense-model.json", "ci-moe-model.json"):
            compiler = compiler_for(load_model_any(ROOT / "examples" / fixture))
            batch = compiler.compile(schedule(next(iter(compiler.models.values())), 0, 5, 1, "decode", True))
            finish = {}
            start = {}
            for op in batch.operations:
                start[op.id] = max((finish[p] for p in op.dependencies), default=0.)
                duration = op.duration_ns if op.is_barrier else (1000. if op.role == "attention/weights" else 1.)
                finish[op.id] = start[op.id] + duration
            for write in batch.memory_operations:
                if write.role != "attention/kv_write":
                    continue
                layer = batch.audit[write.id]["layer"]
                weights = next(op for op in batch.memory_operations
                               if op.role == "attention/weights" and batch.audit[op.id]["layer"] == layer)
                with self.subTest(fixture=fixture, layer=layer, write=write.id):
                    self.assertGreaterEqual(start[write.id], finish[weights.id])

    def test_public_dense_gqa_moe_gqa_mla_multirequest_and_chunks(self):
        for path in sorted((ROOT / "models").glob("*.json")):
            model = load_model_any(path)
            for depth in (0, 1, 2):
                compiler = compiler_for(model, depth=depth)
                # Full prefill and a separate chunked-prefill/decode sequence.
                cases = [(0, 0, 5, "prefill", True), (1, 0, 2, "prefill", False),
                         (2, 2, 3, "prefill", True), (3, 5, 1, "decode", True),
                         (4, 6, 1, "decode", True)]
                ids = set()
                for args in cases:
                    with self.subTest(model=path.name, depth=depth, case=args):
                        batch = compiler.compile(schedule(model, *args))
                        self.assert_source_contract(compiler, batch)
                        current_ids = {op.id for op in batch.operations}
                        self.assertFalse(ids & current_ids)
                        ids.update(current_ids)
                        timing = compiler.timing.timing_for(model=model, batch=batch.schedule)
                        for i, layer in enumerate(model.layers):
                            by_role = {op.role: op for op in batch.operations if op.is_barrier}
                            cost = by_role[f"layer/{i}/compute"].duration_ns
                            if layer.is_moe:
                                cost += by_role[f"layer/{i}/routing_ready"].duration_ns
                            self.assertEqual(cost, timing.layer_ns[i])
                            weight = next(op for op in batch.memory_operations
                                          if op.role == "attention/weights" and batch.audit[op.id]["layer"] == i)
                            release = i - 1 - depth
                            expected = by_role[f"layer/{release}/compute" if release >= 0 else "embedding/complete"]
                            self.assertEqual(weight.dependencies, (expected.id,))
                        self.assertEqual(by_role["tail/compute"].duration_ns, timing.tail_ns)

    def test_no_attention_weights_uses_input_without_waiting_for_ffn(self):
        for fixture in ("ci-dense-model.json", "ci-moe-model.json"):
            model = load_model_any(ROOT / "examples" / fixture)
            model = replace(model, layers=tuple(replace(layer, attention_weight_bytes=0)
                                                for layer in model.layers))
            compiler = compiler_for(model)
            for begin, count, phase in ((0, 2, "prefill"), (2, 3, "prefill"), (5, 1, "decode")):
                batch = compiler.compile(schedule(model, begin, begin, count, phase, begin != 0))
                self.assert_source_contract(compiler, batch)
                self.assertFalse(any(op.role == "attention/weights" for op in batch.operations))

    def test_mixed_request_payload_routes_and_memory_ids_preserved(self):
        for fixture in ("ci-dense-model.json", "ci-moe-model.json"):
            model = load_model_any(ROOT / "examples" / fixture)
            compiler = compiler_for(model)
            mixed = ScheduledBatch(7, model.model_id, (
                BatchSlice("r0", 5, 1, 5, True, "decode"),
                BatchSlice("r1", 2, 3, 2, True, "prefill")), 0.)
            batch = compiler.compile(mixed)
            self.assert_source_contract(compiler, batch)
            expected = []
            for row in mixed.slices:
                for token in range(row.token_begin, row.token_end):
                    token_id, _ = compiler._token_id(model, compiler.requests[row.request_id], token)
                    expected.append(("embedding/read", "R", model.object_id("embedding"),
                                     token_id * model.embedding_row_bytes, model.embedding_row_bytes))
            for i, layer in enumerate(model.layers):
                expected.append(("attention/weights", "R", model.object_id("attention", i), 0, layer.attention_weight_bytes))
                for row in mixed.slices:
                    expected.append(("attention/kv_write", "W", model.kv_object_id(row.request_id, i),
                                     row.token_begin * layer.kv_bytes_per_token, row.token_count * layer.kv_bytes_per_token))
                for row in mixed.slices:
                    expected.append(("attention/kv_read", "R", model.kv_object_id(row.request_id, i),
                                     0, row.context_tokens_before * layer.kv_bytes_per_token))
                if layer.ffn_weight_bytes:
                    expected.append(("ffn/weights", "R", model.object_id("ffn", i), 0, layer.ffn_weight_bytes))
                if layer.is_moe:
                    expected.append(("moe/router_weights", "R", model.object_id("router", i), 0, layer.router_weight_bytes))
                    if layer.shared_expert_weight_bytes:
                        expected.append(("moe/shared_expert_weights", "R", model.object_id("shared_expert", i), 0, layer.shared_expert_weight_bytes))
                    counts = Counter(expert for row in mixed.slices
                                     for token in range(row.token_begin, row.token_end)
                                     for expert in compiler.router.experts_for(
                                         request=compiler.requests[row.request_id], token_index=token, layer=i, model=model))
                    for expert, tokens in sorted(counts.items()):
                        expected.append(("moe/routed_expert_weights", "R", model.expert_object_id(i, expert), 0, layer.expert_weight_bytes[expert]))
                        actual = next(op for op in batch.memory_operations
                                      if op.object_id == model.expert_object_id(i, expert))
                        self.assertEqual(batch.audit[actual.id]["routed_tokens"], tokens)
                        self.assertEqual(batch.audit[actual.id]["expert"], expert)
                reads = [op for op in batch.memory_operations
                         if batch.audit[op.id].get("layer") == i and op.role == "attention/kv_read"]
                for read in reads:
                    self.assertIn(read.id, ancestors(batch, next(op.id for op in batch.operations
                                                               if op.role == f"layer/{i}/compute")))
            expected.extend([( "final_norm/read", "R", model.object_id("final_norm"), 0, model.final_norm_bytes),
                             ("lm_head/read", "R", model.lm_head_object_id(), 0, model.object_by_id[model.lm_head_object_id()].bytes)])
            actual = [(op.role, op.op, op.object_id, op.offset, op.bytes) for op in batch.memory_operations]
            self.assertEqual(actual, expected)
            self.assertEqual([op.id for op in batch.memory_operations],
                             [f"serve/b7/m{i}" for i in range(len(expected))])


if __name__ == "__main__":
    unittest.main()

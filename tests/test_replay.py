"""A jumped logical replay is the stepped replay, state for state."""

from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_serveloop import _dense, _engine, _pool_spec, _trace  # noqa: E402
from serveloop.contracts import RequestSpec, SchedulerPolicy, ServeLoopError  # noqa: E402
from serveloop.placement import ServeLoopPlacement  # noqa: E402
from serveloop.replay import LogicalReplay, ReplayExecutor  # noqa: E402


def _requests(model, count, *, seed, shared_prefix=0, arrival_spread_ns=0.0):
    """Requests with varied lengths, some sharing a prompt prefix, arriving over time."""
    generator = random.Random(seed)
    requests = []
    for index in range(count):
        prompt = generator.choice((3, 17, 33, 64, 70))
        output = generator.choice((1, 2, 9, 40, 65))
        tokens = tuple(generator.randrange(model.vocab_size) for _ in range(prompt + output - 1))
        if shared_prefix and index % 2:
            tokens = (7,) * min(shared_prefix, prompt) + tokens[min(shared_prefix, prompt):]
        requests.append(RequestSpec(f"r{index}", generator.uniform(0.0, arrival_spread_ns), model.model_id,
                                    prompt, output, tokens))
    return tuple(sorted(requests, key=lambda request: (request.arrival_ns, request.request_id)))


def _replay(model, requests, spec, policy, *, iteration_ns=10.0):
    placement = ServeLoopPlacement(models={model.model_id: model}, spec=spec)
    executor = ReplayExecutor(placement, iteration_ns)
    engine = _engine({model.model_id: model}, _trace(*requests), executor, policy)
    return LogicalReplay(engine, executor), placement


def _state(replay, placement):
    engine = replay.engine
    live = {
        request_id: placement.request_blocks(request_id)
        for request_id, _, _ in engine.running()
    }
    return {
        "iterations": engine.iterations,
        "frontier_ns": replay.executor.frontier_ns,
        "progress": engine.request_progress(),
        "running": engine.running(),
        "live_kv": live,
        "prefix": placement.prefix_receipt(),
        "retained": placement.retained_prefix(),
        "layout": placement.frontier_ns_independent_state,
        "receipt": placement.receipt(),
    }


class ReplayTests(unittest.TestCase):
    def assert_replays_agree(self, model, requests, spec, policy, *, limits=(None, 1, 2, 3, 5, 8)):
        stepped, stepped_placement = _replay(model, requests, spec, policy)
        states = {}
        while not stepped.finished:
            stepped.step()
            states[stepped.iterations] = _state(stepped, stepped_placement)
        self.assertGreater(stepped.iterations, 0)
        jumped_somewhere = False
        for limit in limits:
            jumped, placement = _replay(model, requests, spec, policy)
            advances = 0
            while not jumped.finished:
                _, count = jumped.advance(limit)
                advances += 1
                jumped_somewhere |= count > 1
                self.assertEqual(_state(jumped, placement), states[jumped.iterations],
                                 f"limit {limit}: state differs after iteration {jumped.iterations}")
            self.assertEqual(jumped.iterations, stepped.iterations)
            if limit is None:
                self.assertLess(advances, stepped.iterations)
        self.assertTrue(jumped_somewhere)
        return stepped.iterations

    def test_pool_with_room_jumps_through_decode_and_block_boundaries(self):
        model = _dense(layers=3)
        policy = SchedulerPolicy(max_batch_requests=4, max_batch_tokens=64, prefill_chunk_tokens=16)
        requests = _requests(model, 12, seed=1, shared_prefix=32, arrival_spread_ns=200.0)
        spec = replace(_pool_spec(model, 400), prefix_cache_bytes=40 * 4096)
        self.assert_replays_agree(model, requests, spec, policy)

    def test_tight_pool_leaves_evictions_and_recompute_to_real_iterations(self):
        model = _dense(layers=2)
        policy = SchedulerPolicy(max_batch_requests=3, max_batch_tokens=40, prefill_chunk_tokens=8)
        requests = _requests(model, 10, seed=2, shared_prefix=16)
        spec = replace(_pool_spec(model, 40), prefix_cache_bytes=12 * 4096)
        self.assert_replays_agree(model, requests, spec, policy)

    def test_cold_tier_migrations_match(self):
        model = _dense(layers=2)
        policy = SchedulerPolicy(max_batch_requests=3, max_batch_tokens=40, prefill_chunk_tokens=8)
        requests = _requests(model, 10, seed=3, arrival_spread_ns=50.0)
        spec = replace(_pool_spec(model, 36, cold="hbf", hbf_capacity_bytes=4096 * 400, weight_tier="hbf"),
                       prefix_cache_bytes=8 * 4096)
        from test_serveloop import _GEOMETRY

        def build(model, requests, spec, policy, *, iteration_ns=10.0):
            placement = ServeLoopPlacement(models={model.model_id: model}, spec=spec, hbf_geometry=_GEOMETRY)
            executor = ReplayExecutor(placement, iteration_ns)
            engine = _engine({model.model_id: model}, _trace(*requests), executor, policy)
            return LogicalReplay(engine, executor), placement

        global _replay
        original = _replay
        _replay = build
        try:
            self.assert_replays_agree(model, requests, spec, policy)
        finally:
            _replay = original

    def test_prefix_ttl_caps_a_jump_before_the_expiry(self):
        model = _dense(layers=2)
        policy = SchedulerPolicy(max_batch_requests=2, max_batch_tokens=32, prefill_chunk_tokens=16)
        requests = _requests(model, 6, seed=4, shared_prefix=32)
        spec = replace(_pool_spec(model, 300), prefix_cache_bytes=30 * 4096, prefix_cache_ttl_ns=75.0)
        self.assert_replays_agree(model, requests, spec, policy)

    def test_thrashing_prefix_cache_is_still_exact(self):
        # Every running request's prompt exceeds the budget together, so each
        # publication evicts and re-inserts; the replay cannot settle and steps.
        model = _dense(layers=1)
        policy = SchedulerPolicy(max_batch_requests=3, max_batch_tokens=64, prefill_chunk_tokens=16)
        requests = tuple(RequestSpec(f"r{i}", 0.0, model.model_id, 64, 30,
                                     tuple((i * 7 + t) % model.vocab_size for t in range(93)))
                         for i in range(5))
        spec = replace(_pool_spec(model, 120), prefix_cache_bytes=7 * 4096)
        self.assert_replays_agree(model, requests, spec, policy)

    def test_horizon_and_advance_contracts(self):
        model = _dense(layers=1)
        policy = SchedulerPolicy(max_batch_requests=2, max_batch_tokens=32, prefill_chunk_tokens=16)
        requests = (RequestSpec("a", 0.0, model.model_id, 16, 20, (1,) * 35),
                    RequestSpec("b", 0.0, model.model_id, 16, 5, (2,) * 20),
                    RequestSpec("late", 95.0, model.model_id, 16, 2, (3,) * 17))
        replay, placement = _replay(model, requests, _pool_spec(model, 100), policy)
        engine = replay.engine
        first = replay.step()  # prefill of a and b, one chunk each
        self.assertEqual(engine.decode_horizon(first, frontier_ns=10.0, iteration_ns=10.0), 0)
        second = replay.step()  # both decode
        self.assertEqual(second.kind, "decode")
        # b has emitted 2 of 5 tokens; the batch is full, so the late arrival does not bound the horizon.
        self.assertEqual(engine.decode_horizon(second, frontier_ns=20.0, iteration_ns=10.0), 3)
        with self.assertRaises(ServeLoopError):
            engine.advance_decode(second, 4, frontier_ns=20.0, iteration_ns=10.0)
        applied = engine.advance_decode(second, 3, frontier_ns=20.0, iteration_ns=10.0)
        self.assertEqual(applied, 3)
        self.assertEqual(engine.iterations, 5)
        progress = {row["request_id"]: row for row in engine.request_progress()}
        self.assertEqual(progress["b"]["completion_ns"], 50.0)
        self.assertEqual(progress["a"]["emitted"], 5)
        # With a free slot, the horizon stops where the late request's arrival admits it:
        # the fifth iteration after the frontier of 60 starts at 100, past the arrival at 95.
        schedule = replay.step()  # a alone; late has not arrived at 50
        self.assertEqual([item.request_id for item in schedule.slices], ["a"])
        self.assertEqual(replay.executor.frontier_ns, 60.0)
        self.assertEqual(engine.decode_horizon(schedule, frontier_ns=60.0, iteration_ns=10.0), 4)
        while not replay.finished:
            replay.advance()
        progress = {row["request_id"]: row for row in engine.request_progress()}
        self.assertEqual(progress["late"]["first_token_ns"], 110.0)


if __name__ == "__main__":
    unittest.main()

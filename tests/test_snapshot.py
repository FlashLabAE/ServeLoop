"""A decode snapshot resumes the replayed state: contexts hit, holes stay, order holds."""

from dataclasses import replace
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_serveloop import PlacementExecutor, _dense, _engine, _pool_spec, _trace  # noqa: E402
from serveloop.contracts import MemoryOnlyTimingProvider, RequestSpec, SchedulerPolicy, ServeLoopError  # noqa: E402
from serveloop.placement import ServeLoopPlacement  # noqa: E402
from serveloop.prefix_image import install_prefix_image, load_prefix_image, write_prefix_image  # noqa: E402
from serveloop.snapshot import SnapshotSettings, take_snapshot  # noqa: E402


def _stream(model, count, *, seed):
    generator = random.Random(seed)
    requests = []
    for index in range(count):
        prompt = generator.choice((40, 64, 96))
        output = generator.choice((30, 60))
        tokens = [generator.randrange(model.vocab_size) for _ in range(prompt + output - 1)]
        if index % 3 == 0:
            tokens[:32] = [5] * 32  # a shared prefix of two whole blocks
        requests.append(RequestSpec(f"r{index}", float(index * 5), model.model_id, prompt, output, tuple(tokens)))
    return tuple(requests)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.model = _dense(layers=2)
        self.models = {self.model.model_id: self.model}
        self.policy = SchedulerPolicy(max_batch_requests=4, max_batch_tokens=64, prefill_chunk_tokens=16)
        self.spec = replace(_pool_spec(self.model, 300), prefix_cache_bytes=48 * 4096)
        self.requests = _stream(self.model, 24, seed=11)
        self.settings = SnapshotSettings(iteration_ns=10.0, suffix_tokens=4, output_tokens=3, seed=1)

    def resumed_spec(self, snapshot):
        """The replay's placement with a prefix budget that covers the image (contexts included)."""
        needed = snapshot.metadata["image"]["minimum_prefix_cache_bytes"]
        return replace(self.spec, prefix_cache_bytes=max(self.spec.prefix_cache_bytes, needed))

    def snapshot(self):
        return take_snapshot(
            models=self.models, request_trace=_trace(*self.requests), router=None, placement_spec=self.spec,
            hbf_geometry=None, hbm_stripe_bytes=None, policy=self.policy, timing=MemoryOnlyTimingProvider(),
            prefetch_depth=1, settings=self.settings)

    def test_resumed_requests_hit_their_whole_context_and_nothing_else_moves(self):
        snapshot = self.snapshot()
        selection = snapshot.metadata["selection"]
        self.assertGreater(selection["candidates"], 0)
        sequences = [row for row in snapshot.metadata["sequences"] if row["resumed"] is not None]
        self.assertEqual(len(sequences), self.policy.max_batch_requests)
        for row in sequences:
            self.assertEqual(row["context_tokens"], row["processed_tokens"] // 16 * 16)
        placement = ServeLoopPlacement(models=self.models, spec=self.resumed_spec(snapshot))
        installed = install_prefix_image(placement, snapshot.image)
        self.assertEqual(installed["blocks"], snapshot.image.blocks)
        self.assertEqual(placement.prefix_receipt()["resident_bytes"], installed["bytes"])
        # The resumed requests reuse exactly their context; only the suffix is prefilled.
        executor = PlacementExecutor(placement, latency_ns=10.0)
        engine = _engine(self.models, snapshot.requests, executor, self.policy, timing=MemoryOnlyTimingProvider())
        result = engine.run()
        contexts = {row["resumed"]: row["context_tokens"] for row in sequences}
        first = result["batches"][0]["schedule"]["slices"]
        self.assertEqual({item["request_id"] for item in first}, set(contexts))
        for item in first:
            self.assertEqual(item["phase"], "prefill")
            self.assertEqual(item["token_begin"], contexts[item["request_id"]])
            self.assertEqual(item["token_count"], self.settings.suffix_tokens)
        for row in result["requests"]:
            self.assertEqual(row["prefix_hit_tokens"], contexts[row["request_id"]])
        self.assertEqual(result["summary"]["iterations"], self.settings.output_tokens)
        self.assertEqual(result["summary"]["requests"], len(contexts))

    def test_image_round_trips_through_json_and_reproduces_the_layout(self):
        snapshot = self.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix-image.json"
            write_prefix_image(path, snapshot.image)
            loaded = load_prefix_image(path)
        self.assertEqual(loaded, snapshot.image)
        self.assertEqual(loaded.digest, snapshot.image.digest)
        first = ServeLoopPlacement(models=self.models, spec=self.resumed_spec(snapshot))
        second = ServeLoopPlacement(models=self.models, spec=self.resumed_spec(snapshot))
        install_prefix_image(first, snapshot.image)
        install_prefix_image(second, loaded)
        self.assertEqual(first.retained_prefix(), second.retained_prefix())
        self.assertEqual(first.receipt()["final_state"], second.receipt()["final_state"])
        blocks = {block for entry in snapshot.image.entries for block in entry.blocks}
        # Holes between the image's blocks stay free: the next allocation takes the lowest hole.
        holes = sorted(set(range(max(blocks) + 1)) - blocks)
        if holes:
            self.assertEqual(first._hot.allocate(1), [holes[0]])

    def test_images_are_checked_against_the_placement(self):
        snapshot = self.snapshot()
        other = ServeLoopPlacement(models=self.models, spec=replace(self.spec, kv_block_tokens=32))
        with self.assertRaisesRegex(ServeLoopError, "token blocks"):
            install_prefix_image(other, snapshot.image)
        small = ServeLoopPlacement(models=self.models, spec=replace(self.spec, prefix_cache_bytes=4096))
        with self.assertRaisesRegex(ServeLoopError, "budget"):
            install_prefix_image(small, snapshot.image)
        used = ServeLoopPlacement(models=self.models, spec=self.spec)
        used.admit_request(self.requests[0])
        with self.assertRaisesRegex(ServeLoopError, "empty"):
            install_prefix_image(used, snapshot.image)

    def test_a_stream_without_token_ids_cannot_be_snapshotted(self):
        requests = tuple(replace(request, token_ids=None) for request in self.requests)
        with self.assertRaisesRegex(ServeLoopError, "token ids"):
            take_snapshot(
                models=self.models, request_trace=_trace(*requests), router=None, placement_spec=self.spec,
                hbf_geometry=None, hbm_stripe_bytes=None, policy=self.policy, timing=MemoryOnlyTimingProvider(),
                prefetch_depth=1, settings=self.settings)


if __name__ == "__main__":
    unittest.main()

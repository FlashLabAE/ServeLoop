"""Snapshots resume and sampled windows estimate, end to end against the engine."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serveloop.io import write_json_atomic  # noqa: E402


def _requests(directory: Path, *, count: int, prompt: float, output: float) -> Path:
    path = directory / "requests.json"
    write_json_atomic(path, {
        "schema": {"name": "serveloop.synthetic_requests", "version": 1},
        "request_count": count, "arrival_rate_per_second": 1000000.0,
        "prompt_lognormal_mean_tokens": prompt, "prompt_lognormal_sigma": 0.2,
        "output_lognormal_mean_tokens": output, "output_lognormal_sigma": 0.2,
        "model_probabilities": {"dense-ci": 1.0}, "seed": 3, "first_arrival_policy": "zero",
        "shared_prefix_tokens": 16, "shared_prefix_groups": 2, "prefix_reuse_probability": 0.5,
    })
    return path


def _run_config(directory: Path) -> Path:
    path = directory / "run-config.json"
    write_json_atomic(path, {
        "schema": {"name": "serveloop.run_config", "version": 2},
        "scheduler": {"max_batch_requests": 4, "max_batch_tokens": 64, "prefill_chunk_tokens": 16},
        "timing": {"type": "memory_only", "prefetch_depth": 1}, "compute": None,
    })
    return path


class LongStreamTests(unittest.TestCase):
    simulator: Path | None = None

    def command(self, subcommand: str, *arguments: str) -> list[str]:
        return [
            sys.executable, "-B", "-m", "serveloop", subcommand,
            "--model", str(ROOT / "examples/ci-dense-model.json"),
            "--system", str(ROOT / "configs/systems/eight-stack-baseline.cfg"),
            "--overlay", str(ROOT / "configs/simulation-session-mini.cfg"),
            "--placement", "weights-hbf-kv-hbm", "--simulator", str(self.simulator), *arguments,
        ]

    def run_serveloop(self, *command: str) -> subprocess.CompletedProcess:
        executed = subprocess.run(list(command), cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(executed.returncode, 0, executed.stderr)
        return executed

    def result(self, output_root: Path) -> dict:
        results = list(output_root.glob("*/result.json"))
        self.assertEqual(len(results), 1)
        return json.loads(results[0].read_text(encoding="utf-8"))

    def test_snapshot_resumes_every_context_and_times_only_the_outputs(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = _requests(root, count=12, prompt=40.0, output=20.0)
            run_config = _run_config(root)
            self.run_serveloop(*self.command(
                "snapshot", "--requests", str(requests), "--run-config", str(run_config),
                "--prefix-cache-bytes", "65536", "--iteration-ns", "1000", "--suffix-tokens", "4",
                "--output-tokens", "3", "--out", str(root / "snapshot")))
            snapshot = json.loads((root / "snapshot/snapshot.json").read_text(encoding="utf-8"))
            sequences = [row for row in snapshot["sequences"] if row["resumed"] is not None]
            self.assertEqual(len(sequences), 4)
            self.assertEqual(snapshot["selection"]["rule"], "median_context_steady_state_v1")
            executed = self.run_serveloop(*self.command(
                "run", "--requests", str(root / "snapshot/requests.json"),
                "--prefix-image", str(root / "snapshot/prefix-image.json"), "--run-config", str(run_config),
                "--prefix-cache-bytes", str(snapshot["image"]["minimum_prefix_cache_bytes"]),
                "--out", str(root / "resumed")))
            result = self.result(root / "resumed")
            serving = result["serveloop"]
            self.assertEqual(serving["summary"]["iterations"], 3)
            self.assertEqual(serving["scheduler"]["iterations_by_kind"], {"prefill": 1, "decode": 2, "mixed": 0})
            contexts = {row["resumed"]: row["context_tokens"] for row in sequences}
            for row in serving["requests"]:
                self.assertEqual(row["prefix_hit_tokens"], contexts[row["request_id"]])
            for item in serving["batches"][0]["schedule"]["slices"]:
                self.assertEqual(item["token_begin"], contexts[item["request_id"]])
                self.assertEqual(item["token_count"], 4)
            self.assertEqual(result["prefix_image"]["entries"], snapshot["image"]["entries"])
            self.assertIn("timing model: memory_only", executed.stdout)

    def test_sampled_window_times_a_share_and_reports_the_estimate(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = _requests(root, count=12, prompt=40.0, output=20.0)
            run_config = _run_config(root)
            executed = self.run_serveloop(*self.command(
                "run", "--requests", str(requests), "--run-config", str(run_config),
                "--prefix-cache-bytes", "65536", "--sample-rate", "0.3", "--sample-seed", "1",
                "--out", str(root / "sampled")))
            self.assertIn("sampled: timed", executed.stdout)
            result = self.result(root / "sampled")
            report = result["sampling"]
            serving = result["serveloop"]
            self.assertEqual(report["batches"], serving["summary"]["iterations"])
            self.assertLess(report["timed_batches"], report["batches"])
            self.assertEqual(report["plan"], {"rate": 0.3, "pilot": 2, "warmup": 1, "seed": 1})
            for row in report["strata"]:
                self.assertEqual(row["batches"], row["timed"] + row["skipped"])
                self.assertGreaterEqual(row["timed"], min(2, row["batches"]))
                self.assertGreaterEqual(row["counted"], 1)
            decisions = [batch["physical_completion"]["sampling"] for batch in serving["batches"]]
            self.assertEqual(sum(not decision["timed"] for decision in decisions),
                             sum(row["skipped"] for row in report["strata"]))
            elapsed = sum(batch["finish_ns"] - batch["batch_origin_ns"] for batch in serving["batches"])
            self.assertAlmostEqual(elapsed, report["clock_ns"], delta=1e-6 * max(1.0, elapsed))
            for batch, decision in zip(serving["batches"], decisions):
                if not decision["timed"]:
                    self.assertGreaterEqual(batch["finish_ns"] - batch["batch_origin_ns"], decision["estimate_ns"])
            # The same seed reproduces the same decisions.
            again = self.run_serveloop(*self.command(
                "run", "--requests", str(requests), "--run-config", str(run_config),
                "--prefix-cache-bytes", "65536", "--sample-rate", "0.3", "--sample-seed", "1",
                "--out", str(root / "again")))
            self.assertEqual(self.result(root / "again")["sampling"], report)
            self.assertEqual(again.returncode, 0)
            # Rate 1 times everything and the estimate is the window itself.
            self.run_serveloop(*self.command(
                "run", "--requests", str(requests), "--run-config", str(run_config),
                "--prefix-cache-bytes", "65536", "--sample-rate", "1", "--sample-pilot", "0",
                "--out", str(root / "exact")))
            exact = self.result(root / "exact")["sampling"]
            self.assertEqual(exact["timed_batches"], exact["batches"])
            self.assertAlmostEqual(exact["estimate_ns"], exact["clock_ns"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path)
    args, remaining = parser.parse_known_args()
    LongStreamTests.simulator = None if args.simulator is None else args.simulator.resolve()
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    raise SystemExit(0 if program.result.wasSuccessful() else 1)

from dataclasses import replace
from pathlib import Path
import unittest

from serveloop.compiler import ServeLoopCompiler
from serveloop.coarse_catalog import CatalogCoverageCompiler
from serveloop.coarse_analytic import AnalyticCoverageCompiler, build_profile
from serveloop.coarse_coverage import CoarseCoverageCompiler, CoarseCoverageProfile
from serveloop.coarse_l2 import CoarseL2Compiler
from serveloop.contracts import BatchSlice, ScheduledBatch, LinearTimingProvider, RooflineTimingProvider, ServeLoopError
from serveloop.catalog import load_model_any
from serveloop.io import load_request_trace
from test_coarse_catalog import setup

ROOT = Path(__file__).resolve().parents[1]


def budgets(result):
    return {o.id: (o.role, o.duration_ns) for o in result.operations if o.is_barrier}


class ExtendedScopeTests(unittest.TestCase):
    def test_chunked_prefill_all_models_preserves_ledger_and_compute(self):
        for path in sorted((ROOT/'models').glob('*.json')):
            _, model, kw, _ = setup(path)
            batches = [ScheduledBatch(i, model.model_id,
                (BatchSlice('request', begin, n, begin, emits, phase),), 0.)
                for i, (begin, n, emits, phase) in enumerate([
                    (0, 5, False, 'prefill'), (5, 5, False, 'prefill'),
                    (10, 6, True, 'prefill'), (16, 1, True, 'decode')])]
            for timing in (None, RooflineTimingProvider(200, .5),
                           LinearTimingProvider(10, 2, 5, 3, moe_routing_fraction=.2)):
                opts = dict(kw, **({'timing': timing} if timing else {}))
                original = ServeLoopCompiler(**opts)
                enhanced = CatalogCoverageCompiler(**opts)
                for batch in batches:
                    with self.subTest(model=path.name, timing=timing, batch=batch.batch_id):
                        a = original.compile(batch); b = enhanced.compile(batch)
                        self.assertEqual(budgets(a), budgets(b))
                        weights = lambda x: [(o.id, o.op, o.object_id, o.offset, o.bytes)
                            for o in x.memory_operations if o.object_id.startswith('model/')]
                        self.assertEqual(weights(a), weights(b))
                        writes = lambda x: [(o.object_id, o.offset, o.bytes) for o in x.memory_operations
                            if o.role == 'attention/kv_write']
                        self.assertEqual(writes(a), writes(b))
                        s = batch.slices[0]
                        if s.phase == 'prefill':
                            reads = lambda x: [(o.object_id, o.offset, o.bytes) for o in x.memory_operations
                                if o.role == 'attention/kv_read']
                            self.assertEqual(reads(a), reads(b))
                        calls = {b.audit[o.id].get('call') for o in b.operations}
                        if not s.emits_output:
                            self.assertNotIn('lm_head', calls)
                            self.assertNotIn('last_token_select', calls)
                            self.assertNotIn('logits_cast', calls)
                            self.assertIn('final_add_rmsnorm', calls)
                        else:
                            self.assertIn('lm_head', calls)
                        if timing and timing.timing_model == 'linear':
                            audit = b.audit[b.operations[-1].id]['coarse_compute']
                            self.assertEqual(audit['provider'], timing.canonical())
                            self.assertFalse(audit['calibrated'])

    def test_chunk_context_does_not_expand_temporary_ranges(self):
        model = load_model_any(ROOT/'examples/coarse-coverage/model.json')
        a = build_profile(model, '0'*64, 'prefill', 8, 0)
        b = build_profile(model, '0'*64, 'prefill', 8, 1024)
        for key in ('embedding', 'layers', 'tail'):
            self.assertEqual(a.phases['prefill'][key], b.phases['prefill'][key])
        self.assertEqual(a.workspace_bytes, b.workspace_bytes)

    def test_chunked_cache_lifecycle_drains_only_at_request_end(self):
        _, model, kw, _ = setup(ROOT/'models/llama31-8b-w8-kv-bf16.json')
        cached = CoarseL2Compiler(CatalogCoverageCompiler(**kw), 41943040)
        for i, (begin, n, emits, phase) in enumerate([
                (0, 8, False, 'prefill'), (8, 8, True, 'prefill'), (16, 1, True, 'decode')]):
            batch = ScheduledBatch(i, model.model_id,
                (BatchSlice('request', begin, n, begin, emits, phase),), 0.)
            result = cached.compile(batch)
            has_drain = any(result.audit[o.id].get('role') == 'coarse_l2/final_drain'
                            for o in result.operations)
            if i == 2:
                self.assertTrue(has_drain)
                self.assertEqual(cached.cache.summary()['dirty_sectors'], 0)
            else:
                self.assertFalse(has_drain)
        self.assertEqual(len(cached.records), 3)

    def test_captured_profile_keeps_fixed_input_and_timing_boundary(self):
        data = ROOT/'examples/coarse-coverage'
        model = load_model_any(data/'model.json'); trace = load_request_trace(data/'requests.json')
        kw = dict(models={model.model_id: model}, request_trace=trace)
        profile = CoarseCoverageProfile.load(data/'qwen2.5-1.5b-sglang-bf16-p16-d1.json')
        captured = CoarseCoverageCompiler(**kw, coverage_profile=profile)
        non_emitting = ScheduledBatch(0, model.model_id,
            (BatchSlice('observed-request', 0, 16, 0, False, 'prefill'),), 0.)
        with self.assertRaisesRegex(ServeLoopError, 'output-emitting'):
            captured.compile(non_emitting)
        with self.assertRaisesRegex(ServeLoopError, 'captured coverage requires memory_only'):
            CoarseCoverageCompiler(**kw, coverage_profile=profile,
                timing=LinearTimingProvider(1, 1, 1, 1))
        # Two requests in one batch still require a distinct batching design.
        second = replace(trace.requests[0], request_id='second')
        compiler = AnalyticCoverageCompiler(models={model.model_id: model},
            request_trace=replace(trace, requests=(*trace.requests, second)))
        batch = ScheduledBatch(0, model.model_id, (
            BatchSlice('observed-request', 0, 16, 0, True, 'prefill'),
            BatchSlice('second', 0, 16, 0, True, 'prefill')), 0.)
        with self.assertRaisesRegex(ServeLoopError, 'B1'):
            compiler.compile(batch)


if __name__ == '__main__':
    unittest.main()

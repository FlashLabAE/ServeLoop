"""Generated producer causality without a new weight scheduling policy."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from serveloop.compiler import ServeLoopCompiler
from serveloop.coarse_catalog import CatalogCoverageCompiler
from serveloop.coarse_coverage import CoarseCoverageCompiler, CoarseCoverageProfile
from serveloop.coarse_causality import bind_producers
from serveloop.contracts import BatchSlice, ScheduledBatch, LinearTimingProvider, RooflineTimingProvider, ServeLoopError
from test_coarse_catalog import setup

ROOT = Path(__file__).resolve().parents[1]


def ancestors(result, identifier):
    ops = {o.id: o for o in result.operations}; seen = set(); pending = list(ops[identifier].dependencies)
    while pending:
        parent = pending.pop()
        if parent not in seen:
            seen.add(parent); pending.extend(ops[parent].dependencies)
    return seen


class ProfileCausalityTests(unittest.TestCase):
    def check_producers(self, result, layer):
        meta = result.audit
        joins = [o for o in result.operations if o.role == 'consumer/complete' and meta[o.id]['layer'] == layer]
        semantic = lambda kind: next(o for o in joins if meta[o.id]['semantic']['kind'] == kind)
        source = semantic('kv_source'); attention = semantic('attention')
        write = next(o for o in result.memory_operations if o.role == 'attention/kv_write' and meta[o.id]['layer'] == layer)
        self.assertIn(source.id, ancestors(result, write.id))
        self.assertIn(source.id, ancestors(result, attention.id))
        for read in result.memory_operations:
            if meta[read.id].get('layer') != layer:
                continue
            if read.role == 'attention/kv_read':
                self.assertNotIn(write.id, ancestors(result, read.id))
                self.assertIn(read.id, ancestors(result, attention.id))
            if read.role == 'attention/kv_read_current':
                self.assertIn(write.id, ancestors(result, read.id))
                self.assertIn(read.id, ancestors(result, attention.id))
                self.assertEqual((read.object_id, read.offset, read.bytes), (write.object_id, write.offset, write.bytes))
        if any(meta[o.id]['semantic']['kind'] == 'dispatch' for o in joins):
            route = next(o for o in result.operations if o.role == f'layer/{layer}/routing_ready')
            self.assertIn(semantic('route_selection').id, ancestors(result, route.id))
            self.assertIn(route.id, ancestors(result, semantic('dispatch').id))

    def test_five_models_all_phases_keep_weight_parents_payloads_and_compute(self):
        for path in sorted((ROOT/'models').glob('*.json')):
            _, model, kw, _ = setup(path)
            for timing in [None, RooflineTimingProvider(200, .5), LinearTimingProvider(10, 2, 5, 3, moe_routing_fraction=.2)]:
                opts = dict(kw, **({'timing': timing} if timing else {}))
                for phase, n, context, emits in [('prefill', 16, 0, True), ('prefill', 4, 8, False), ('decode', 1, 16, True)]:
                    with self.subTest(model=path.name, phase=phase, context=context, timing=timing):
                        batch = ScheduledBatch(0, model.model_id, (BatchSlice('request', context, n, context, emits, phase),), 0.)
                        original = ServeLoopCompiler(**opts).compile(batch)
                        result = CatalogCoverageCompiler(**opts).compile(batch)
                        weights = lambda r: {o.id: (o.op, o.object_id, o.offset, o.bytes, o.dependencies)
                            for o in r.memory_operations if o.role.endswith('weights')}
                        self.assertEqual(weights(original), weights(result))
                        durations = lambda r: {o.id: (o.role, o.duration_ns) for o in r.operations
                            if o.is_barrier and o.role != 'consumer/complete'}
                        self.assertEqual(durations(original), durations(result))
                        self.assertTrue(all(o.duration_ns == 0 for o in result.operations if o.role == 'consumer/complete'))
                        self.assertEqual(original.audit_summary()['moe_expert_routes'], result.audit_summary()['moe_expert_routes'])
                        self.assertEqual(result.audit[result.operations[-1].id]['coarse_schedule']['policy'], 'legacy-prefetch')
                        seen = set()
                        for op in result.operations:
                            self.assertTrue(set(op.dependencies) <= seen); seen.add(op.id)
                        self.assertEqual(ancestors(result, result.operations[-1].id), seen - {result.operations[-1].id})
                        for layer in sorted({0, min(3, model.num_layers-1), model.num_layers-1}):
                            self.check_producers(result, layer)

    def test_missing_corrupt_semantics_and_memory_references_are_rejected(self):
        _, model, kw, batches = setup(ROOT/'models/qwen3-8b-bf16-kv-bf16.json')
        compiler = CatalogCoverageCompiler(**kw); batch = batches[0]
        phase = compiler.coverage_for_batch(batch).phases['prefill']
        result = compiler.compile(batch)
        cases = []
        bad = deepcopy(phase); bad.pop('consumer_contract'); cases.append(bad)
        bad = deepcopy(phase); bad['layers'][0][0]['semantic']['kind'] = 'unknown'; cases.append(bad)
        bad = deepcopy(phase)
        for call in bad['layers'][0]:
            if call['semantic']['kind'] == 'kv_source': call['semantic']['kind'] = 'operation'
        cases.append(bad)
        for bad in cases:
            with self.assertRaises(ServeLoopError): bind_producers(result, bad)
        join = next(o for o in result.operations if o.role == 'consumer/complete')
        audit = deepcopy(result.audit); audit[join.id]['memory_ids'] = ['missing']
        with self.assertRaisesRegex(ServeLoopError, 'memory call reference'):
            bind_producers(replace(result, audit=audit), phase)
        # Generated compilers may not silently fall back to the captured floor.
        profile = deepcopy(compiler.coverage_for_batch(batch).document)
        profile['phases']['prefill'].pop('consumer_contract')
        compiler.coverage_for_batch = lambda _: CoarseCoverageProfile(profile)
        with self.assertRaisesRegex(ServeLoopError, 'explicit model semantic'):
            compiler.compile(batch)

    def test_opaque_operator_names_and_ideal_temporaries_keep_producers(self):
        _, model, kw, batches = setup(ROOT/'models/deepseek-v3-fp8-kv-bf16.json')
        for cache_bound in ['off', 'ideal-temporaries']:
            compiler = CatalogCoverageCompiler(**kw, cache_bound=cache_bound)
            document = deepcopy(compiler.coverage_for_batch(batches[0]).document)
            for phase in document['phases'].values():
                for calls in phase['layers'] + phase['routing_layers']:
                    for i, call in enumerate(calls): call['id'] = f'opaque-{i}'; call['operator'] = 'opaque'
            profile = CoarseCoverageProfile(document)
            compiler.coverage_for_batch = lambda _: profile
            result = compiler.compile(batches[0]); self.check_producers(result, 3)

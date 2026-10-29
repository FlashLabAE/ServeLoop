"""Optional weight release must not replace causal dependencies or defaults."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from serveloop.coarse_catalog import CatalogCoverageCompiler
from serveloop.coarse_coverage import CoarseCoverageCompiler
from serveloop.coarse_schedule import apply_weight_schedule
from serveloop.contracts import BatchSlice, ScheduledBatch, ServeLoopError, LinearTimingProvider
from serveloop.io import run_config_from_dict
from serveloop.run import _resolve_coarse_run_config, default_run_config, build_parser
from test_coarse_catalog import setup
from test_coarse_profile_causality import ancestors

ROOT = Path(__file__).resolve().parents[1]


class OptionalScheduleTests(unittest.TestCase):
    def test_default_and_explicit_legacy_are_identical(self):
        for path in sorted((ROOT/'models').glob('*.json')):
            _, model, kw, batches = setup(path)
            for batch in batches:
                default = CatalogCoverageCompiler(**kw).compile(batch)
                explicit = CatalogCoverageCompiler(**kw, schedule_policy='legacy-prefetch').compile(batch)
                self.assertEqual(default.operations, explicit.operations)
                self.assertEqual(default.audit, explicit.audit)
                # Legacy does not even reorder the already causal graph.
                again = apply_weight_schedule(default, CatalogCoverageCompiler(**kw).coverage_for_batch(batch).phases[batch.slices[0].phase], 'legacy-prefetch')
                self.assertIs(default.operations, again.operations)

    def test_consumer_changes_only_weight_parents_and_preserves_causality(self):
        for path in sorted((ROOT/'models').glob('*.json')):
            _, model, kw, _ = setup(path)
            for phase,n,ctx,emits in [('prefill',16,0,True),('prefill',4,8,False),('decode',1,16,True)]:
                with self.subTest(model=path.name, phase=phase, context=ctx):
                    batch = ScheduledBatch(0, model.model_id, (BatchSlice('request',ctx,n,ctx,emits,phase),),0.)
                    opts=dict(kw,timing=LinearTimingProvider(10,2,5,3,moe_routing_fraction=.2))
                    before=CatalogCoverageCompiler(**opts).compile(batch)
                    result=CatalogCoverageCompiler(**opts,schedule_policy='consumer-order').compile(batch)
                    old={o.id:o for o in before.operations};new={o.id:o for o in result.operations}
                    self.assertEqual(old.keys(),new.keys())
                    for identifier,o in old.items():
                        x=new[identifier]
                        self.assertEqual((o.op,o.object_id,o.offset,o.bytes,o.duration_ns,o.role),
                                         (x.op,x.object_id,x.offset,x.bytes,x.duration_ns,x.role))
                        if not o.role.endswith('weights') or o.role=='moe/routed_expert_weights':
                            self.assertEqual(o.dependencies,x.dependencies)
                    seen=set()
                    for o in result.operations:
                        self.assertTrue(set(o.dependencies)<=seen);seen.add(o.id)
                    self.assertEqual(ancestors(result,result.operations[-1].id),seen-{result.operations[-1].id})
                    layer=0
                    first=next(o.id for o in result.operations if o.role=='embedding/complete')
                    att=next(o for o in result.memory_operations if o.role=='attention/weights')
                    self.assertEqual(att.dependencies,(first,))
                    for layer in sorted({0,min(3,model.num_layers-1),model.num_layers-1}):
                        joins=[o for o in result.operations if o.role=='consumer/complete' and result.audit[o.id]['layer']==layer]
                        att=next(o for o in result.memory_operations if o.role=='attention/weights' and result.audit[o.id]['layer']==layer)
                        expected=next(o.id for o in result.operations if o.role==f'layer/{layer-1}/compute') if layer else first
                        self.assertEqual(att.dependencies,(expected,))
                        post=next(o for o in joins if result.audit[o.id]['semantic']['kind']=='post_norm')
                        position=result.audit[post.id]['consumer_position']
                        output=next(o for o in joins if result.audit[o.id]['consumer_position']==[layer,position[1],position[2]-1])
                        for weight in result.memory_operations:
                            if weight.role=='ffn/weights' and result.audit[weight.id]['layer']==layer:
                                self.assertEqual(weight.dependencies,(output.id,))
                        source=next(o for o in joins if result.audit[o.id]['semantic']['kind']=='kv_source')
                        write=next(o for o in result.memory_operations if o.role=='attention/kv_write' and result.audit[o.id]['layer']==layer)
                        self.assertIn(source.id,ancestors(result,write.id))
                        for read in result.memory_operations:
                            if result.audit[read.id].get('layer')==layer and read.role=='attention/kv_read_current':
                                self.assertIn(write.id,ancestors(result,read.id))

    def test_moe_shared_and_routed_weights_have_distinct_release_dependencies(self):
        _, model, kw, batches=setup(ROOT/'models/deepseek-v3-fp8-kv-bf16.json')
        result=CatalogCoverageCompiler(**kw,schedule_policy='consumer-order').compile(batches[0])
        layer=3
        route=next(o for o in result.operations if o.role==f'layer/{layer}/routing_ready')
        post=next(o for o in result.operations if o.role=='consumer/complete' and result.audit[o.id]['layer']==layer
                  and result.audit[o.id]['semantic']['kind']=='post_norm')
        for o in result.memory_operations:
            if result.audit[o.id].get('layer')!=layer:continue
            if o.role=='moe/routed_expert_weights':self.assertIn(route.id,ancestors(result,o.id))
            if o.role=='moe/shared_expert_weights':
                self.assertEqual(o.dependencies,(post.id,))
                self.assertNotIn(route.id,ancestors(result,o.id))

    def test_options_are_resolved_without_changing_base_or_snapshot_defaults(self):
        config=default_run_config(timing='memory_only',peak_tflops=100,efficiency=.5,prefetch_depth=1)
        self.assertEqual(_resolve_coarse_run_config(config,SimpleNamespace()),config)
        for coverage in ['dense-16bit-swiglu','dense-bf16-swiglu','model-derived']:
            resolved=_resolve_coarse_run_config(config,SimpleNamespace(coarse_coverage_policy=coverage))
            self.assertEqual(resolved['coarse_schedule_policy'],'legacy-prefetch')
            run_config_from_dict(resolved)
        with self.assertRaisesRegex(ServeLoopError,'requires generated'):
            _resolve_coarse_run_config(dict(config,coarse_schedule_policy='consumer-order'),SimpleNamespace())
        with self.assertRaisesRegex(ServeLoopError,'not both'):
            _resolve_coarse_run_config(dict(config,coarse_schedule_policy='legacy-prefetch'),
                SimpleNamespace(coarse_coverage_policy='model-derived',coarse_schedule_policy='consumer-order'))
        with self.assertRaisesRegex(ServeLoopError,'unsupported'):
            run_config_from_dict(dict(config,coarse_schedule_policy='unknown'))
        parser=build_parser()
        args=parser.parse_args(['--requests','x','--out','out','--coarse-coverage-policy','model-derived',
                               '--coarse-schedule-policy','consumer-order'])
        self.assertEqual(args.coarse_schedule_policy,'consumer-order')

    def test_captured_profiles_reject_explicit_schedule(self):
        _, model, kw, batches=setup(ROOT/'models/qwen3-8b-bf16-kv-bf16.json')
        generated=CatalogCoverageCompiler(**kw)
        with self.assertRaisesRegex(ServeLoopError,'captured profiles'):
            CoarseCoverageCompiler(**kw,coverage_profile=generated.coverage_for_batch(batches[0]),schedule_policy='consumer-order')

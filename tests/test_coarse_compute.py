from pathlib import Path
import unittest
from serveloop.compiler import ServeLoopCompiler
from serveloop.coarse_catalog import CatalogCoverageCompiler
from serveloop.coarse_analytic import AnalyticCoverageCompiler
from serveloop.coarse_l2 import CoarseL2Compiler
from serveloop.contracts import RooflineTimingProvider, ServeLoopError
from serveloop.catalog import load_model_any
from serveloop.io import load_request_trace
from test_coarse_catalog import setup
ROOT=Path(__file__).resolve().parents[1]


def compute_nodes(batch):
    return {o.id:(o.role,o.duration_ns) for o in batch.operations if o.duration_ns>0}

def accesses(batch):
    return [(o.op,o.object_id,o.offset,o.bytes) for o in batch.memory_operations]


class CoarseComputeTests(unittest.TestCase):
    def test_five_models_compute_once_traffic_and_cache_unchanged(self):
        provider=RooflineTimingProvider(200,.5)
        for path in sorted((ROOT/'models').glob('*.json')):
            with self.subTest(model=path.name):
                old,m,kw,batches=setup(path)
                original=ServeLoopCompiler(**kw,timing=provider)
                enhanced=CatalogCoverageCompiler(**kw,timing=provider)
                memory=CatalogCoverageCompiler(**kw)
                cached=CoarseL2Compiler(CatalogCoverageCompiler(**kw,timing=provider),41943040)
                memory_cached=CoarseL2Compiler(CatalogCoverageCompiler(**kw),41943040)
                for b in batches:
                    a=original.compile(b);e=enhanced.compile(b);zero=memory.compile(b)
                    expected=provider.timing_for(model=m,batch=b)
                    budget=sum(expected.layer_ns)+expected.tail_ns
                    self.assertEqual(compute_nodes(a),compute_nodes(e))
                    self.assertAlmostEqual(sum(v[1] for v in compute_nodes(e).values()),budget,places=6)
                    self.assertEqual(accesses(e),accesses(zero))
                    self.assertEqual(len(e.operations),len(zero.operations))
                    self.assertEqual(e.timing_evidence_state,'modeled_roofline')
                    self.assertIn('coarse_compute',e.audit[e.operations[-1].id])
                    ec=cached.compile(b);mc=memory_cached.compile(b)
                    self.assertEqual(compute_nodes(e),compute_nodes(ec))
                    self.assertEqual(accesses(ec),accesses(mc))
                self.assertEqual(cached.cache.summary()['dirty_sectors'],0)

    def test_moe_routing_split_and_prefetch_dependency_retained(self):
        old,m,kw,bs=setup(ROOT/'models/deepseek-v3-fp8-kv-bf16.json')
        rate=RooflineTimingProvider(200,.5)
        e=CatalogCoverageCompiler(**kw,timing=rate,prefetch_depth=1).compile(bs[0])
        timing=rate.timing_for(model=m,batch=bs[0]);byrole={o.role:o for o in e.operations}
        for i,layer in enumerate(m.layers):
            total=byrole[f'layer/{i}/compute'].duration_ns
            if layer.is_moe:
                total+=byrole[f'layer/{i}/routing_ready'].duration_ns
                self.assertGreater(byrole[f'layer/{i}/routing_ready'].duration_ns,0)
            self.assertAlmostEqual(total,timing.layer_ns[i],places=6)
        weights=next(o for o in e.memory_operations if o.role=='attention/weights' and e.audit[o.id]['layer']==2)
        self.assertEqual(weights.dependencies,(byrole['layer/0/compute'].id,))

    def test_dense16_alias_and_compute_rate_scaling(self):
        model=load_model_any(ROOT/'examples/coarse-coverage/model.json');trace=load_request_trace(ROOT/'examples/coarse-coverage/requests.json')
        kw=dict(models={model.model_id:model},request_trace=trace)
        from serveloop.contracts import ScheduledBatch,BatchSlice
        batch=ScheduledBatch(0,model.model_id,(BatchSlice('observed-request',0,16,0,True,'prefill'),),0.)
        a=AnalyticCoverageCompiler(**kw,timing=RooflineTimingProvider(200,.5)).compile(batch)
        b=AnalyticCoverageCompiler(**kw,timing=RooflineTimingProvider(400,.5)).compile(batch)
        self.assertEqual(accesses(a),accesses(b))
        self.assertAlmostEqual(sum(o.duration_ns for o in a.operations),2*sum(o.duration_ns for o in b.operations),places=6)

    def test_other_timing_modes_stay_rejected(self):
        from types import SimpleNamespace
        old,m,kw,bs=setup(ROOT/'models/llama31-8b-w8-kv-bf16.json')
        with self.assertRaisesRegex(ServeLoopError,'memory_only or roofline'):
            CatalogCoverageCompiler(**kw,timing=SimpleNamespace(timing_model='gpu_calibrated'))

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest
from serveloop.catalog import load_model_any
from serveloop.coarse_catalog import CatalogCoverageCompiler, build_profile
from serveloop.coarse_analytic import build_profile as dense_profile
from serveloop.coarse_l2 import CoarseL2Compiler
from serveloop.compiler import ServeLoopCompiler
from serveloop.contracts import (BatchSlice,ScheduledBatch,ModelSpec,RequestSpec,RequestTrace,TraceProvenance,ServeLoopError)
from serveloop.synthetic import HotsetZipfRouter
ROOT=Path(__file__).resolve().parents[1]
def setup(path,prompt=16):
    old=load_model_any(path);m=load_model_any(path,include_coverage_descriptor=True)
    request=RequestTrace(TraceProvenance('synthetic_sensitivity','catalog coverage software qualification'),(RequestSpec('request',0.,m.model_id,prompt,2),))
    router=HotsetZipfRouter(17,2,.9,1.2) if any(l.is_moe for l in m.layers) else None
    kw=dict(models={m.model_id:m},request_trace=request,router=router)
    bs=[ScheduledBatch(i,m.model_id,(BatchSlice('request',ctx,n,ctx,True,p),),0.) for i,(p,n,ctx) in enumerate([('prefill',prompt,0),('decode',1,prompt)])]
    return old,m,kw,bs
class CatalogCoverageTests(unittest.TestCase):
    def test_all_five_models_and_two_cache_policies(self):
        paths=sorted((ROOT/'models').glob('*.json'));self.assertEqual(len(paths),5)
        for path in paths:
            with self.subTest(model=path.name):
                old,m,kw,bs=setup(path)
                self.assertNotIn('coverage_descriptor',old.canonical())
                self.assertEqual(old.memory_objects,m.memory_objects);self.assertEqual(old.layers,m.layers)
                self.assertEqual(m.digest,ModelSpec.from_dict(m.canonical()).digest)
                plain=ServeLoopCompiler(**{**kw,"models":{old.model_id:old}});enhanced=CatalogCoverageCompiler(**kw)
                for batch in bs:
                    a=plain.compile(batch);b=enhanced.compile(batch)
                    weights=lambda c:sorted((o.op,o.object_id,o.offset,o.bytes) for o in c.memory_operations if o.object_id.startswith('model/'))
                    self.assertEqual(weights(a),weights(b))
                    self.assertEqual(a.audit_summary()['moe_expert_routes'],b.audit_summary()['moe_expert_routes'])
                    self.assertGreater(sum(o.bytes for o in b.memory_operations if o.role=='activation_coverage/W'),0)
                for policy in ['write-allocate-fetch','full-store-no-fetch']:
                    c=CoarseL2Compiler(CatalogCoverageCompiler(**kw),40*1024**2,write_policy=policy)
                    for batch in bs:self.assertTrue(c.compile(batch).memory_operations)
                    self.assertEqual(c.cache.summary()['dirty_sectors'],0)
    def test_quantized_weights_not_expanded_to_bf16(self):
        old,m,kw,bs=setup(ROOT/'models/llama31-8b-w8-kv-bf16.json')
        self.assertEqual(m.weight_footprint_bytes,old.weight_footprint_bytes)
        p=build_profile(m,kw['request_trace'].digest,'prefill',16,0)
        call=next(c for c in p.phases['prefill']['layers'][0] if c['id']=='qkv_projection')
        self.assertEqual(call['stages'][0]['ranges'][1][3],2*16*(32+2*8)*128)
        self.assertEqual(p.document['evidence']['external_activation_bytes'],2)
    def test_moe_dispatch_shared_experts_and_routing_dependency(self):
        old,m,kw,bs=setup(ROOT/'models/deepseek-v3-fp8-kv-bf16.json')
        p=build_profile(m,kw['request_trace'].digest,'prefill',16,0).phases['prefill']
        self.assertFalse(p['routing_layers'][0]);self.assertTrue(p['routing_layers'][3])
        call=next(c for c in p['layers'][3] if c['id']=='moe_dispatch')
        self.assertEqual(call['stages'][0]['ranges'][-1][3],16*8*7168*2)
        self.assertTrue(any(c['id']=='shared_gate_up' for c in p['layers'][3]))
        b=CatalogCoverageCompiler(**kw).compile(bs[0]);ops={o.id:o for o in b.operations}
        routed=next(o for o in b.memory_operations if o.role=='moe/routed_expert_weights');ready=ops[routed.dependencies[0]]
        self.assertTrue(ready.role.endswith('/routing_ready'))
        self.assertTrue(any(b.audit[d].get('operator')=='router_topk' for d in ready.dependencies))
    def test_mla_compressed_kv_and_query_rank(self):
        old,m,kw,bs=setup(ROOT/'models/deepseek-v3-fp8-kv-bf16.json')
        profile=build_profile(m,kw['request_trace'].digest,'prefill',16,0);calls=profile.phases['prefill']['layers'][0]
        self.assertEqual(next(c for c in calls if c['id']=='mla_query_down')['stages'][0]['ranges'][1][3],16*1536*2)
        self.assertEqual(next(c for c in calls if c['id']=='mla_kv_down')['stages'][0]['ranges'][1][3],16*(512+64)*2)
        self.assertNotIn('dense_kv',profile.objects);self.assertEqual(m.layers[0].kv_bytes_per_token,1152)
    def test_dense_preserves_established_coverage(self):
        old,m,kw,bs=setup(ROOT/'models/qwen3-8b-bf16-kv-bf16.json')
        for phase,n,ctx in [('prefill',16,0),('decode',1,16)]:
            self.assertEqual(build_profile(m,kw['request_trace'].digest,phase,n,ctx).phases,dense_profile(old,kw['request_trace'].digest,phase,n,ctx).phases)
    def test_mismatch_missing_geometry_and_shape_rejected(self):
        old,m,kw,bs=setup(ROOT/'models/llama31-8b-w8-kv-bf16.json')
        corrupt=deepcopy(m.coverage_descriptor);corrupt['architecture']['hidden_size']+=128
        with self.assertRaises(ServeLoopError):replace(m,coverage_descriptor=corrupt)
        with self.assertRaisesRegex(ServeLoopError,'public descriptor'):build_profile(old,kw['request_trace'].digest,'prefill',16,0)
        with self.assertRaises(ServeLoopError):build_profile(m,kw['request_trace'].digest,'decode',2,16)

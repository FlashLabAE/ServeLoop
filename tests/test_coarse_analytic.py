from dataclasses import replace
from pathlib import Path
import unittest

from serveloop.catalog import load_model_any
from serveloop.coarse_analytic import AnalyticCoverageCompiler, build_profile
from serveloop.coarse_coverage import CoarseCoverageCompiler, CoarseCoverageProfile
from serveloop.contracts import BatchSlice, ScheduledBatch, ServeLoopError
from serveloop.io import load_request_trace

DATA=Path(__file__).resolve().parents[1]/'examples/coarse-coverage'


class AnalyticTests(unittest.TestCase):
    def setUp(self):
        self.model=load_model_any(DATA/'model.json')
        self.trace=load_request_trace(DATA/'requests.json')

    def compiler(self, **kwargs):
        return AnalyticCoverageCompiler(models={self.model.model_id:self.model},
            request_trace=kwargs.pop('request_trace',self.trace),**kwargs)

    def batch(self, phase='prefill', tokens=16, context=0):
        return ScheduledBatch(0,self.model.model_id,
            (BatchSlice('observed-request',context,tokens,context,True,phase),),0.)

    def test_shape_formulas_and_bounded_operation_count(self):
        counts=[]
        for n in (1,16,128,512):
            p=build_profile(self.model,self.trace.digest,'prefill',n,0)
            phase=p.phases['prefill']
            self.assertEqual(sum(r[3] for r in phase['embedding'][0]['stages'][0]['ranges']),n*(8+3072))
            qkv=phase['layers'][0][1]['stages'][0]['ranges']
            self.assertEqual([r[3] for r in qkv],[n*3072,n*4096])
            gate=phase['layers'][0][-3]['stages'][0]['ranges']
            self.assertEqual([r[3] for r in gate],[n*3072,n*35840])
            trace=replace(self.trace,requests=(replace(self.trace.requests[0],prompt_tokens=n,
                token_ids=tuple(range(n+1))),))
            counts.append(sum(o.role.startswith('activation_coverage/') for o in
                self.compiler(request_trace=trace).compile(self.batch(tokens=n)).operations))
        self.assertEqual(len(set(counts)),1)

    def test_context_affects_persistent_kv_not_invented_score_matrix(self):
        totals=[]
        for ctx in (16,127,511):
            trace=replace(self.trace,requests=(replace(self.trace.requests[0],prompt_tokens=ctx,
                token_ids=tuple(range(ctx+1))),))
            result=self.compiler(request_trace=trace).compile(self.batch('decode',1,ctx))
            self.assertEqual(sum(o.bytes for o in result.memory_operations
                if o.role=='attention/kv_read'),28*(ctx+1)*1024)
            totals.append(sum(o.bytes for o in result.memory_operations
                if o.role.startswith('activation_coverage/')))
        self.assertEqual(len(set(totals)),1)

    def test_capture_binding_and_unsupported_policies_stay_explicit(self):
        profile=CoarseCoverageProfile.load(DATA/'qwen2.5-1.5b-sglang-bf16-p16-d1.json')
        captured=CoarseCoverageCompiler(models={self.model.model_id:self.model},
            request_trace=self.trace,coverage_profile=profile)
        with self.assertRaises(ServeLoopError):captured.compile(self.batch(tokens=128))
        with self.assertRaises(ServeLoopError):self.compiler().compile(self.batch('unsupported'))
        with self.assertRaises(ServeLoopError):self.compiler().compile(self.batch('decode',2,16))
        other=replace(self.model,final_norm_bytes=6144)
        with self.assertRaises(ServeLoopError):AnalyticCoverageCompiler(
            models={other.model_id:other},request_trace=self.trace)

    def test_ideal_bound_preserves_persistent_traffic(self):
        batch=self.batch('decode',1,16)
        a=self.compiler().compile(batch);b=self.compiler(cache_bound='ideal-temporaries').compile(batch)
        def persistent(r):return [(o.id,o.op,o.offset,o.bytes) for o in r.memory_operations
                                 if not o.role.startswith('activation_coverage/')]
        self.assertEqual(persistent(a),persistent(b))
        self.assertFalse(any(o.role.startswith('activation_coverage/') for o in b.operations))


if __name__=='__main__':unittest.main()

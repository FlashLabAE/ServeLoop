from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from serveloop.catalog import load_model_any
from serveloop.coarse_coverage import CoarseCoverageCompiler, CoarseCoverageProfile
from serveloop.compiler import ServeLoopCompiler
from serveloop.contracts import BatchSlice, ScheduledBatch, ServeLoopError, RooflineTimingProvider
from serveloop.io import load_request_trace

DATA = Path(__file__).resolve().parents[1] / 'examples/coarse-coverage'


class CoarseCoverageTests(unittest.TestCase):
    def setUp(self):
        self.model = load_model_any(DATA / 'model.json')
        self.trace = load_request_trace(DATA / 'requests.json')
        self.profile = CoarseCoverageProfile.load(DATA / 'qwen2.5-1.5b-sglang-bf16-p16-d1.json')

    def compiler(self, **kwargs):
        return CoarseCoverageCompiler(models={self.model.model_id:self.model},request_trace=self.trace,
            coverage_profile=self.profile,**kwargs)

    def batch(self, phase='prefill', tokens=16, context=0):
        return ScheduledBatch(0,self.model.model_id,
            (BatchSlice('observed-request',context,tokens,context,True,phase),),0.)

    def test_unsupported_shapes_and_model_are_rejected(self):
        with self.assertRaisesRegex(ServeLoopError,'outside'):
            self.compiler().compile(self.batch(tokens=15))
        with self.assertRaisesRegex(ServeLoopError,'outside'):
            self.compiler().compile(self.batch('decode',1,17))
        other=replace(self.model,provenance={**self.model.provenance,'source':'different model binding'})
        with self.assertRaisesRegex(ServeLoopError,'model does not match'):
            CoarseCoverageCompiler(models={other.model_id:other},request_trace=self.trace,
                coverage_profile=self.profile)
        different_trace=replace(self.trace,requests=(replace(self.trace.requests[0],arrival_ns=1.),))
        with self.assertRaisesRegex(ServeLoopError,'request trace does not match'):
            CoarseCoverageCompiler(models={self.model.model_id:self.model},request_trace=different_trace,
                coverage_profile=self.profile)

    def test_compute_calibration_cannot_silently_enter_memory_profile(self):
        with self.assertRaisesRegex(ServeLoopError,'memory_only'):
            self.compiler(timing=RooflineTimingProvider(peak_tflops=100,efficiency=.5))

    def test_profiles_cannot_escape_reserved_storage(self):
        for offset,size in [(-1,32),(self.profile.workspace_bytes,32),(0,0),(0,True)]:
            doc=deepcopy(self.profile.document)
            doc['phases']['prefill']['embedding'][0]['stages'][0]['ranges'][0][2:]=[offset,size]
            with self.assertRaises(ServeLoopError):CoarseCoverageProfile(doc)

    def test_decode_current_kv_and_ideal_bound_keep_persistent_traffic(self):
        batch=self.batch('decode',1,16)
        original=ServeLoopCompiler(models={self.model.model_id:self.model},request_trace=self.trace).compile(batch)
        coverage=self.compiler().compile(batch)
        ideal=self.compiler(cache_bound='ideal-temporaries').compile(batch)
        def persistent(result):
            return [(o.id,o.op,o.object_id,o.offset,o.bytes) for o in result.memory_operations
                if not o.role.startswith('activation_coverage/')]
        self.assertEqual(persistent(coverage),persistent(ideal))
        old=sum(o.bytes for o in original.memory_operations if o.role in ('attention/kv_read','attention/kv_read_current'))
        new=sum(o.bytes for o in coverage.memory_operations if o.role in ('attention/kv_read','attention/kv_read_current'))
        self.assertEqual(new-old,28*1024)
        self.assertFalse(any(o.role.startswith('activation_coverage/') for o in ideal.operations))
        self.assertTrue(any(o.role=='activation_coverage/W' for o in coverage.operations))
        self.assertEqual(ideal.audit[ideal.operations[-1].id]['coarse_cache_bound'],'ideal-temporaries')


if __name__ == '__main__':
    unittest.main()

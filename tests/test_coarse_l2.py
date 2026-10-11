from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path
import random
import unittest

from serveloop.catalog import load_model_any
from serveloop.coarse_analytic import AnalyticCoverageCompiler
from serveloop.coarse_l2 import RangeLRU, CoarseL2Compiler
from serveloop.contracts import BatchSlice, ScheduledBatch, ServeLoopError
from serveloop.io import load_request_trace

DATA=Path(__file__).resolve().parents[1]/'examples/coarse-coverage'


class RangeCacheTests(unittest.TestCase):
    def oracle_check(self, capacity, accesses, no_fetch=False):
        ranged=RangeLRU(capacity*32,full_store_no_fetch=no_fetch)
        # Independent scalar oracle; production stores compressed intervals.
        resident=OrderedDict();stats=Counter()
        def compact(events):
            return Counter((t.op,t.object_id,s) for t in events for s in range(t.offset//32,(t.offset+t.bytes+31)//32))
        for ordinal,(obj,op,begin,end) in enumerate(accesses):
            expected=Counter()
            for sector in range(begin,end):
                key=(obj,sector)
                if key in resident:
                    stats[op+'_hit_sectors']+=1;dirty=resident.pop(key)
                else:
                    stats[op+'_miss_sectors']+=1;dirty=False
                    if op=='R' or not no_fetch:expected[('R',obj,sector)]+=1
                resident[key]=dirty or op=='W'
                if len(resident)>capacity:
                    (victim_obj,victim_sector),victim_dirty=resident.popitem(last=False)
                    if victim_dirty:expected[('W',victim_obj,victim_sector)]+=1
            actual,_=ranged.access(obj,op,begin*32,(end-begin)*32,4096,str(ordinal))
            self.assertEqual(compact(actual),expected)
            self.assertEqual([(e.object_id,s) for e in ranged.entries for s in range(e.begin,e.end)],list(resident))
            self.assertEqual({(e.object_id,s) for e in ranged.entries if e.dirty for s in range(e.begin,e.end)},{k for k,v in resident.items() if v})
            for direction in ('R','W'):
                for kind in ('hit','miss'):
                    key=direction+'_'+kind+'_sectors';self.assertEqual(ranged.stats[key],stats[key])
            self.assertLessEqual(ranged.resident,capacity)
        expected=Counter(('W',obj,sector) for (obj,sector),dirty in resident.items() if dirty)
        self.assertEqual(compact(ranged.drain()),expected)
        self.assertEqual(ranged.summary()['dirty_sectors'],0)

    def test_exact_to_scalar_oracle_under_declared_geometry(self):
        rng=random.Random(917)
        for capacity in (1,2,7,32):
            accesses=[]
            for i in range(160):
                a=rng.randrange(64);b=rng.randrange(a+1,100)
                accesses.append((rng.choice(('a','b','c')),rng.choice(('R','W')),a,b))
            self.oracle_check(capacity,accesses)
            self.oracle_check(capacity,accesses,no_fetch=True)

    def test_stream_can_evict_initial_future_hits(self):
        self.oracle_check(4,[('a','W',4,8),('a','R',0,8),('a','R',6,8),('b','W',0,7)])

    def test_gigabyte_write_is_not_sector_expansion(self):
        c=RangeLRU(1024**2)
        output,_=c.access('a','W',0,1024**3,1024**3,'write')
        self.assertEqual(c.stats['interval_steps'],1)
        self.assertEqual(len(c.entries),1)
        self.assertEqual(c.resident,1024**2//32)
        self.assertLessEqual(len(output),3)
        self.assertEqual(sum(t.bytes for t in output if t.op=='R'),1024**3)
        self.assertEqual(sum(t.bytes for t in output+c.drain() if t.op=='W'),1024**3)

    def test_full_store_no_fetch_keeps_partial_store_fills(self):
        c=RangeLRU(64,full_store_no_fetch=True)
        output,_=c.access('a','W',1,62,96,'write')
        self.assertEqual(sum(t.bytes for t in output if t.op=='R'),64)
        c=RangeLRU(64,full_store_no_fetch=True)
        output,_=c.access('a','W',0,64,96,'write')
        self.assertFalse(output)
        self.assertEqual(sum(t.bytes for t in c.drain()),64)

    def test_partial_sector_clipping_and_invalidation(self):
        c=RangeLRU(64)
        output,_=c.access('a','W',32,3,35,'write')
        self.assertEqual([(t.op,t.offset,t.bytes) for t in output],[('R',32,3)])
        self.assertEqual([(t.op,t.offset,t.bytes) for t in c.invalidate('a')],[('W',32,3)])
        self.assertEqual(c.resident,0)
        for bad in (0,-32,33,True):
            with self.assertRaises(ServeLoopError):RangeLRU(bad)


class CompilerCacheTests(unittest.TestCase):
    def compiler(self, capacity):
        model=load_model_any(DATA/'model.json');trace=load_request_trace(DATA/'requests.json')
        self.model=model;self.trace=trace
        return CoarseL2Compiler(AnalyticCoverageCompiler(models={model.model_id:model},request_trace=trace),capacity)

    def test_unsupported_bound_and_multi_request_are_rejected(self):
        self.compiler(64)
        options=dict(models={self.model.model_id:self.model},request_trace=self.trace)
        with self.assertRaises(ServeLoopError):
            CoarseL2Compiler(AnalyticCoverageCompiler(cache_bound='ideal-temporaries',**options),64)
        multiple=replace(self.trace,requests=(self.trace.requests[0],
            replace(self.trace.requests[0],request_id='other-request')))
        with self.assertRaises(ServeLoopError):
            CoarseL2Compiler(AnalyticCoverageCompiler(models=options['models'],request_trace=multiple),64)

    def test_continuous_states_causal_graph_and_final_dirty_drain(self):
        c=self.compiler(1024**2)
        for i,(phase,tokens,context) in enumerate([('prefill',16,0),('decode',1,16)]):
            b=ScheduledBatch(i,self.model.model_id,(BatchSlice('observed-request',context,tokens,context,True,phase),),0.)
            result=c.compile(b)  # contract verifies backward-only dependencies/roles
            self.assertTrue(result.memory_operations)
            self.assertTrue(any(o.role=='coarse_l2/read_miss' for o in result.memory_operations))
        self.assertEqual(c.cache.summary()['dirty_sectors'],0)
        self.assertGreater(c.cache.stats['W_hit_sectors'],0)
        self.assertGreater(c.cache.stats['R_hit_sectors'],0)
        self.assertGreater(c.cache.stats['final_drain_sectors'],0)
        with self.assertRaises(ServeLoopError):c.compile(b)


if __name__=='__main__':unittest.main()

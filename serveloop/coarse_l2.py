"""Range-compressed, fully associative sector LRU approximation.

Exact byte/sector accounting for the declared ascending coarse range sequence,
not NVIDIA set indexing, warp issue order, L1 or GPU timing. Intervals in LRU
order replace per-sector entries; long streaming accesses remain bulk ranges.
"""
from collections import Counter
from dataclasses import dataclass, replace

from serveloop.contracts import ServeLoopError, SemanticOperation

SECTOR_BYTES = 32


@dataclass(frozen=True)
class Segment:
    object_id: str
    begin: int
    end: int
    dirty: bool
    owner: str
    extent: int


@dataclass(frozen=True)
class Transfer:
    object_id: str
    op: str
    offset: int
    bytes: int
    reason: str
    owner: str


class RangeLRU:
    def __init__(self, capacity_bytes, *, full_store_no_fetch=False):
        if type(capacity_bytes) is not int or capacity_bytes < SECTOR_BYTES or capacity_bytes % SECTOR_BYTES:
            raise ServeLoopError('coarse L2 capacity must be a positive multiple of32 bytes')
        self.capacity=capacity_bytes//SECTOR_BYTES
        if type(full_store_no_fetch) is not bool:
            raise ServeLoopError('invalid coarse L2 write policy')
        self.full_store_no_fetch=full_store_no_fetch
        self.entries=[]  # oldest first; each interval is internally ascending
        self.resident=0
        self.stats=Counter()

    def _writeback(self, segment, reason):
        offset=segment.begin*SECTOR_BYTES
        size=min(segment.end*SECTOR_BYTES,segment.extent)-offset
        self.stats['writeback_bytes']+=size
        self.stats[reason+'_sectors']+=segment.end-segment.begin
        return Transfer(segment.object_id,'W',offset,size,reason,segment.owner)

    def _append(self, segment):
        if self.entries and self.entries[-1].end==segment.begin and all(
                getattr(self.entries[-1],k)==getattr(segment,k)
                for k in ('object_id','dirty','owner','extent')):
            self.entries[-1]=replace(self.entries[-1],end=segment.end)
        else:
            self.entries.append(segment)
        self.resident+=segment.end-segment.begin
        evicted=[]
        while self.resident>self.capacity:
            first=self.entries[0];n=min(first.end-first.begin,self.resident-self.capacity)
            victim=replace(first,end=first.begin+n)
            if n==first.end-first.begin:self.entries.pop(0)
            else:self.entries[0]=replace(first,begin=first.begin+n)
            self.resident-=n;self.stats['evicted_sectors']+=n
            if victim.dirty:evicted.append(self._writeback(victim,'dirty_eviction'))
        self.stats['peak_intervals']=max(self.stats['peak_intervals'],len(self.entries))
        self.stats['peak_resident_sectors']=max(self.stats['peak_resident_sectors'],self.resident)
        return evicted

    def access(self, object_id, op, offset, size, extent, owner):
        if (op not in ('R','W') or any(type(v) is not int for v in (offset,size,extent))
                or offset<0 or size<=0 or offset+size>extent):
            raise ServeLoopError('invalid bounded coarse L2 access')
        self.stats['input_'+op+'_bytes']+=size
        begin=offset//SECTOR_BYTES;end=(offset+size+SECTOR_BYTES-1)//SECTOR_BYTES
        transfers=[];dependencies=set();pos=begin
        while pos<end:
            self.stats['interval_steps']+=1
            found=None;next_start=end
            for i,e in enumerate(self.entries):
                if e.object_id!=object_id:continue
                if e.begin<=pos<e.end:found=(i,e);break
                if e.begin>pos:next_start=min(next_start,e.begin)
            if found is not None:
                i,e=found;stop=min(end,e.end)
                self.stats[op+'_hit_sectors']+=stop-pos;dependencies.add(e.owner)
                fragments=[]
                if e.begin<pos:fragments.append(replace(e,end=pos))
                if stop<e.end:fragments.append(replace(e,begin=stop))
                self.entries[i:i+1]=fragments;self.resident-=stop-pos
                transfers.extend(self._append(Segment(object_id,pos,stop,e.dirty or op=='W',owner,extent)))
            else:
                stop=next_start
                # At most two partial boundary sectors need fetch when full
                # stores allocate without reading old values.
                fetch=(op=='R' or not self.full_store_no_fetch)
                if op=='W' and self.full_store_no_fetch:
                    if pos==begin and offset>pos*SECTOR_BYTES:
                        stop=min(stop,pos+1);fetch=True
                    elif stop==end and offset+size<min(end*SECTOR_BYTES,extent):
                        if pos==end-1:fetch=True
                        else:stop=end-1
                self.stats[op+'_miss_sectors']+=stop-pos
                # All misses fetch initialized sectors, including write-allocate.
                start_byte=pos*SECTOR_BYTES;byte_count=min(stop*SECTOR_BYTES,extent)-start_byte
                if fetch:self.stats['fill_read_bytes']+=byte_count
                else:self.stats['write_allocations_without_fetch']+=stop-pos
                evicted=self._append(Segment(object_id,pos,stop,op=='W',owner,extent))
                # Bulk representation changes within-range service ordering,
                # but not sectors/bytes/state. Own streaming dirty evictions
                # must follow this access's fill; earlier owners precede it.
                transfers.extend(t for t in evicted if t.owner!=owner)
                if fetch:
                    transfers.append(Transfer(object_id,'R',start_byte,byte_count,
                                              'read_miss' if op=='R' else 'write_allocate_fill',owner))
                transfers.extend(t for t in evicted if t.owner==owner)
            pos=stop
        return transfers,dependencies

    def invalidate(self, object_id):
        output=[];kept=[]
        for e in self.entries:
            if e.object_id==object_id:
                self.resident-=e.end-e.begin
                if e.dirty:output.append(self._writeback(e,'workspace_rebind'))
            else:kept.append(e)
        self.entries=kept
        return output

    def drain(self):
        output=[]
        for e in self.entries:
            if e.dirty:output.append(self._writeback(e,'final_drain'))
        self.entries=[replace(e,dirty=False) for e in self.entries]
        return output

    def summary(self):
        return dict(capacity_bytes=self.capacity*SECTOR_BYTES,sector_bytes=SECTOR_BYTES,
                    full_store_no_fetch=self.full_store_no_fetch,
                    resident_sectors=self.resident,intervals=len(self.entries),
                    dirty_sectors=sum(e.end-e.begin for e in self.entries if e.dirty),
                    stats=dict(self.stats))


class CoarseL2Compiler:
    """Apply the range cache before placement; retain original IDs as joins."""
    def __init__(self, compiler, capacity_bytes, *, write_policy='write-allocate-fetch'):
        if len(compiler.request_trace.requests)!=1:
            raise ServeLoopError('coarse L2 currently supports one complete request per run')
        if compiler.cache_bound!='off':
            raise ServeLoopError('coarse L2 cannot combine with ideal-temporaries')
        if write_policy not in ('write-allocate-fetch','full-store-no-fetch'):
            raise ServeLoopError('unsupported coarse L2 write policy')
        self.compiler=compiler;self.cache=RangeLRU(capacity_bytes,
            full_store_no_fetch=write_policy=='full-store-no-fetch');self.last_batch=-1
        self.records=[]

    def __getattr__(self,name):
        return getattr(self.compiler,name)

    def compile(self,batch):
        if batch.batch_id!=self.last_batch+1:
            raise ServeLoopError('stateful coarse L2 requires consecutive batches from0')
        coverage=self.compiler.coverage_for_batch(batch)
        source=self.compiler.compile(batch)
        if any(o.walk is not None for o in source.memory_operations):
            raise ServeLoopError('coarse L2 requires contiguous range operations')
        model=self.models[batch.model_id];s=batch.slices[0]
        request=self.request_trace.requests[0];output=[];audit=dict(source.audit)
        before=dict(self.cache.stats);index=0;known=set();boundary=[]

        def emit(transfers, dependencies, current=None):
            nonlocal index
            ids=[];previous=()
            for t in transfers:
                owner=(t.owner,) if t.owner in known and t.owner!=current else ()
                deps=tuple(dict.fromkeys((*dependencies,*owner,*previous)))
                identifier=f'serve/b{batch.batch_id}/l2io{index}';index+=1
                role='coarse_l2/'+t.reason
                output.append(SemanticOperation(identifier,t.op,t.object_id,t.offset,t.bytes,0.,deps,role))
                audit[identifier]=dict(role=role,source_operation=current,traffic_semantics='modeled post-L2; range LRU approximation')
                known.add(identifier);ids.append(identifier);previous=(identifier,)
            return ids

        # Packed temporary allocation is rebound each forward: do not fabricate
        # hits when shape/position data change at reused workspace offsets.
        boundary=emit(self.cache.invalidate(f'workspace/{model.model_id}'),())
        for operation in source.operations:
            if operation.is_barrier:
                deps=operation.dependencies
                if operation.role=='batch/complete' and s.token_end==request.processed_input_tokens:
                    flushed=emit(self.cache.drain(),deps)
                    deps=tuple(dict.fromkeys((*deps,*flushed)))
                output.append(replace(operation,dependencies=deps));known.add(operation.id)
                continue
            if operation.object_id.startswith('workspace/'):
                extent=coverage.workspace_bytes
            elif operation.object_id.startswith('request/'):
                layer=source.audit[operation.id]['layer']
                extent=request.processed_input_tokens*model.layers[layer].kv_bytes_per_token
            else:
                extent=model.object_by_id[operation.object_id].bytes
            transfers,hits=self.cache.access(operation.object_id,operation.op,operation.offset,
                                             operation.bytes,extent,operation.id)
            deps=tuple(dict.fromkeys((*operation.dependencies,*boundary,
                                      *(h for h in sorted(hits) if h in known))))
            generated=emit(transfers,deps,operation.id)
            output.append(replace(operation,op=None,object_id=None,offset=0,bytes=0,
                                  dependencies=tuple(dict.fromkeys((*deps,*generated)))))
            audit[operation.id]=dict(audit[operation.id],l2_input_bytes=operation.bytes,
                                    l2_input_op=operation.op,traffic_semantics='cache access completion, zero L2 service latency')
            known.add(operation.id)
        self.last_batch=batch.batch_id
        summary=self.cache.summary()
        summary['phase_delta']={k:v-before.get(k,0) for k,v in self.cache.stats.items()
                                if not k.startswith('peak_')}
        summary.update(batch_id=batch.batch_id,phase=batch.kind)
        self.records.append(summary)
        audit[output[-1].id]=dict(audit[output[-1].id],coarse_l2=summary)
        return replace(source,operations=tuple(output),audit=audit)

    def cache_receipt(self):
        return dict(model='fully_associative_sector_lru_range',initial='cold',
                    write_policy='write-back/write-allocate/'+('full-store-no-fetch' if self.cache.full_store_no_fetch else 'fetch'),final_policy='dirty drain at last forward',
                    scope='one request; canonical insertion order, ascending ranges; persistent weights/KV continuous; temporary workspace rebound each forward',
                    excludes='GPU set conflicts, warp order/repeats, L1/shared memory, L2 service latency; not hardware validated',
                    batches=self.records,final=self.cache.summary())

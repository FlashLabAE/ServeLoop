"""A100 FA2/Marlin application addresses and their explicit footprint projection.

The application iterator preserves addresses and order within each warp. It
does not prescribe native inter-warp issue timing. The execution projection
collapses repeats to the tensor-footprint basis of the existing timing fit;
it is not a cache simulator or a measured DRAM-miss stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

from serveloop.contracts import ServeLoopError


ADDRESS_MODEL = 'a100_fa2_marlin_application_v1'
MEMORY_PROJECTION = 'kernel_ordered_tensor_coverage_v1'
TENSOR_ADDRESS_MODEL = 'tensor_footprint_unvalidated'
TENSOR_MEMORY_PROJECTION = 'tensor_footprint_v1'


def projection_identity(hardware):
    """Only the archived A100 backend has measured application-address walks.

    Other GPUs still use the same tensor ledger and page-local KV layout;
    their traversal is a footprint assumption, not an inherited A100 trace.
    """
    address_model = hardware.get('address_model', TENSOR_ADDRESS_MODEL)
    if address_model == ADDRESS_MODEL:
        return address_model, MEMORY_PROJECTION
    if address_model == TENSOR_ADDRESS_MODEL:
        return address_model, TENSOR_MEMORY_PROJECTION
    raise ServeLoopError(f'unsupported GPU address model {address_model}')
FA2_SOURCE = 'vllm-project/flash-attention@caaa4eb59845388a20b1f435ecaafb4bd9517ad8'
MARLIN_SOURCE = 'vllm-project/vllm@v0.26.0/csrc/libtorch_stable/quantization/marlin'


def ceildiv(a, b):
    return (a+b-1)//b


def decode_splits(batch, kv_heads, sequence, sm_count=108):
    """FA2 occupancy heuristic; checked against measured launch grids."""
    slots = 2*sm_count
    parallel = batch*kv_heads
    if parallel >= .8*slots:
        return 1
    blocks = ceildiv(sequence, 128)
    scores = []
    for n in range(1, min(128, slots, blocks)+1):
        if n > 1 and ceildiv(blocks,n) == ceildiv(blocks,n-1):
            continue
        waves = parallel*n/slots
        scores.append((n, waves/math.ceil(waves)))
    best = max(s for _,s in scores)
    return next(n for n,s in scores if s >= .85*best)


def attention_plan(g, queries, contexts):
    if g['head_dim'] != 128 or g['heads'] % g['kv_heads']:
        raise ServeLoopError('A100 address model requires measured head-128 GQA geometry')
    q,c = list(queries),list(contexts)
    decode = max(q) == 1
    splits = decode_splits(len(q),g['kv_heads'],max(x+y for x,y in zip(q,c))) if decode else 1
    return dict(model=ADDRESS_MODEL, kernel='fa2_paged_head128', source=FA2_SOURCE,
        queries=q, contexts=c, heads=g['heads'], kv_heads=g['kv_heads'],
        head_bytes=256, block_tokens=256, query_tile=64, key_tile=128,
        decode=decode, splits=splits,
        grid=[1 if decode else ceildiv(max(q),64),
              splits if splits>1 else len(q),
              len(q)*g['kv_heads'] if splits>1 else (g['kv_heads'] if decode else g['heads'])],
        warps=4, native_interwarp_order_measured=False,
        covered_tensors=['key_cache','value_cache'])


def _attention_cta(plan, cta):
    m,y,z = cta
    split = y if plan['splits']>1 else 0
    if plan['splits']>1:
        request,head = divmod(z,plan['kv_heads'])
    else:
        request,head = y,z
    q,c = plan['queries'][request],plan['contexts'][request]
    if m*64 >= (plan['heads']//plan['kv_heads'] if plan['decode'] else q):
        return request,head,0,0
    blocks = ceildiv(q+c,128)
    per_split = ceildiv(blocks,plan['splits'])
    lower,upper = split*per_split,min(blocks,(split+1)*per_split)
    if not plan['decode']:
        upper = min(upper,ceildiv(c+(m+1)*64,128))
        head //= plan['heads']//plan['kv_heads']
    return request,head,lower,upper


@dataclass(frozen=True)
class WarpAccess:
    cta: tuple[int,int,int]
    warp: int
    tensor: str
    offsets: tuple[int | None, ...]
    width: int = 16
    request: int | None = None

    @property
    def mask(self):
        return sum(1<<i for i,a in enumerate(self.offsets) if a is not None)

    @property
    def requested_bytes(self):
        return sum(a is not None for a in self.offsets)*self.width


def attention_warp(plan, cta, warp, block_tables=None):
    """K/V memory instructions, including predicated partial 128-token tiles.

    With block_tables, offsets address the GPU's separate pooled K/V tensors.
    Without them, each request has its own logical K/V tensor address space.
    """
    request,head,lo,hi = _attention_cta(plan,cta)
    length = plan['queries'][request]+plan['contexts'][request]
    row_bytes = plan['kv_heads']*256
    for tile in range(hi-1,lo-1,-1):
        for tensor in ('key_cache','value_cache'):
            for row,half in product(range(8),(0,128)):
                offsets = []
                for lane in range(32):
                    token = tile*128+warp*32+(lane//8)*8+row
                    if token >= length:
                        offsets.append(None)
                        continue
                    if block_tables is not None:
                        page,within = divmod(token,256)
                        token = block_tables[request][page]*256+within
                    offsets.append(token*row_bytes+head*256+(lane%8)*16+half)
                if any(x is not None for x in offsets):
                    yield WarpAccess(tuple(cta),warp,tensor,tuple(offsets),request=request)


def attention_ctas(plan):
    # This deterministic enumeration is NOT an assertion about GPU scheduling.
    x,y,z = plan['grid']
    for iz,iy,ix in product(range(z),range(y),range(x)):
        yield ix,iy,iz


def attention_requested_bytes(plan):
    factor = 2*256
    if plan['decode']:
        return factor*plan['kv_heads']*sum(q+c for q,c in zip(plan['queries'],plan['contexts']))
    return factor*plan['heads']*sum(
        min(q+c,ceildiv(c+(m+1)*64,128)*128)
        for q,c in zip(plan['queries'],plan['contexts']) for m in range(ceildiv(q,64)))


def marlin_plan(g, tokens):
    n = (g['heads']+2*g['kv_heads'])*128
    # Other Marlin tile dispatches remain explicitly unmeasured. They must not
    # inherit the small-M iterator simply because their weight width is 8 bit.
    if not (1 <= tokens <= 16 and g['hidden']==4096 and n in (6144,9216)):
        return None
    return dict(model=ADDRESS_MODEL, kernel='marlin_qkv_m16_n128_k128',
        source=MARLIN_SOURCE, tokens=tokens, k=4096, n=n, grid=[108,1,1],
        warps=8, quant_group=128, weight_bits=8,
        covered_tensors=['projection_input','projection_weight','projection_scales'],
        native_interwarp_order_measured=False)


def marlin_units(plan, cta):
    ktiles,ntiles = plan['k']//128,plan['n']//128
    per_cta = ceildiv(ktiles*ntiles,plan['grid'][0])
    for unit in range(cta*per_cta,min((cta+1)*per_cta,ktiles*ntiles)):
        yield divmod(unit,ktiles)  # N tile, then ascending K tile within a slice.


def marlin_warp(plan, cta, warp):
    k,n,m = plan['k'],plan['n'],plan['tokens']
    tids = range(warp*32,(warp+1)*32)
    for nt,kt in marlin_units(plan,cta[0]):
        offsets = tuple((tid//16)*k*2+kt*256+(tid%16)*16 if tid//16<m else None for tid in tids)
        if any(x is not None for x in offsets):
            yield WarpAccess(tuple(cta),warp,'projection_input',offsets)
        for sub in range(4):
            offsets = tuple((kt*8+tid//128+2*sub)*n*16+(nt*128+tid%128)*16 for tid in tids)
            yield WarpAccess(tuple(cta),warp,'projection_weight',offsets)
        offsets = tuple(kt*n*2+nt*256+tid*16 if tid<16 else None for tid in tids)
        if any(x is not None for x in offsets):
            yield WarpAccess(tuple(cta),warp,'projection_scales',offsets)


def application_summary(plan):
    if plan['kernel']=='fa2_paged_head128':
        footprint = 512*plan['kv_heads']*sum(q+c for q,c in zip(plan['queries'],plan['contexts']))
        requested = attention_requested_bytes(plan)
        return dict(covered_read_bytes=requested, covered_tensor_bytes=footprint,
                    kv_instruction_read_bytes=requested, repeat_ratio=requested/footprint)
    m,k,n = (plan[k] for k in ('tokens','k','n'))
    weights,scales,input_bytes = k*n,k//128*n*2,m*k*2
    return dict(covered_read_bytes=weights+scales+input_bytes*(n//128),
        covered_tensor_bytes=weights+scales+input_bytes,
        input_instruction_read_bytes=input_bytes*(n//128), input_repeat_ratio=n//128,
        weight_instruction_read_bytes=weights, scale_instruction_read_bytes=scales)


def kv_object_offset(token, plane, row_bytes, block_tokens=256):
    """Page-local K then V planes; placement retains physical shared-page IDs."""
    page,within = divmod(token,block_tokens)
    return page*2*block_tokens*row_bytes+plane*block_tokens*row_bytes+within*row_bytes


def coverage_ranges(walk):
    """Compact execution projection: count each request's covered bytes once.

    Repeats remain in the application plan, NOT silently sent to DRAM. This
    projection deliberately retains the old fit's byte basis, including one
    shared-prefix read per request. Its range order is a declared coarsening
    of the kernel traversal, not measured cache-miss order.
    """
    kind = walk['kind']
    if kind=='fa2_kv':
        p,i = walk['plan'],walk['request']
        q,c = p['queries'][i],p['contexts'][i]
        row_bytes = p['kv_heads']*256
        seen = set()
        for m in range(1 if p['decode'] else ceildiv(q,64)):
            for split in range(p['splits']):
                blocks = ceildiv(q+c,128)
                per_split = ceildiv(blocks,p['splits'])
                lo,hi = split*per_split,min(blocks,(split+1)*per_split)
                if not p['decode']:
                    hi = min(hi,ceildiv(c+(m+1)*64,128))
                for tile in range(hi-1,lo-1,-1):
                    if tile in seen:
                        continue
                    seen.add(tile)
                    begin = tile*128
                    count = min(128,q+c-begin)
                    for plane in (0,1):
                        yield kv_object_offset(begin,plane,row_bytes),count*row_bytes
    elif kind in ('kv_append','kv_tensor'):
        cursor,end = walk['begin'],walk['end']
        row_bytes = walk['row_bytes']
        while cursor<end:
            count = min(end-cursor,256-cursor%256)
            for plane in (0,1):
                yield kv_object_offset(cursor,plane,row_bytes),count*row_bytes
            cursor += count
    elif kind=='marlin_qkv':
        p = walk['plan']
        n,k = p['n'],p['k']
        for cta in range(p['grid'][0]):
            for nt,kt in marlin_units(p,cta):
                # Merge lanes/warps within a packed-weight row. Cross-warp
                # order is intentionally unspecified by the measured model.
                for sub in range(8):
                    yield (kt*8+sub)*n*16+nt*2048,2048
                yield n*k+kt*n*2+nt*256,256
    else:
        raise ServeLoopError(f'unknown GPU coverage walk {kind}')


def iter_operation_ranges(operation):
    if operation.walk is None:
        yield operation.offset,operation.bytes
    else:
        for offset,size in coverage_ranges(operation.walk):
            yield operation.offset+offset,size


def iter_batch_application(batch, block_tables=None):
    """Stream measured-family application addresses without materializing them.

    Returns (operator ID, WarpAccess). block_tables maps a layer to request
    block tables when GPU-pool-relative K/V offsets are desired.
    """
    for operation in batch.operations:
        labels = batch.audit[operation.id]
        plan = labels.get('application_access_plan')
        if plan is None:
            continue
        if plan['kernel']=='fa2_paged_head128':
            tables = None if block_tables is None else block_tables[labels['layer']]
            for cta in attention_ctas(plan):
                for warp in range(plan['warps']):
                    for access in attention_warp(plan,cta,warp,tables):
                        yield operation.id,access
        else:
            for cta in range(plan['grid'][0]):
                for warp in range(plan['warps']):
                    for access in marlin_warp(plan,(cta,0,0),warp):
                        yield operation.id,access

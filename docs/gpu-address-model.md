# A100 application addresses in ServeLoop

The `gpu_calibrated` compiler now emits shape-dependent address plans and
kernel-ordered coverage walks. No additional option is needed. It replaces
the old single linear KV-history read and, for supported QKV shapes, the
single linear packed-weight read. Other timing providers remain object-level
models with their own declared scope.

## What is generated

`serveloop/gpu_addresses.py` describes two measured A100 kernel families:

- Paged FA2, BF16, head dimension 128, 256-token cache blocks. Pure decode
  uses GQA folding and the source-derived split-KV dispatch; mixed/prefill
  uses query-head CTAs with 64-query and 128-key tiles. Within each warp,
  the generator reproduces descending key tiles, K/V instruction order,
  16-byte lane vectors and partial-tile masks. It accepts each request's
  own query/context length and optional physical block table.
- Small-M Marlin QKV with `1 <= M <= 16`, `K=4096`, and `N=6144` or `9216`.
  The template describes activation loads, packed-weight slices and scales.
  The original measurements cover M=4 for 8B and 235B geometry; the other M
  values follow that kernel's source and require held-out measurement.
  70B QKV and larger-M dispatches retain labeled footprint estimates.

This is based on vLLM 0.26.0 and its pinned FA2 dependency, not a generic
FlashAttention model. Source references:
[FA2 dispatch](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/csrc/flash_attn/flash_api.cpp),
[FA2 kernel](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/csrc/flash_attn/src/flash_fwd_kernel.h),
[Marlin kernel](https://github.com/vllm-project/vllm/blob/v0.26.0/csrc/libtorch_stable/quantization/marlin/marlin_template.h).

To inspect application requests without materializing an entire workload:

```python
from serveloop.gpu_addresses import iter_batch_application

for operator_id, access in iter_batch_application(canonical_batch):
    # access: CTA, logical warp, tensor, 32 lane offsets, width and mask.
    # Inactive lanes have offset None. Offsets are allocation-relative.
    consume(operator_id, access)
```

The iterator includes the covered tensors only. For logical KV offsets,
`access.request` indexes the batch's request slices; equal offsets in two
requests do not imply shared physical storage. Optional `block_tables` maps
layer IDs to request block tables and produces pooled K/V-relative offsets.

## How the memory simulator consumes it

Canonical operations retain compact `walk` descriptors (batch schema v3).
Placement expands these into mapped contiguous ranges using the actual
request page table, including shared prefixes, fragmentation and HBF migration.
FA2 walks preserve the tile traversal, coalesce heads within a tile and count
each request's covered bytes once. QKV walks preserve CTA weight slices and
interleaved scale ranges. A page's first half contains K, the second V; KV
append uses the same layout and splits correctly at 256-token boundaries.
This placement packs the paired K/V planes into each simulator allocation;
the application iterator retains the GPU's separate pooled K/V tensor spaces.
The simulator placement is not a reconstruction of GPU physical addresses.

The execution projection is named `kernel_ordered_tensor_coverage_v1`.
Its byte basis remains the tensor coverage used by the existing timing fit,
including one shared-prefix read per requesting sequence. Application repeats
are separately reported in `audit_summary()['application_addresses']`.
They are **not** multiplied into HBF/DRAM traffic or the memory-service
coefficient. The projection is an explicit modeling assumption: it neither
simulates GPU cache filtering nor reproduces cache-miss order.

Native placement receipts carry this distinction and set
`cache_misses_measured=False`. Kernel issue times across warps/SMs, other
operators' addresses, expert memory streams, workspace polling and actual
L2/DRAM traffic remain outside the address evidence. Aggregate native latency
may change when ranges are reordered; unchanged reference-bandwidth timing
does not establish native controller accuracy.

## Evidence and checks

The original ten A100 acquisition cases match **5,760/5,760** active per-warp
streams exactly: tensor identity, lane addresses, masks, access widths and
covered-tensor order. These were used to develop the templates, so this is
a development result. Eight additional shapes were fixed and submitted as
Slurm job 263472; the implementation was frozen before inspecting their
results. At this update those results are not yet retrieved because SSH times out. No held-out success is
claimed.

The HBFSim companion repository contains the comparator and frozen hashes in
`evidence/hardware/gpu_operators/`, plus integration tests in
`tests/python/test_gpu_address_generator.py`. These test partial tiles,
K/V append, real shared/fragmented mappings, native HBM execution, migration
and direct HBF execution. The paper adapter now tracks disjoint initialized
K/V ranges, so cache fills and writeback exclude the gap between partial
planes. All 46 archived serving batches retain their prior
reference-bandwidth timing predictions. The timing coefficients are unchanged.

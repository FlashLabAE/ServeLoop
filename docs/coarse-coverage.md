# Operator coverage for the fast coarse path

`serveloop run --coarse-coverage-profile FILE` adds activation/control/scratch
coverage to the original weight/KV compiler. The original path remains the
default when the option is absent. GPU compute calibration is not required;
the current coverage interface requires `--timing memory_only`.

The packaged example supports exactly the included Qwen2.5-1.5B BF16/SGLang
model and request digests: B1,P16 plus one actual decode at context17. It uses
previously qualified tensor/kernel coverage, not full observation files or
individual GPU requests. Different models, input traces, batch shapes and
contexts are rejected instead of silently extrapolated. This is a bounded
reusable profile interface, not a general model/shape generator.

From the ServeLoop checkout, after `python -m pip install -e .`, use a fresh output directory:

```sh
python -m serveloop run \
  --model examples/coarse-coverage/model.json \
  --requests examples/coarse-coverage/requests.json \
  --system /path/to/system.cfg --simulator /path/to/hbfsim \
  --placement weights-hbf-kv-hbm --timing memory_only --prefetch-depth 1 \
  --coarse-coverage-profile examples/coarse-coverage/qwen2.5-1.5b-sglang-bf16-p16-d1.json \
  --out /path/to/fresh-output
```

Use `--placement all-hbm` for the other qualified placement. No experimental
environment overrides are required. A standalone installation uses the same
arguments with `python -m serveloop run`. The HBFSim binary/configuration are
explicit inputs; the profile does not pick a historical backend automatically.

## Meaning and boundaries

- Original static weight ranges/KV writes are retained. Decode KV reads include
  the newly appended token; prefill attention reads dense K/V temporaries.
- Norm, QKV, RoPE, residual, MLP and tail activations use tensor footprints.
  RoPE reads indexed position rows, not the entire allocated lookup table.
- Attention uses per-kernel unions from existing captured pre-cache sectors,
  including touched copies/scratch and explicitly modeled SASS local frames.
  Repeated instructions are collapsed; this is not measured GPU post-L2 traffic.
- Object-relative workspace identities/offsets are retained, with disjoint
  packed placement in the HBM runtime reservation. It is not a reconstruction
  of GPU physical placement. Profile ranges must fit that reservation.
- Zero-duration completion joins retain dependencies without a cross-product
  of edges. Original coarse weight/KV prefetch remains; nonattention operations
  are not a fully matched GPU kernel issue schedule.
- In the included captured window, omitted allocator buffers have0 compiled
  requests except592B/23 pre-cache requests in prefill cumsum. This omission is
  quantified for this input only, not assumed valid for other kernels/shapes.

The profile and its byte digest are recorded in the run inputs. The CLI labels
the traffic as coverage and reports that GPU cache misses are not modeled.
Model/request canonical digests and shape checks prevent accidental reuse.

## Cache decision

Default `--coarse-cache-bound off` retains the coverage approximation. Optional
`--coarse-cache-bound ideal-temporaries` removes only temporary coverage traffic,
while retaining weight/KV requests and dependencies. This is a sensitivity
bound assuming temporary data stays on-chip, not a finite GPU cache model.
It cannot bound effects of weight/KV caching, repeats or real GPU scheduling.

For the included two placements, the bound changes total simulated memory time
by at most~3.25% and preserves their ranking. It does not justify ignoring cache
for physical traffic claims: the detailed reference's modeled cache changes
read/write volumes substantially. Do not apply a single fitted hit ratio to
coverage bytes. Use separately qualified detailed traces when needed; GPU-cache accuracy is
outside this coarse approximation's scope.

## Rebuild a profile

Runtime profiles require no observation/template binaries. An offline export
from a qualified ledger is available separately:

```sh
python -m serveloop.coarse_coverage_export \
  --ledger /path/to/qualified-ledger.json \
  --model examples/coarse-coverage/model.json \
  --requests examples/coarse-coverage/requests.json \
  --out /path/to/new-profile.json
```

The exporter currently recognizes the qualified28-layer call structure only.
Changing its input or adding a profile requires new coverage qualification;
successful schema parsing is not hardware validation. Export refuses to
overwrite an existing file. Verbose audits remain separate from compact
runtime descriptors (~219KB for this profile).

## Model-derived analytic policy (no per-input capture)

Use `--coarse-coverage-policy dense-16bit-swiglu` instead of
`--coarse-coverage-profile FILE` to generate tensor footprints from the model
ledger and request shapes. The two options are mutually exclusive. Example:

```sh
python -m serveloop run \
  --model examples/coarse-coverage/model.json \
  --requests /path/to/requests.json --system /path/to/system.cfg \
  --simulator /path/to/hbfsim --placement weights-hbf-kv-hbm \
  --timing memory_only --prefetch-depth 1 \
  --coarse-coverage-policy dense-16bit-swiglu --out /path/to/fresh-output
```

No captured profile, observation ledger or detailed trace is read by this
provider. New models must carry explicit `structure`: hidden/intermediate
sizes, query/KV head counts, head dimension, BF16/FP16 dtype, matching KV dtype,
rotary dimension, RoPE-table dtype, SwiGLU/RMSNorm and optional Q/K head norms.
Layer count and tied embeddings remain in the model ledger. No weight-byte
reverse inference is used for structured models; incompatible weight/KV
payloads are rejected. Structure is optional in serveloop.model v2: old JSON
canonical contents/digests are unchanged. Adding structure deliberately changes
the model digest, so a captured profile bound to old JSON must still use its
original model. Older strict clients cannot read the new optional field.

Import a supported dense Llama/Qwen/Mistral Hugging Face config without loading
weights or executing the model:

```sh
python -m serveloop model /path/to/config.json --hf-config \
  --dtype bfloat16 --output /path/to/model.json
```

Use `--dtype float16` for FP16. KV defaults to the same dtype; mixed KV,
quantization, MoE and sliding-window attention are currently rejected for this
structure path. Eligible unquantized dense GQA public catalog descriptors also
carry structure when converted. Catalog precision is explicit and cannot be
overridden by the HF dtype flags. Existing native SGLang callers of dense_model
retain their old ledger/digest unless they explicitly request include_structure.
The old `dense-bf16-swiglu` policy name remains a compatibility alias. Legacy
unstructured1.5B JSON retains only its previously qualified geometry fallback;
other models must supply structure.

Scenario support remains B1 unchunked prefill from zero prior context and
single-token decode with positive context; every forward must emit output.
Batching, prefix/chunked prefill and non-output forwards are rejected.
Complete qualification covers1.5B,7B (BF16/FP16, P16/P128+D1) and public
Qwen3-8B (P16+D1), not arbitrary framework/hardware accuracy. Synthetic formula
tests also cover MHA/GQA, query width different from hidden width, partial
rotary dimensions and16-bit RoPE tables. Q/K head norms are represented as
separate footprint stages when declared.

The explicit execution policy assumes fused RoPE/residual-RMSNorm, fused
attention without materialized score matrices, dense prefill K/V copies and
last-token16-bit head plus FP32 logits. Selecting this
policy asserts these implementation assumptions; a model file alone cannot
identify framework fusion. Attention internal scratch/local frames, GEMM
workspace, control accesses and repeated loads are omitted. GPU caches and
compute remain unmodeled. These omissions make this a different approximation
from captured kernel-sector coverage, not an equivalent reconstruction.

Sequential layers reuse packed tensor workspace. Footprint bytes and required
HBM runtime reserve grow with token count; insufficient reserve is rejected by
placement. New activation range count stays fixed with token count, while
original embedding gathers and placement stripe splits may add operations.
`--coarse-cache-bound ideal-temporaries` is also available as the same explicit
sensitivity bound, not a finite cache model. Run receipts include the policy
and assumptions; compiled workload audit includes generated profile evidence.

P16 original and captured-profile scientific results remain unchanged. In the
complete all-HBM example analytic/captured simulated totals are0.828522/0.831579ms;
host execution0.183/0.479s (single samples). Traffic differs by about3–6%, not
hardware error. A compact complete-model comparison is included in [coarse evidence](coarse-evidence.md).

## Optional range L2 approximation

Add `--coarse-l2-capacity-bytes 1048576` to an analytic-policy or captured-profile
run to enable a cold1MiB cache. Omit the option to preserve the existing uncached
path. Cache capacity must be a positive multiple of32B. The named model is a
fully associative32B-sector LRU cache, write-back/write-allocate, shared across
all application objects before HBM/HBF placement. Weights, KV and tensor ranges
compete for the same capacity; controller metadata/NAND traffic is not cached
by this frontend. Capacity is an assumption, not a particular GPU's geometry.

Default `--coarse-l2-write-policy write-allocate-fetch` fetches a sector on every
write miss. Optional `--coarse-l2-write-policy full-store-no-fetch` skips read
fill when the operation overwrites the entire valid sector; partial stores
still fetch initialized bytes. Both are explicit sensitivity policies, not
claims about a particular NVIDIA kernel. Dirty evictions and final dirty drain
are submitted to HBFSim. Dirty requests do not silently disappear.

The current supported run contains one request, B1 output-emitting forwards,
HBM-only KV with no migration; model-derived coverage can use memory_only or roofline. It cannot combine with
ideal-temporaries. Stateful compilation starts at batch0, with consecutive IDs.
Weights/KV cache state persists across forwards. Packed temporary workspace is
flushed and invalidated each forward, since input shapes/positions may rebind
its offsets; this does not model arbitrary cross-forward temporary reuse.
Dirty drain occurs at the last actual forward, before request completion/free.
Prefill/decode traffic is reported separately, but temporary boundary writebacks
are charged to the next forward; full-request drained totals are the cleaner
comparison. Initial warm-cache or no-drain policies are not currently exposed.

This implementation stores LRU intervals, not one Python object per sector.
Ascending range access is exact to the declared fully associative sector model;
interval fragmentation can still grow with access patterns. Bulk fills and
writebacks preserve byte/sector accounting while approximating intra-range
service ordering. Cache accesses use canonical insertion order, not a resolved
GPU warp schedule. Original IDs become zero-duration completion joins; hit and
dirty-owner dependencies preserve causality. Added fills/dependencies can
increase simulated time, even when traffic declines. L2 hit/service latency,
NVIDIA set conflicts, L1/shared-memory behavior and missing instruction repeats
are unmodeled. It is a cache approximation, not hardware calibration.

Run inputs record settings; `result.json` exposes `coarse_l2` with per-forward
hit/miss/fill/writeback counters, resident/dirty sectors and interval peaks.
Canonical batch summaries also contain per-phase cache accounting. Reported
memory traffic is modeled post-L2, not measured hardware traffic. Existing
coverage assumptions describe the pre-cache source; the separate L2 receipt
identifies its transform.

Qualification: an independent test-only scalar LRU oracle checks randomized
range sequences under both write policies, transfer multisets, LRU order, dirty
state and hit/miss counts. Historical captured-derived P16/decode ranges also
matched a separate reference with fully associative32B configuration. This does
not establish equivalence to the historical128B-line set-associative cache or
to hardware. 1.5B P16 has both placements and1MiB/40MiB controls; P128/P512 and7B P16
have allHBM1MiB controls. Captured-profile mode also works, but its P16 host cost
is4.3× the uncached profile, so prefer the analytic provider for fast studies.
See [coarse evidence](coarse-evidence.md) for included measurements and limitations.

## Catalog-wide model-derived policy

`--coarse-coverage-policy model-derived` covers all five packaged public descriptors:
Llama3.1-8B W8, Llama3.1-70B W8A16, Qwen3-8B BF16, Qwen3-235B FP8 MoE/GQA,
and DeepSeek-V3 FP8 MoE/MLA. The original weight, quantization-scale and KV
ledgers remain unchanged. The analytic provider adds external16-bit tensor
footprints, routed token dispatch/expert activation/combine and shared-expert
accesses, plus low-rank query/KV and compressed-context MLA accesses.

For a public descriptor use the same run command, replacing the policy value:

```sh
python -m serveloop run --model models/deepseek-v3-fp8-kv-bf16.json \
  --requests /path/to/requests.json --router examples/ci-synthetic-router.json \
  --system /path/to/system.cfg --simulator /path/to/hbfsim \
  --placement weights-hbf-kv-hbm --timing memory_only --prefetch-depth 1 \
  --coarse-coverage-policy model-derived --out /path/to/new-output
```

MoE still requires the original router trace/provider. Its selected-expert
weight reads are preserved; this rule does not manufacture model routing.
Use the same cache options as the dense provider when desired. Device capacity
and runtime workspace must fit; supporting a descriptor does not mean it fits
an arbitrary system configuration.

To convert a public descriptor into portable native JSON with coverage geometry:

```sh
python -m serveloop model models/llama31-8b-w8-kv-bf16.json \
  --coverage-descriptor --output /path/to/model.json
```

Without this flag, conversion/default run retains its old canonical data/digest.
The optional embedded descriptor is checked against weight/KV/FLOP ledgers;
older strict readers do not recognize the new optional field. Native ledgers
without explicit dimensions cannot reconstruct quantized/MoE architecture.
Existing supported HF-imported dense structures also work with model-derived;
the established dense16 policy and its numerical results remain available.

These are declared footprint assumptions: external activations are16-bit;
quantized GEMMs consume weights/scales directly without an expanded weight
workspace. Activation re-quantization scratch is omitted. MLA uses fused latent
attention without external expanded-context K/V; MoE uses packed top-k copies
without expert-capacity padding, token drops or network traffic. Matrix weight
dtype alone does not identify a backend's activation/fusion implementation.
These assumptions are recorded in the generated profile, not presented as
hardware-qualified kernel traffic. Scope remains B1 full prefill/single-token
decode, one model; scheduler/batch support is unchanged. Model-derived policies also support the layer-aggregate roofline described below.

## Compute-aware model-derived simulation

Both `dense-16bit-swiglu` (including its compatibility alias) and `model-derived`
now accept `--timing roofline` as well as `memory_only`. For example:

```sh
python -m serveloop run --model models/llama31-8b-w8-kv-bf16.json \
  --requests /path/to/requests.json --system /path/to/system.cfg \
  --simulator /path/to/hbfsim --placement all-hbm --prefetch-depth 1 \
  --coarse-coverage-policy model-derived \
  --timing roofline --peak-tflops 200 --efficiency 0.5 --out /path/to/new-output
```

MoE uses the same original router input. Finite-cache options are unchanged.
Captured input-bound profiles still require `memory_only`; `gpu_calibrated` and
linear timing are not admitted by the enhanced providers in this stage.

Compute durations come from the original model FLOP ledger divided by
`peak_tflops * efficiency`; no memory service time is included in that formula.
The original compute node IDs/durations are preserved. MoE attention/router and
post-routing durations sum to one layer budget, not two; cache transformations
preserve those durations. Next-layer weight/KV prefetch can overlap current
compute through the original dependency graph. Covered activation accesses
precede their aggregate layer compute; this is not operator-level compute/memory
overlap. Final cache drain remains part of request completion.

Access-generator assumptions describe footprints only. Separate `coarse_compute`
audit and run input metadata describe the added timing model and explicitly mark
it uncalibrated. External16-bit activation,quantized GEMM and MLA assumptions
remain unchanged. Cache recency still follows the declared canonical range order,
not a measured GPU issue timeline.

Use explicit effective throughput appropriate to the modeled workload. The
example200TFLOP/s and efficiency0.5 (100effectiveTFLOP/s) is a normalized sensitivity
point,not a hardware measurement or an FP8/W8 performance guarantee. One fixed
rate applies to both phases in a run; precision/shape-dependent throughput and
real kernel overlap require separate qualification. `memory_only` remains the
unchanged reference for isolating device traffic effects.

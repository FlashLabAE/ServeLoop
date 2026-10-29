# Enhanced coarse workload generation

The original coarse compiler accounts mainly for weights and KV. The opt-in
analytic policy adds activation and intermediate tensor footprints derived from
model structure and forward shapes. HBFSim still models device service.
Neither path executes inference or requires GPU capture at runtime.

## Standalone usage

Install from this checkout with `python -m pip install -e .`. Convert a supported
Hugging Face config without loading model weights:

```sh
python -m serveloop model /path/to/config.json --hf-config \
  --dtype bfloat16 --output /path/to/model.json
python -m serveloop run \
  --model /path/to/model.json --requests /path/to/requests.json \
  --system /path/to/system.cfg --simulator /path/to/hbfsim \
  --placement weights-hbf-kv-hbm --timing memory_only --prefetch-depth 1 \
  --coarse-coverage-policy dense-16bit-swiglu --out /path/to/new-output
```

Use `all-hbm` for the other placement. Requests must match the model ID.
The simulator and configuration are explicit inputs; there is no bundled
historical executable or dependency on a research workspace.

To examine a finite-cache assumption, add:

```sh
--coarse-l2-capacity-bytes 41943040 \
--coarse-l2-write-policy write-allocate-fetch
```

This is 40MiB (41943040 bytes). An alternative policy,
`full-store-no-fetch`, omits read fill on complete sector overwrites; partial
stores still fetch. Omit cache options for the recommended fast baseline.
Omit all coverage options to keep the original compiler/default behavior.

## Dense16 policy scope

Uniform dense BF16/FP16 SwiGLU/RMSNorm models with explicit hidden/intermediate
sizes, query/KV heads, head and rotary dimensions. B1 full or chunked prefill, followed by single-token output-emitting decode
forwards; `memory_only`, layer-aggregate `roofline`, or `linear` sensitivity timing. Analytic coverage rejects unsupported quantization, MoE,
sliding-window attention, multi-request batches and mixed KV precision.
A model structure does not identify framework fusion: fused attention, RoPE,
residual normalization and last-token logits are declared assumptions.
Instruction repeats, internal attention/GEMM scratch and compute are omitted.

## Cache boundary

The cache is a cold, fully associative 32B-sector interval LRU, write-back and
write-allocate, over all application objects before placement. Weight/KV state
persists across forwards. Temporary workspace is flushed/rebound each forward,
and the final forward drains dirty data. Controller metadata/NAND traffic is
outside this cache. It does not model GPU sets, L1, warp issue order or L2 hit
latency, and is not calibrated to NVIDIA hardware. Input and result receipts
record assumptions, fills, writebacks, hit/miss counts and interval/state peaks.

The included input-bound captured profile is a separate fixed-input diagnostic;
it is not required for analytic generation. See [technical details](coarse-coverage.md)
and [included evidence](coarse-evidence.md). Coarse approximation supports
fast device studies; hardware traffic and memory-only simulated time must not
be presented as validated inference latency.

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
hardware-qualified kernel traffic. Scope remains B1 full/chunked prefill and single-token
decode, one model; scheduler/batch support is unchanged. Model-derived policies also support the layer-aggregate roofline described below.

## Compute-aware model-derived simulation

Both `dense-16bit-swiglu` (including its compatibility alias) and `model-derived`
accept `--timing roofline` as well as `memory_only`, and linear sensitivity
timing through the original `--run-config` interface. For example:

```sh
python -m serveloop run --model models/llama31-8b-w8-kv-bf16.json \
  --requests /path/to/requests.json --system /path/to/system.cfg \
  --simulator /path/to/hbfsim --placement all-hbm --prefetch-depth 1 \
  --coarse-coverage-policy model-derived \
  --timing roofline --peak-tflops 200 --efficiency 0.5 --out /path/to/new-output
```

MoE uses the same original router input. Finite-cache options are unchanged.
Captured input-bound profiles still require output-emitting covered shapes and
`memory_only`. Model-derived providers continue to reject `gpu_calibrated`.

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

## B1 chunked prefill and linear sensitivity

Set `scheduler.max_batch_requests` to1 in the original run-config format, then
choose `prefill_chunk_tokens` and `max_batch_tokens`. Each prefill chunk reads
existing KV through the original compiler and covers newly computed QKV and
intermediates. Non-emitting chunks retain final normalization but omit token
selection, LM-head and logits accesses. Only the output-emitting chunk includes
the head. Existing single-request cache options also apply: temporary workspace
is rebound each chunk and dirty data drains only at the request's last forward.

The original linear provider's fixed/per-token layer costs and emitting-request
tail cost are reused unchanged. MoE nonzero layer costs require an explicit
`moe_routing_fraction`, as in the original compiler. Linear results remain
uncalibrated sensitivity results, not measured inference timing.

```json
{
  "schema": {"name": "serveloop.run_config", "version": 2},
  "scheduler": {"max_batch_requests": 1, "max_batch_tokens": 8, "prefill_chunk_tokens": 8},
  "timing": {"type": "linear", "prefetch_depth": 1,
    "fixed_ns_per_layer": 100, "ns_per_token_per_layer": 10,
    "tail_fixed_ns": 50, "tail_ns_per_output_request": 20, "moe_routing_fraction": 0.2},
  "compute": null
}
```

Pass this JSON with `--run-config /path/to/run.json`; omit CLI timing/prefetch
overrides. For memory-only or roofline chunking, keep the same scheduler section
and use the corresponding original timing/compute sections. Batching several
requests into one forward, mixed-phase batches, multiple models and cache-aware
KV migration remain outside enhanced coverage scope.

## Current coverage and remaining work

The enhanced generator is opt-in; omitting coverage options retains the original
compiler and its broader scheduling interfaces. Unsupported enhanced inputs
raise errors rather than silently claiming complete coverage.

| Area | Current support | Remaining work |
|---|---|---|
| Models | Five packaged dense/quantized/MoE GQA/MLA descriptors; parameterized supported dense structures | New architectures, non-SwiGLU/non-RMSNorm rules, other activation/KV precisions and fusion policies |
| Scheduling | One model; B1 full/chunked prefill and one-token decode; intermediate chunks omit head/logits | Multiple requests in one forward, mixed prefill/decode batches, multiple-model enhancement |
| Compute | Original memory-only, layer roofline and linear sensitivity providers | GPU-calibrated provider integration, operator-level overlap, precision/shape-dependent throughput |
| Optional cache | One complete request; HBM-only KV; persistent weight/KV state, per-forward workspace rebinding, final drain | Multi-request lifetimes, KV tier migration and coherence |
| Validation | Contract tests, default canonical regression, public-backend native runs and installed-wheel smoke | Hosted OS/Python matrix; accuracy qualification for any newly introduced model/runtime assumptions |

The next scheduling extension should first target homogeneous batches without
the optional cache. It must preserve shared weight reads, request-specific KV,
output-head selection and original compute budgets while defining buffer
lifetimes. Mixed batches and stateful-cache migration follow separately. These
items are future work and are not required to use the current supported scope.
The current contribution improves workload coverage for HBFSim; native software
checks do not establish measured hardware traffic or inference accuracy.

KV production and history/current-read dependencies are described in [coarse causality](coarse-causality.md). These fixes preserve the existing weight release policy.

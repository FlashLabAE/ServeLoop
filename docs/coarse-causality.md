# Coarse KV causal dependencies

The causal fix establishes readiness and consumer dependencies independently of
weight scheduling. Its default preserves main's weight release parents and
on-demand expert reads. An optional [weight scheduling policy](coarse-scheduling.md)
is a separate layer; no new weight layout or expert prefetch is introduced.

## Problem and behavior

Current-layer KV writes previously waited only for the previous layer or embedding;
they could therefore publish before the current projection's required weights.
Historical context reads unnecessarily waited for all current writes in the layer,
including other requests' writes. Generated tensor coverage was attached after the
whole-layer memory envelope, so its declared QKV/RoPE or MLA source did not govern
KV publication; adding current-token reads to that envelope could otherwise make
an incorrect producer/consumer cycle.

The base compiler adds a zero-time input/attention-weight readiness join for
current writes and removes the spurious historical-read dependency. This remains
an aggregate readiness abstraction: it does not time QKV compute or prove that
whole-weight completion is a minimal hardware barrier.

Generated dense/GQA, MoE/GQA and MoE/MLA profiles explicitly declare their producer
and consumer roles. Existing serial call envelopes are retained while binding
current KV publication to its declared source and attention to its KV reads.
Router/top-k completion precedes the original routing-time barrier and dispatch.
Weight release parents and original layer/routing compute durations are preserved.
No expert prefetch mechanism or new scheduling default is introduced.

Decode profile reads are separated into adjacent historical and current-token
ranges. History has no dependency on this forward's publication; the current read
waits for its corresponding write. Their combined logical payload is unchanged.
Current reads use the `attention/kv_read_current` audit role. The completion frontier
still joins all required readers and writes. Prefill clones/fused MLA workspace
consumers need their source data, not an artificial persistent-store dependency.

## Boundaries and compatibility

Captured profiles without an explicit semantic contract receive only the base
readiness abstraction; operator names are not interpreted as a dataflow graph.
Generated profiles fail if their required producer contract is missing or corrupt.
The `ideal-temporaries` sensitivity bound retains zero-time producer joins even
when temporary memory operations are omitted.

Dependency/canonical IDs, profile hashes, request partition, cache behavior and
simulated time can change. Logical byte conservation does not promise unchanged
post-cache traffic or old timing. Reproduce historical results with their original
revision/configuration. Generated B1/full-or-chunked-prefill/single-token-decode
and captured fixed-shape admission domains are unchanged; base multi-request
compilation also receives the readiness correction.

The patch does not add instruction re-reads, cache calibration, scratch traffic,
measured compute stages or kernel-level overlap. Software DAG/ledger tests do not
establish hardware timing accuracy. Conservative whole-layer/weight envelopes
remain explicit approximations.

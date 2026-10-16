# Configuration provenance

The checked-in system configurations are exploratory simulator profiles, not
vendor performance guarantees. They are included so users can exercise
placement and protocol behavior with a compatible HBFSim build.

## Relationship to HBFSim

HBFSim owns the hardware model, its profiles and its client protocol. Three
parts of this repository are verbatim copies of the same paths in
[HBFSim](https://github.com/FlashLaboratory/HBFSim):

- `configs/systems/` (complete system profiles),
- `configs/parameter-provenance.json` (source and evidence grade of every
  physical default), and
- `hbfsim_client/` (the persistent-session client).

They currently match HBFSim `0e46ac0`. Because the relative paths are the same
in both repositories, `--system configs/systems/miniquick/4hbm-4hbf.cfg`
selects the same profile whether a command runs from ServeLoop or from HBFSim.
`configs/ocp-profile-source.json` records the HBFSim commit, the SHA-256 of
each copied profile and of the provenance registry, and every parameter the
latest synchronization changed.

`systems/4hbm-4hbf.cfg`, `systems/2hbm-6hbf.cfg` and `systems/6hbm-2hbf.cfg`
split eight package positions between HBM and HBF stacks.
`systems/eight-stack-baseline.cfg` declares eight HBM and eight HBF stacks and
is the base profile of the fixed-window experiments. Link rates,
queue depths, controller timing, flash-media timing, overprovisioning, and
thermal parameters are modeling assumptions. The `systems/miniquick/` variants
reduce capacity and host work for fast integration tests; they are not smaller
hardware product claims. `systems/sglang-small.cfg` is a tiny overlay for the
native SGLang frontend.

The remaining files are ServeLoop's own. `simulation-session-mini.cfg`,
`cxl-memory.cfg` and `nvme-ssd.cfg` exercise the persistent session protocol
and external backing alternatives. `windows/` contains matched
miniquick/full-scale experiments; its paths resolve relative to each
experiment JSON. `overlays/` provides their capacity controls, backing media
and thermal assumptions. See [Fixed windows](windows.md) for scale and
completion boundaries.

The `overlays/backing/calibrated/dana-a100-*` coefficients are inherited from
HBFSim's A100 host-offload fits, not portable DRAM/NVMe specifications. HBFSim
publishes the fit under `evidence/hardware/dana_a100_offload/`; it is not
copied here. Treat the overlays as example parameterizations, not
self-contained calibration evidence; validate or replace them for your setup.

For publishable results, freeze the complete config set, hash it in the run
receipt, cite a source or calibration artifact for every physical parameter,
and validate against a holdout workload. Parameters without such evidence must
remain labeled assumptions or sensitivity variables.

## Calibrated GPU operators

A run config may select `timing.type = "gpu_calibrated"`, with an absolute
`profile` JSON path, `model_bindings` (model ID to `8b`, `70b`, or `235b`) and
`prefetch_depth: 0`; set `compute` to null. The CLI binds the packed weight
layout before placement and uses the profile's 256-token KV blocks. Supply
sufficient HBM runtime scratch. The provider reports the profile hash and
measured validation errors in the result; `calibrated` does not imply a
production SLO guarantee.

No measured operator profile is distributed with ServeLoop or with the public
HBFSim repository; supply one measured on your own hardware. The v3 runtime
rejects old contiguous-KV v2 profiles. Its native client requires dynamic
memory-span barrier support.

Scheduler semantics match the tested vLLM 0.26.0 synchronous FCFS configuration:
running requests (including unfinished prefills) in admission order, then
waiting requests; incremental KV allocation; complete prefix blocks shareable
within a batch; youngest-running-request preemption. Prefix release frees
suffixes first, and active entries remain protected during capacity eviction.
This does not claim vLLM async scheduling, speculative decoding, full-prompt
reservation, TP/EP or every scheduler version.

## Backend compatibility

Supported backend: HBFSim `0e46ac0d073bab5df3c9e7e7ecbfac42e3756b6f`.
In the other direction, HBFSim's integration tests accept ServeLoop at or after
`HBFSIM_SERVELOOP_MIN_REVISION` in its `CMakeLists.txt`.

A client and an engine from different HBFSim revisions are not
interchangeable: the client validates receipt fields (transaction census,
host-DRAM and external accounting, wear snapshots, resolved geometry) that
change together with the engine. Running from the ServeLoop directory selects
the bundled client; running from an HBFSim checkout selects HBFSim's. At the
supported revision the two are identical.

The latest synchronization replaced HBM per-command timing with HBFSim's
aggregate channel model. The engine rejects the removed keys. This is a
physical model change, not a key rename, so HBM timing simulated with the
previous profiles is not reproduced. HBF, capacity, thermal and backing values
are unchanged.

- Removed: `hbm-tccd-s-cycles`, `hbm-tccd-l-cycles`, `hbm-tcl-ns`,
  `hbm-tcwl-ns`, `hbm-tfaw-ns`, `hbm-tras-ns`, `hbm-trc-ns`, `hbm-trcdrd-ns`,
  `hbm-trcdwr-ns`, `hbm-trp-ns`, `hbm-trrd-s-ns`, `hbm-trrd-l-ns`,
  `hbm-trtp-ns`, `hbm-trtw-ns`, `hbm-twr-ns`, `hbm-twtr-s-ns`,
  `hbm-twtr-l-ns`, `hbm-refresh`, `hbm-same-bank-refresh`, `hbm-trefi-ns`,
  `hbm-trfc-ns`, `hbm-trfcsb-ns`, `hbm-trrefd-ns`.
- Added: `hbm-read-latency-ns=42`, `hbm-write-latency-ns=38`,
  `hbm-read-to-write-ns=8`, `hbm-write-to-read-ns=8`,
  `hbm-bandwidth-efficiency=0.94` (average command and refresh overhead),
  `hbm-service-group-channels=4`, `hbm-service-quantum-bytes=4096`.

HBFSim reserves the HBF controller buffers in HBM, so a fixed-window topology
with HBF stacks but no HBM cannot execute; the experiment loader rejects it
before any simulation starts.

Run the focused checks with explicit paths (no editable install required):

```sh
python tests/test_config_compatibility.py \
  --hbfsim-root /path/to/HBFSim --simulator /path/to/hbfsim
python tests/test_config_compatibility.py --client external \
  --hbfsim-root /path/to/HBFSim --simulator /path/to/hbfsim
python -B tests/test_session_compatibility.py --simulator /path/to/hbfsim
```

Without `--simulator`, the session suite runs portable receipt acceptance and
corruption checks in CI; native HBM-only, Host-DRAM/NVMe coexistence and pending
wear/checkpoint tests require the explicit executable.

These checks verify config acceptance, profile provenance values, capacity
invariants and window config composition. Synthetic session checks only certify
software wiring. They do not validate inference performance, GPU cache accuracy,
phase-specific hardware errors, or a new model/context. Freeze new configurations
for any rerun; simulated time, traffic and placement may change, and old fits,
calibration coefficients and reported errors must not automatically carry over.

## MoE prefix routing identity

Python integrations may enable MoE prefix caching by passing the same router to
`ServeLoopCompiler` and `ServeLoopPlacement`. The router must implement
`PrefixRouterProvider.prefix_block_keys(request=..., model=..., block_tokens=...)`.
Return one key for each complete prompt block. Equal keys must imply the same
model, causal token prefix and expert decisions for all layers through that
block. Include the parent chain, model, cache salt and block geometry.

A router determined solely by causal token identity may namespace content keys
with its algorithm and parameters. A recorded or request-indexed router must
also include the relevant expert decisions; identical tokens alone are
insufficient. Providers without this contract still cannot enable MoE prefix
caching. Placement checks that each compiled batch uses the same router digest
and records `prefix_router_sha256` in its receipt.

## Optional operator coverage

The coarse memory-only path can load `--coarse-coverage-profile FILE`; the
original weight/KV path remains default. See [coverage scope and cache
sensitivity](coarse-coverage.md) for the packaged input-bound1.5B example.
This option does not select GPU compute calibration or claim post-cache traffic.

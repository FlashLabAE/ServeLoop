#!/usr/bin/env python3
"""One workload frontend with explicit closed-loop and fixed-window inputs.

``python -m serveloop run`` takes a model (catalog descriptor or
``serveloop.model`` JSON), a system config with optional overlays, a request
source (synthetic spec or request trace), and a placement preset or JSON,
derives the placement capacities from the system config, runs the closed
loop, writes every input and the receipted result into a fresh timestamped
directory, and prints a short headline. Alternatively, ``--experiment``
selects a matched fixed memory window, without request scheduling or compute.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

from hbfsim_client.sampling import Sampler, SamplingPlan  # noqa: E402
from hbfsim_client.simulation_session import (  # noqa: E402
    ResolvedSystemConfig,
    SimulationSessionError,
)
from hbfsim_client.provenance import add_allow_dirty_argument
from serveloop.catalog import load_model_any  # noqa: E402
from serveloop.compiler import ServeLoopCompiler  # noqa: E402
from serveloop.contracts import (  # noqa: E402
    ServeLoopError,
    ModelSpec,
    RequestTrace,
    SchedulerPolicy,
    TimingProvider,
    canonical_sha256,
)
from serveloop.engine import ServeLoopEngine  # noqa: E402
from serveloop.hbfsim import HbfSimExecutor, describe_backend  # noqa: E402
from serveloop.io import (  # noqa: E402
    REQUEST_TRACE_SCHEMA,
    RUN_CONFIG_SCHEMA,
    SYNTHETIC_REQUEST_SCHEMA,
    create_run_directory,
    load_json_object,
    load_router,
    placement_from_dict,
    run_config_from_dict,
    synthetic_request_config_from_dict,
    write_json_atomic,
)
from serveloop.placement import (  # noqa: E402
    KvPlacement,
    ServeLoopPlacement,
    PlacementSpec,
)
from serveloop.prefix_image import (  # noqa: E402
    PrefixImage,
    install_prefix_image,
    load_prefix_image,
)
from serveloop.synthetic import generate_requests  # noqa: E402


EXPERIMENT_SCHEMA = {
    "name": "serveloop.experiment",
    "version": 2,
}
DEFAULT_PEAK_TFLOPS = 989.0  # H100 SXM dense BF16 tensor peak
DEFAULT_EFFICIENCY = 0.6
DEFAULT_SCHEDULER = {
    "max_batch_requests": 256,
    "max_batch_tokens": 8192,
    "prefill_chunk_tokens": 2048,
}
# Placement presets: weight tier for every model and the cold KV tier.
PLACEMENT_PRESETS: dict[str, dict[str, Any]] = {
    "weights-hbf-kv-hbm": {"weights": "hbf", "kv_cold": None},
    "all-hbm": {"weights": "hbm", "kv_cold": None},
    "weights-hbf-kv-hbf-cold": {"weights": "hbf", "kv_cold": "hbf"},
}
HBM_RUNTIME_RESERVE_FRACTION = 0.05


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ServeLoopError(
            f"input artifact must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve()
    if not resolved.is_file():
        raise ServeLoopError(
            f"input artifact must be a regular non-symlink file: {resolved}"
        )
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def derive_placement(
    *,
    system: ResolvedSystemConfig,
    models: Mapping[str, ModelSpec],
    preset: str,
) -> PlacementSpec:
    """Derive placement capacities from the system config for one preset."""

    try:
        rule = PLACEMENT_PRESETS[preset]
    except KeyError as error:
        raise ServeLoopError(
            f"unknown placement preset {preset!r}; choose one of "
            f"{', '.join(sorted(PLACEMENT_PRESETS))} or pass a placement JSON"
        ) from error
    geometry = system.hbf_geometry
    hbm_capacity = system.hbm_capacity_bytes
    if system.logical_hbf_capacity_bytes is None:
        raise ServeLoopError("placement requires geometry resolved by the simulator")
    hbf_capacity = system.logical_hbf_capacity_bytes
    external_capacity = 0
    external_page = 4096
    if system.values.get("external-backing-kind") is not None:
        identity = system.external_backing_identity
        external_capacity = int(identity["capacity_bytes"])
        external_page = int(identity["page_size_bytes"])
    weights_tier = str(rule["weights"])
    if weights_tier == "hbf" and hbf_capacity == 0:
        raise ServeLoopError("preset places weights in HBF but the system has none")
    alignment = 4096
    return PlacementSpec(
        hbm_capacity_bytes=hbm_capacity,
        hbf_capacity_bytes=(
            hbf_capacity
            if weights_tier == "hbf" or rule["kv_cold"] == "hbf"
            else 0
        ),
        external_capacity_bytes=external_capacity,
        hbm_runtime_reserve_bytes=_align_up(
            int(hbm_capacity * HBM_RUNTIME_RESERVE_FRACTION), alignment
        ),
        hbm_model_cache_bytes=0,
        model_weight_tiers={model_id: weights_tier for model_id in models},
        object_tier_overrides={},
        hbm_alignment_bytes=alignment,
        hbf_page_size_bytes=geometry.page_size_bytes,
        external_page_size_bytes=external_page,
        kv_block_tokens=16,
        kv_placement=KvPlacement(hot="hbm", cold=rule["kv_cold"]),
    )


def hbm_stripe_bytes(system: ResolvedSystemConfig) -> int:
    """Bytes one pass over every HBM pseudo-channel covers (the map stripe).

    Mirrors the engine: pseudo-channels x interleave bytes, where an unset
    ``hbm-interleave-bytes`` selects the largest multiple of the burst that
    does not exceed 256 B.  The value only steers host-time-neutral
    alignment and splitting in the placement; a wrong guess costs speed,
    never correctness.
    """

    pseudo_channels = (
        system.integer("hbm-stacks")
        * system.integer("hbm-channels")
        * system.integer("hbm-pseudo-channels")
    )
    burst = system.hbm_burst_bytes
    try:
        interleave = system.integer("hbm-interleave-bytes", minimum=0)
    except SimulationSessionError:
        interleave = 0
    if interleave == 0:
        interleave = max(burst, 256 // burst * burst)
    return pseudo_channels * interleave


def load_requests_any(path: Path) -> RequestTrace:
    """Load a request trace or generate one from a synthetic spec."""

    document = load_json_object(path, "requests")
    schema = document.get("schema")
    if schema == SYNTHETIC_REQUEST_SCHEMA:
        return generate_requests(synthetic_request_config_from_dict(document))
    if schema == REQUEST_TRACE_SCHEMA:
        from serveloop.io import load_request_trace

        return load_request_trace(path)
    raise ServeLoopError(
        f"{path} is neither a {SYNTHETIC_REQUEST_SCHEMA['name']} spec nor a "
        f"{REQUEST_TRACE_SCHEMA['name']} trace"
    )


def default_run_config(
    *,
    timing: str,
    peak_tflops: float,
    efficiency: float,
    prefetch_depth: int,
) -> dict[str, Any]:
    if timing == "roofline":
        timing_section: dict[str, Any] = {
            "type": "roofline",
            "prefetch_depth": prefetch_depth,
        }
        compute: dict[str, Any] | None = {
            "peak_tflops": peak_tflops,
            "efficiency": efficiency,
        }
    elif timing == "memory_only":
        timing_section = {"type": "memory_only", "prefetch_depth": prefetch_depth}
        compute = None
    else:
        raise ServeLoopError("default timing must be roofline or memory_only")
    return {
        "schema": dict(RUN_CONFIG_SCHEMA),
        "scheduler": dict(DEFAULT_SCHEDULER),
        "timing": timing_section,
        "compute": compute,
    }


def _format_bytes(value: float) -> str:
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{int(value)} B"


def _format_ns(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1e9:
        return f"{value / 1e9:.3f} s"
    if value >= 1e6:
        return f"{value / 1e6:.3f} ms"
    if value >= 1e3:
        return f"{value / 1e3:.3f} us"
    return f"{value:.0f} ns"


def headline_lines(
    experiment: Mapping[str, Any],
    *,
    result_path: Path,
    placement_label: str,
) -> list[str]:
    """At most ten lines that say what ran and what came out."""

    serveloop = experiment["serveloop"]
    summary = serveloop["summary"]
    timing = serveloop["timing"]
    scheduler = serveloop["scheduler"]
    models = serveloop["models"]
    trace = experiment["inputs"]["request_trace"]
    session = experiment["final_execution"]["session"]
    devices = session["final_measurement"]["device_workload_totals"]
    percentiles = summary["latency_percentiles"]
    lines = [
        (
            f"serveloop run: {summary['requests']} requests, "
            f"{trace['prompt_tokens']} prompt + {summary['output_tokens']} output "
            f"tokens; model {', '.join(sorted(models))}; "
            f"system {experiment['inputs']['system_label']}"
        )
    ]
    provider = timing["provider"]
    if timing["model"] == "roofline":
        compute_label = (
            f"peak {provider['peak_tflops']:g} TFLOPS x {provider['efficiency']:.2f} "
            f"efficiency, prefetch depth {timing['prefetch_depth']}"
        )
    elif timing["model"] == "gpu_calibrated":
        compute_label = "calibrated GPU operators; inferred compute/memory overlap"
    elif timing["model"] == "linear":
        compute_label = (
            f"{provider['fixed_ns_per_layer']:g} ns + "
            f"{provider['ns_per_token_per_layer']:g} ns/token per layer"
        )
    else:
        compute_label = "no compute; memory critical path only"
    lines.append(
        f"timing model: {timing['model']} ({compute_label}); placement: "
        f"{placement_label}"
    )
    kinds = scheduler["iterations_by_kind"]
    migration = scheduler["kv_migration_bytes"]
    lines.append(
        f"iterations: {scheduler['batches']} (prefill {kinds['prefill']}, decode "
        f"{kinds['decode']}, mixed {kinds['mixed']}); preemptions: "
        f"{summary['preemptions']}; KV migrated out/in: "
        f"{_format_bytes(migration['swap_out'])} / "
        f"{_format_bytes(migration['swap_in'])}"
    )
    if summary["includes_compute"]:
        ttft = percentiles["ttft_ns"] or {}
        tpot = percentiles["tpot_ns"] or {}
        lines.append(
            f"TTFT mean {_format_ns(summary['mean_ttft_ns'])} "
            f"(p50 {_format_ns(ttft.get('p50'))}, p95 {_format_ns(ttft.get('p95'))}); "
            f"TPOT mean {_format_ns(summary['mean_tpot_ns'])} "
            f"(p50 {_format_ns(tpot.get('p50'))}, p95 {_format_ns(tpot.get('p95'))})"
        )
        throughput = summary["output_token_throughput_per_second"]
        lines.append(
            "output throughput "
            + ("n/a" if throughput is None else f"{throughput:,.1f} tokens/s")
            + f"; span {_format_ns(summary['duration_ns'])}"
        )
    else:
        lines.append(
            "memory critical path: first token mean "
            f"{_format_ns(summary['mean_memory_critical_path_first_token_ns'])}, "
            "per token mean "
            f"{_format_ns(summary['mean_memory_critical_path_per_token_ns'])} "
            "(not TTFT/TPOT: no compute in this timing model)"
        )
        lines.append(f"span {_format_ns(summary['duration_ns'])}")
    hbm = devices.get("hbm") or {}
    hbf = devices.get("hbf") or {}
    external = devices.get("external") or {}
    lines.append(
        f"bytes: HBM R {_format_bytes(hbm.get('read_bytes', 0))} / W "
        f"{_format_bytes(hbm.get('write_bytes', 0))}; HBF R "
        f"{_format_bytes(hbf.get('logical_read_bytes', 0))} / W "
        f"{_format_bytes(hbf.get('logical_write_bytes', 0))}; external R "
        f"{_format_bytes(external.get('read_bytes', 0))} / W "
        f"{_format_bytes(external.get('write_bytes', 0))}"
    )
    logical_writes = int(hbf.get("logical_write_bytes", 0))
    physical_writes = int(hbf.get("physical_write_bytes", 0))
    waf = (
        "n/a (no HBF writes)"
        if logical_writes == 0
        else f"{physical_writes / logical_writes:.3f}"
    )
    warnings: list[str] = []
    if not summary["includes_compute"]:
        warnings.append("memory-only timing reports no TTFT/TPOT")
    if summary["preemptions"]:
        warnings.append(f"{summary['preemptions']} preemption(s)")
    if not serveloop["eligibility"]["source_qualified_request_trace"]:
        warnings.append("synthetic requests")
    lines.append(
        f"HBF WAF: {waf}; warnings: {'; '.join(warnings) if warnings else 'none'}"
    )
    lines.append(f"output: {result_path}")
    return lines


def run_experiment(
    *,
    simulator: Path,
    system_config_paths: Sequence[Path],
    models: Mapping[str, ModelSpec],
    request_trace: RequestTrace,
    router: Any,
    placement_spec: PlacementSpec,
    policy: SchedulerPolicy,
    timing: TimingProvider,
    prefetch_depth: int,
    input_artifacts: Mapping[str, Any],
    coarse_coverage_profile: Path | None = None,
    coarse_cache_bound: str = 'off',
    coarse_coverage_policy: str | None = None,
    coarse_schedule_policy: str | None = None,
    coarse_l2_capacity_bytes: int | None = None,
    coarse_l2_write_policy: str = 'write-allocate-fetch',
    prefix_image: PrefixImage | None = None,
    sampling: SamplingPlan | None = None,
) -> dict[str, Any]:
    """Run the closed loop and return the receipted experiment document.

    A ``prefix_image`` (from ``serveloop snapshot``) is installed in the
    placement's prefix cache before the first request, so the run starts
    from that retained state. A ``sampling`` plan times only a sample of the
    iterations in full and estimates the rest from their strata; the
    report lands in the document's ``sampling`` section.
    """

    if coarse_coverage_profile is not None and coarse_coverage_policy is not None:
        raise ServeLoopError('choose a captured profile or an analytic policy')
    if coarse_cache_bound != 'off' and coarse_coverage_profile is None and coarse_coverage_policy is None:
        raise ServeLoopError('coarse cache sensitivity requires a coverage profile or policy')
    if coarse_l2_capacity_bytes is None and coarse_l2_write_policy!='write-allocate-fetch':
        raise ServeLoopError('coarse L2 write policy requires capacity')
    if coarse_l2_capacity_bytes is not None:
        if coarse_coverage_profile is None and coarse_coverage_policy is None:
            raise ServeLoopError('coarse L2 requires a coverage profile or policy')
        if coarse_cache_bound!='off':
            raise ServeLoopError('coarse L2 cannot combine with ideal-temporaries')
        if placement_spec.kv_placement.hot!='hbm' or placement_spec.kv_placement.cold is not None:
            raise ServeLoopError('coarse L2 currently requires HBM-only KV without migration')
    try:
        system = ResolvedSystemConfig.load(list(system_config_paths))
    except SimulationSessionError as error:
        raise ServeLoopError(str(error)) from error
    compiler_type = ServeLoopCompiler
    compiler_options = {}
    if coarse_coverage_profile is not None:
        from serveloop.coarse_coverage import CoarseCoverageCompiler, CoarseCoverageProfile
        compiler_type = CoarseCoverageCompiler
        compiler_options = dict(coverage_profile=CoarseCoverageProfile.load(coarse_coverage_profile),
                                cache_bound=coarse_cache_bound)
    if coarse_coverage_policy is not None:
        from serveloop.coarse_analytic import AnalyticCoverageCompiler, POLICY_ALIASES
        if coarse_coverage_policy == 'model-derived':
            from serveloop.coarse_catalog import CatalogCoverageCompiler
            compiler_type = CatalogCoverageCompiler
        elif coarse_coverage_policy in POLICY_ALIASES:
            compiler_type = AnalyticCoverageCompiler
        else:
            raise ServeLoopError('unsupported analytic coverage policy')
        compiler_options = dict(cache_bound=coarse_cache_bound)
    if coarse_schedule_policy is not None:
        from serveloop.coarse_schedule import validate_schedule_policy
        from serveloop.coarse_analytic import POLICY_ALIASES
        validate_schedule_policy(coarse_schedule_policy)
        if coarse_coverage_policy not in (*POLICY_ALIASES, 'model-derived'):
            raise ServeLoopError('coarse schedule policy requires generated analytic coverage')
        compiler_options['schedule_policy'] = coarse_schedule_policy
    compiler = compiler_type(
        models=models,
        request_trace=request_trace,
        router=router,
        timing=timing,
        prefetch_depth=prefetch_depth,
        **compiler_options,
    )
    if coarse_l2_capacity_bytes is not None:
        from serveloop.coarse_l2 import CoarseL2Compiler
        compiler=CoarseL2Compiler(compiler,coarse_l2_capacity_bytes,write_policy=coarse_l2_write_policy)
    placement = ServeLoopPlacement(
        models=models,
        spec=placement_spec,
        hbf_geometry=system.hbf_geometry,
        hbm_stripe_bytes=hbm_stripe_bytes(system),
        router=router,
    )
    installed_image = None if prefix_image is None else install_prefix_image(placement, prefix_image)
    sampler = None if sampling is None else Sampler(sampling)
    executor = HbfSimExecutor(
        simulator_path=simulator,
        system_config_paths=list(system_config_paths),
        placement=placement,
        sampler=sampler,
    )
    try:
        serveloop_result = ServeLoopEngine(
            models=models,
            request_trace=request_trace,
            compiler=compiler,
            executor=executor,
            policy=policy,
        ).run()
    finally:
        executor.close()
    final_execution = executor.final_receipt()
    source = final_execution["session"]
    if (
        source["simulator_executable"] != input_artifacts["simulator"]
        or source["system_configs"] != input_artifacts["system_configs"]
    ):
        raise ServeLoopError(
            "HBFSim execution identity differs from the pre-run artifacts"
        )
    experiment = {
        "schema": EXPERIMENT_SCHEMA,
        "execution_mode": "closed_loop",
        "result": "pass",
        "inputs": dict(input_artifacts),
        "prefix_image": installed_image,
        "sampling": None if sampler is None else sampler.report(),
        "serveloop": serveloop_result,
        "final_execution": final_execution,
    }
    if coarse_l2_capacity_bytes is not None:
        experiment['coarse_l2']=compiler.cache_receipt()
    experiment["experiment_sha256"] = canonical_sha256(experiment)
    return experiment


SERVING_ARGUMENTS = (
    "model", "system", "overlay", "router", "placement", "run_config",
    "prefix_cache_bytes", "prefix_cache_ttl_ns",
)
TIMING_DEFAULTS = {
    "timing": "roofline",
    "peak_tflops": DEFAULT_PEAK_TFLOPS,
    "efficiency": DEFAULT_EFFICIENCY,
    "prefetch_depth": 1,
}


def add_serving_arguments(group: Any) -> None:
    """The closed loop's inputs besides the requests, shared by the commands that serve or replay."""

    group.add_argument(
        "--model",
        type=Path,
        action="append",
        help="serveloop.public_model descriptor or "
        "serveloop.model JSON; required with --requests; repeat for multi-model serving",
    )
    group.add_argument(
        "--system", type=Path, help="system config; required with --requests"
    )
    group.add_argument(
        "--overlay",
        type=Path,
        action="append",
        help="overlay config applied after --system (repeatable)",
    )
    group.add_argument("--router", type=Path, help="MoE router trace or synthetic router")
    group.add_argument(
        "--placement",
        help="preset (" + ", ".join(sorted(PLACEMENT_PRESETS)) + ") or placement JSON; "
        "required with --requests",
    )
    group.add_argument(
        "--run-config",
        type=Path,
        help="serveloop.run_config JSON; default: token-budgeted mixed iterations "
        "with roofline timing",
    )
    group.add_argument(
        "--timing",
        choices=("roofline", "memory_only"),
        help="default: roofline; cannot combine timing knobs with --run-config",
    )
    group.add_argument(
        "--peak-tflops", type=float, help=f"default: {DEFAULT_PEAK_TFLOPS}"
    )
    group.add_argument(
        "--efficiency", type=float, help=f"default: {DEFAULT_EFFICIENCY}"
    )
    group.add_argument("--prefetch-depth", type=int, help="default: 1")
    group.add_argument("--prefix-cache-bytes", type=int, help="bounded full-block prefix cache in the shared HBM KV pool")
    group.add_argument("--prefix-cache-ttl-ns", type=float, help="optional prefix entry lifetime from publication; requires a nonzero cache budget")


@dataclass(frozen=True)
class ServingInputs:
    """Everything a closed-loop run or replay starts from, loaded and bound."""

    models: Mapping[str, ModelSpec]
    request_trace: RequestTrace
    router: Any
    system_paths: tuple[Path, ...]
    system: ResolvedSystemConfig
    placement_spec: PlacementSpec
    placement_label: str
    run_config: Mapping[str, Any]
    policy: SchedulerPolicy
    timing: TimingProvider
    prefetch_depth: int
    input_artifacts: dict[str, Any]


def _resolve_coarse_run_config(run_config, args):
    """Do not let shared snapshot inputs silently ignore this policy."""
    from serveloop.coarse_analytic import POLICY_ALIASES
    from serveloop.coarse_schedule import validate_schedule_policy
    resolved = dict(run_config)
    value = getattr(args, 'coarse_schedule_policy', None)
    if value is not None:
        if 'coarse_schedule_policy' in resolved:
            raise ServeLoopError('choose coarse schedule in CLI or run config, not both')
        resolved['coarse_schedule_policy'] = value
    generated = getattr(args, 'coarse_coverage_policy', None) in (*POLICY_ALIASES, 'model-derived')
    if 'coarse_schedule_policy' in resolved and not generated:
        raise ServeLoopError('coarse schedule policy requires generated analytic coverage')
    if generated:
        resolved.setdefault('coarse_schedule_policy', 'legacy-prefetch')
        validate_schedule_policy(resolved['coarse_schedule_policy'])
    return resolved


def resolve_serving_inputs(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, model_derived_coverage: bool = False
) -> ServingInputs:
    """Load the closed loop's inputs from parsed ``add_serving_arguments`` and ``--requests``."""

    missing = [
        name for name in ("model", "system", "placement")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("--requests requires " + ", ".join("--" + name for name in missing))
    if args.run_config is not None and any(
        getattr(args, name) is not None for name in TIMING_DEFAULTS
    ):
        parser.error("--run-config cannot be combined with timing/prefetch overrides")
    for name, value in TIMING_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.simulator is None:
        parser.error("--simulator is required for physical execution")
    system_paths = [args.system, *(args.overlay or [])]
    models = {}
    model_artifacts = []
    for path in args.model:
        model = load_model_any(path, include_coverage_descriptor=model_derived_coverage)
        if model.model_id in models:
            raise ServeLoopError(f"duplicate model id {model.model_id}")
        models[model.model_id] = model
        model_artifacts.append(_artifact(path))
    request_trace = load_requests_any(args.requests)
    router = None if args.router is None else load_router(args.router)
    system = ResolvedSystemConfig.load(system_paths).resolve(args.simulator)
    placement_path: Path | None = None
    if args.placement in PLACEMENT_PRESETS:
        placement_spec = derive_placement(
            system=system, models=models, preset=args.placement
        )
        placement_label = args.placement
    else:
        placement_path = Path(args.placement)
        placement_spec = placement_from_dict(
            load_json_object(placement_path, "serving placement")
        )
        placement_label = str(placement_path)
    if args.prefix_cache_bytes is not None or args.prefix_cache_ttl_ns is not None:
        placement_spec = replace(
            placement_spec,
            prefix_cache_bytes=(placement_spec.prefix_cache_bytes if args.prefix_cache_bytes is None else args.prefix_cache_bytes),
            prefix_cache_ttl_ns=(placement_spec.prefix_cache_ttl_ns if args.prefix_cache_ttl_ns is None else args.prefix_cache_ttl_ns),
        )
    if args.run_config is not None:
        run_config = load_json_object(args.run_config, "serving run config")
    else:
        run_config = default_run_config(
            timing=args.timing,
            peak_tflops=args.peak_tflops,
            efficiency=args.efficiency,
            prefetch_depth=args.prefetch_depth,
        )
    run_config = _resolve_coarse_run_config(run_config, args)
    policy, timing, prefetch_depth = run_config_from_dict(run_config)
    if timing.timing_model == "gpu_calibrated":
        models = {key: timing.profile.bind_model(model, timing.model_bindings[key])
                  for key, model in models.items()}
        if args.placement in PLACEMENT_PRESETS:
            placement_spec = replace(derive_placement(system=system,models=models,preset=args.placement),
                prefix_cache_bytes=placement_spec.prefix_cache_bytes,
                prefix_cache_ttl_ns=placement_spec.prefix_cache_ttl_ns,
                kv_block_tokens=timing.profile.document['kv_block_tokens'])
        elif placement_spec.kv_block_tokens != timing.profile.document['kv_block_tokens']:
            raise ServeLoopError("placement page size differs from the calibrated attention backend")
    input_artifacts = {
        "simulator": _artifact(args.simulator),
        "system_configs": [_artifact(path) for path in system_paths],
        "system_label": " + ".join(
            path.resolve().relative_to(ROOT).as_posix()
            if path.resolve().is_relative_to(ROOT)
            else str(path)
            for path in system_paths
        ),
        "models": model_artifacts,
        "requests": _artifact(args.requests),
        "request_trace": {
            "sha256": request_trace.digest,
            "requests": len(request_trace.requests),
            "prompt_tokens": sum(
                request.prompt_tokens for request in request_trace.requests
            ),
            "output_tokens": sum(
                request.output_tokens for request in request_trace.requests
            ),
        },
        "router": None if args.router is None else _artifact(args.router),
        "placement": (
            {"preset": args.placement, "sha256": placement_spec.digest}
            if placement_path is None
            else _artifact(placement_path)
        ),
        "run_config": (
            _artifact(args.run_config)
            if args.run_config is not None
            else {"default": True, "sha256": canonical_sha256(run_config)}
        ),
    }
    return ServingInputs(
        models=models, request_trace=request_trace, router=router, system_paths=tuple(system_paths),
        system=system, placement_spec=placement_spec, placement_label=placement_label,
        run_config=run_config, policy=policy, timing=timing, prefetch_depth=prefetch_depth,
        input_artifacts=input_artifacts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serveloop run",
        description=(
            "Run closed-loop requests (--requests) or a fixed memory window "
            "(--experiment) through HBFSim. Scale is selected by the input "
            "configs, not by a different generator."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--requests",
        type=Path,
        help="closed loop: serveloop.synthetic_requests or serveloop.request_trace JSON",
    )
    source.add_argument(
        "--experiment",
        type=Path,
        help="fixed window: explicit experiment JSON (miniquick or full scale); "
        "no compute, request admission feedback, or TTFT/TPOT",
    )
    serving = parser.add_argument_group("closed-loop requests only")
    add_serving_arguments(serving)
    serving.add_argument(
        "--prefix-image",
        type=Path,
        help="serveloop.prefix_image JSON from serveloop snapshot, installed in the prefix cache before the first request",
    )
    serving.add_argument(
        "--sample-rate", type=float,
        help="time only this share of the iterations in full (after each stratum's pilot) and "
             "estimate the rest from their strata; the others still run their writes",
    )
    serving.add_argument("--sample-pilot", type=int, help="timed iterations that open every stratum (default: 2)")
    serving.add_argument("--sample-warmup", type=int,
                         help="timed iterations after a skipped one that stay out of the estimate (default: 1)")
    serving.add_argument("--sample-seed", type=int, help="seed of the sampling draws (default: 0)")
    serving.add_argument('--coarse-coverage-profile',type=Path,
        help='opt-in digest-bound operator coverage profile; memory_only, covered shapes only')
    serving.add_argument('--coarse-coverage-policy',choices=('dense-16bit-swiglu','dense-bf16-swiglu','model-derived'),
        help='model-derived B1 tensor footprints with explicit fusion/attention assumptions')
    serving.add_argument('--coarse-schedule-policy', choices=('legacy-prefetch','consumer-order'),
        help='generated coverage weight release; default legacy-prefetch preserves main behavior')
    serving.add_argument('--coarse-cache-bound',choices=('off','ideal-temporaries'),
        help='temporary-traffic sensitivity bound with a coverage profile or policy; not a GPU cache simulator')
    serving.add_argument('--coarse-l2-write-policy',choices=('write-allocate-fetch','full-store-no-fetch'),
        help='default write-allocate-fetch; optional complete-sector stores skip fetch, partial stores still fetch')
    serving.add_argument('--coarse-l2-capacity-bytes',type=int,
        help='opt-in fully associative32B-sector range LRU; write-back/allocate, cold start, final drain')
    window = parser.add_argument_group("fixed-window experiments only")
    window.add_argument(
        "--topologies",
        help="comma-separated topology IDs to execute; default: all declared rows",
    )
    window.add_argument(
        "--preflight-only",
        action="store_true",
        help="generate and validate the entire window/matrix without a simulator",
    )
    add_allow_dirty_argument(window)
    parser.add_argument("--simulator", type=Path, help="path to a compatible external HBFSim executable")
    parser.add_argument("--out", type=Path, required=True, help="output root directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    serving_options = (
        *SERVING_ARGUMENTS,
        "prefix_image",
        "sample_rate",
        "sample_pilot",
        "sample_warmup",
        "sample_seed",
        "coarse_coverage_profile",
        "coarse_coverage_policy",
        "coarse_schedule_policy",
        "coarse_cache_bound",
        "coarse_l2_capacity_bytes",
        "coarse_l2_write_policy",
        *TIMING_DEFAULTS,
    )
    if args.experiment is not None:
        invalid = [name for name in serving_options if getattr(args, name) is not None]
        if invalid:
            parser.error(
                "--experiment cannot use closed-loop options: "
                + ", ".join("--" + name.replace("_", "-") for name in invalid)
            )
        if args.preflight_only and args.topologies is not None:
            parser.error(
                "--preflight-only validates all rows; --topologies is execution-only"
            )
        from serveloop.windows.run import run_window

        if not args.preflight_only and args.simulator is None:
            parser.error("--simulator is required for physical execution")
        return run_window(args, parser)
    if args.topologies is not None or args.preflight_only or args.allow_dirty:
        parser.error(
            "--topologies, --preflight-only, and --allow-dirty require --experiment"
        )
    if args.coarse_coverage_profile is not None and args.coarse_coverage_policy is not None:
        parser.error('choose --coarse-coverage-profile or --coarse-coverage-policy')
    if args.coarse_cache_bound is not None and args.coarse_coverage_profile is None and args.coarse_coverage_policy is None:
        parser.error('--coarse-cache-bound requires a coverage profile or policy')
    if args.coarse_l2_write_policy is not None and args.coarse_l2_capacity_bytes is None:
        parser.error('--coarse-l2-write-policy requires --coarse-l2-capacity-bytes')
    if args.coarse_l2_capacity_bytes is not None:
        if args.coarse_coverage_profile is None and args.coarse_coverage_policy is None:
            parser.error('--coarse-l2-capacity-bytes requires a coverage profile or policy')
        if args.coarse_cache_bound not in (None,'off'):
            parser.error('coarse L2 cannot combine with ideal-temporaries')
        if args.coarse_l2_capacity_bytes<32 or args.coarse_l2_capacity_bytes%32:
            parser.error('coarse L2 capacity must be a positive multiple of32 bytes')
    try:
        inputs = resolve_serving_inputs(
            args, parser, model_derived_coverage=args.coarse_coverage_policy == 'model-derived')
        models, request_trace, router = inputs.models, inputs.request_trace, inputs.router
        system_paths, placement_spec = list(inputs.system_paths), inputs.placement_spec
        placement_label, run_config = inputs.placement_label, inputs.run_config
        policy, timing, prefetch_depth = inputs.policy, inputs.timing, inputs.prefetch_depth
        input_artifacts = inputs.input_artifacts
        prefix_image = None
        if args.prefix_image is not None:
            prefix_image = load_prefix_image(args.prefix_image)
            input_artifacts["prefix_image"] = _artifact(args.prefix_image)
        sampling = None
        if args.sample_rate is None:
            if any(getattr(args, name) is not None for name in ("sample_pilot", "sample_warmup", "sample_seed")):
                parser.error("--sample-pilot, --sample-warmup and --sample-seed need --sample-rate")
        else:
            try:
                sampling = SamplingPlan(
                    rate=args.sample_rate,
                    pilot=2 if args.sample_pilot is None else args.sample_pilot,
                    warmup=1 if args.sample_warmup is None else args.sample_warmup,
                    seed=0 if args.sample_seed is None else args.sample_seed)
            except ValueError as error:
                parser.error(str(error))
            input_artifacts["sampling"] = sampling.canonical()
        output_directory = create_run_directory(args.out)
        if args.coarse_coverage_profile is not None:
            input_artifacts['coarse_coverage'] = dict(**_artifact(args.coarse_coverage_profile),
                cache_bound=args.coarse_cache_bound or 'off',
                traffic_domain='tensor/kernel coverage; not hardware or modeled GPU-cache misses')
        if args.coarse_coverage_policy is not None:
            if args.coarse_coverage_policy == 'model-derived':
                from serveloop.coarse_catalog import ASSUMPTIONS
            else:
                from serveloop.coarse_analytic import ASSUMPTIONS
            input_artifacts['coarse_coverage'] = dict(policy=args.coarse_coverage_policy,
                assumptions=ASSUMPTIONS,cache_bound=args.coarse_cache_bound or 'off',
                traffic_domain='analytic tensor footprints; not GPU-cache misses or captured kernel coverage')
            if timing.timing_model == 'roofline':
                input_artifacts['coarse_coverage']['compute'] = dict(
                    model='layer-aggregate roofline', peak_tflops=timing.peak_tflops,
                    efficiency=timing.efficiency, calibrated=False,
                    overlap='next-layer weight/KV prefetch; activations precede aggregate compute')
            elif timing.timing_model == 'linear':
                input_artifacts['coarse_coverage']['compute'] = dict(
                    model='layer-aggregate linear sensitivity',provider=timing.canonical(),
                    calibrated=False,overlap='next-layer weight/KV prefetch; activations precede aggregate compute')
        if args.coarse_l2_capacity_bytes is not None:
            input_artifacts['coarse_l2']=dict(capacity_bytes=args.coarse_l2_capacity_bytes,
                sector_bytes=32,model='fully_associative_sector_lru_range',
                initial='cold',write_policy=args.coarse_l2_write_policy or 'write-allocate-fetch',final='dirty drain',
                traffic_domain='modeled post-L2 demand; not measured hardware traffic')
        if args.coarse_coverage_policy is not None:
            input_artifacts['coarse_coverage']['schedule_policy'] = run_config['coarse_schedule_policy']
            if run_config['coarse_schedule_policy'] == 'consumer-order' and 'compute' in input_artifacts['coarse_coverage']:
                input_artifacts['coarse_coverage']['compute']['overlap'] = 'consumer-order aggregate weight release; compute budgets unchanged'
        for model in models.values():
            write_json_atomic(
                output_directory / f"model-{model.model_id}.json", model.canonical()
            )
        write_json_atomic(output_directory / "requests.json", request_trace.canonical())
        write_json_atomic(output_directory / "placement.json", placement_spec.canonical())
        write_json_atomic(output_directory / "run-config.json", run_config)
        experiment = run_experiment(
            simulator=args.simulator,
            system_config_paths=system_paths,
            models=models,
            request_trace=request_trace,
            router=router,
            placement_spec=placement_spec,
            policy=policy,
            timing=timing,
            prefetch_depth=prefetch_depth,
            input_artifacts=input_artifacts,
            coarse_coverage_profile=args.coarse_coverage_profile,
            coarse_coverage_policy=args.coarse_coverage_policy,
            coarse_schedule_policy=run_config.get('coarse_schedule_policy'),
            coarse_cache_bound=args.coarse_cache_bound or 'off',
            coarse_l2_capacity_bytes=args.coarse_l2_capacity_bytes,
            coarse_l2_write_policy=args.coarse_l2_write_policy or 'write-allocate-fetch',
            prefix_image=prefix_image,
            sampling=sampling,
        )
        result_path = output_directory / "result.json"
        write_json_atomic(result_path, experiment)
        lines = headline_lines(
            experiment, result_path=result_path, placement_label=placement_label
        )
        if experiment["sampling"] is not None:
            report = experiment["sampling"]
            interval = report["confidence_interval_95_ns"]
            unknown = report["unknown_variance_batches"]
            lines.append(
                f"sampled: timed {report['timed_batches']} of {report['batches']} iterations; "
                f"window estimate {_format_ns(report['estimate_ns'])} "
                f"(95% interval {_format_ns(interval[0])} to {_format_ns(interval[1])}"
                + (f", {unknown} skipped iterations in strata with one observation are outside it" if unknown else "")
                + "); latencies above mix timed and estimated iterations")
        if args.coarse_coverage_policy is not None:
            lines.append('analytic coverage: assumed tensor footprints; attention scratch/compute unmodeled')
        if args.coarse_coverage_profile is not None:
            lines.append('coarse coverage: tensor/kernel footprints' if args.coarse_l2_capacity_bytes is not None else
                         'coarse coverage: tensor/kernel footprints; GPU cache misses are not modeled')
        if args.coarse_cache_bound == 'ideal-temporaries':
            lines.append('sensitivity bound: temporary traffic ideally on-chip; weight/KV traffic unchanged')
        if args.coarse_l2_capacity_bytes is not None:
            lines.append('coarse L2: fully associative range LRU; modeled post-cache traffic, not hardware validated')
        elif args.coarse_coverage_policy is not None:
            lines.append('GPU cache misses are not modeled')
        (output_directory / "headline.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    except (OSError, ServeLoopError, SimulationSessionError) as error:
        print(f"serveloop run failed: {error}", file=sys.stderr)
        if isinstance(error, SimulationSessionError):
            print(describe_backend(args.simulator), file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0

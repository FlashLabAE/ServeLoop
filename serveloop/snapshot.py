#!/usr/bin/env python3
"""Decode snapshots: start a timed run from the in-flight state of a long replay.

A snapshot is the closed loop's state at one decode iteration of a logical
replay (``serveloop.replay``) over a request stream: which requests are
being served, how much context each holds and where that context lives in
the KV pool, and which prompt blocks the prefix cache retains besides. A
timed run resumes from it with the prefix cache installed from the snapshot
(``serveloop run --prefix-image``) and one request per in-flight sequence
whose prompt is the sequence's context plus a short private suffix: the
context hits the cache, the suffix is prefilled in one iteration, and the
requested output tokens decode from the snapshot's state. The timed run
spends its time on the decode iterations of interest, not on the history
that produced them.

The iteration is chosen among those at which every batch slot is busy,
requests still wait, and the prefix cache (when there is one) has reached
its budget: the one whose running sequences hold the median total context.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from hbfsim_client.simulation_session import SimulationSessionError
from serveloop.compiler import ServeLoopCompiler
from serveloop.contracts import (
    ModelSpec,
    RequestSpec,
    RequestTrace,
    SchedulerPolicy,
    ServeLoopError,
    TimingProvider,
    TraceProvenance,
    canonical_sha256,
)
from serveloop.engine import ServeLoopEngine
from serveloop.io import write_json_atomic
from serveloop.placement import PlacementSpec, ServeLoopPlacement
from serveloop.prefix_image import ImageEntry, ImagePrompt, PrefixImage, write_prefix_image
from serveloop.replay import LogicalReplay, ReplayExecutor


SNAPSHOT_SCHEMA = {"name": "serveloop.decode_snapshot", "version": 1}


@dataclass(frozen=True)
class SnapshotSettings:
    """How the replay is clocked and how the resumed requests continue."""

    iteration_ns: float
    suffix_tokens: int
    output_tokens: int
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.iteration_ns, bool) or not isinstance(self.iteration_ns, (int, float)) \
                or not math.isfinite(self.iteration_ns) or self.iteration_ns <= 0:
            raise ServeLoopError("the replay iteration time must be positive and finite")
        for name, value in (("suffix_tokens", self.suffix_tokens), ("output_tokens", self.output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ServeLoopError(f"snapshot {name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ServeLoopError("snapshot seed must be an integer")

    def canonical(self) -> dict[str, Any]:
        return {"iteration_ns": float(self.iteration_ns), "suffix_tokens": self.suffix_tokens,
                "output_tokens": self.output_tokens, "seed": self.seed}


@dataclass(frozen=True)
class Snapshot:
    requests: RequestTrace
    image: PrefixImage
    metadata: dict[str, Any]


def _replay(
    *,
    models: Mapping[str, ModelSpec],
    request_trace: RequestTrace,
    router: Any,
    placement_spec: PlacementSpec,
    hbf_geometry: Any,
    hbm_stripe_bytes: int | None,
    policy: SchedulerPolicy,
    timing: TimingProvider,
    prefetch_depth: int,
    iteration_ns: float,
) -> tuple[LogicalReplay, ServeLoopPlacement]:
    placement = ServeLoopPlacement(models=models, spec=placement_spec, hbf_geometry=hbf_geometry,
                                   hbm_stripe_bytes=hbm_stripe_bytes, router=router)
    executor = ReplayExecutor(placement, iteration_ns)
    compiler = ServeLoopCompiler(models=models, request_trace=request_trace, router=router,
                                 timing=timing, prefetch_depth=prefetch_depth)
    engine = ServeLoopEngine(models=models, request_trace=request_trace, compiler=compiler,
                             executor=executor, policy=policy)
    return LogicalReplay(engine, executor), placement


def representative_iteration(replay: LogicalReplay, placement: ServeLoopPlacement,
                             policy: SchedulerPolicy) -> tuple[int, dict[str, Any]]:
    """Replay to the end and pick the median-context iteration among the steady-state ones."""
    budget = placement.spec.prefix_cache_bytes
    filled = budget == 0
    candidates: list[tuple[int, int]] = []
    while not replay.finished:
        schedule, count = replay.advance()
        if filled and len(schedule.slices) == policy.max_batch_requests \
                and replay.engine.unadmitted_requests > 0:
            context = sum(item.token_end for item in schedule.slices)
            candidates.extend((context + i * len(schedule.slices), schedule.batch_id + i) for i in range(count))
        if not filled:
            filled = placement.prefix_receipt()["resident_bytes"] + placement.kv_block_bytes > budget
    if not candidates:
        raise ServeLoopError(
            "no iteration keeps every batch slot busy with requests waiting"
            + (" after the prefix cache fills" if budget else "")
            + "; offer more requests or a smaller batch")
    ranked = sorted(candidates)
    context, iteration = ranked[(len(ranked) - 1) // 2]
    return iteration, {
        "candidates": len(candidates), "replayed_iterations": replay.iterations,
        "context_tokens": context,
        "context_rank": sum(value < context for value, _ in candidates) / len(candidates),
    }


def _surrogate_tokens(seed: int, purpose: str, count: int, vocab_size: int) -> tuple[int, ...]:
    generator = random.Random(f"serveloop.snapshot/{seed}/{purpose}")
    return tuple(generator.randrange(vocab_size) for _ in range(count))


def take_snapshot(
    *,
    models: Mapping[str, ModelSpec],
    request_trace: RequestTrace,
    router: Any,
    placement_spec: PlacementSpec,
    hbf_geometry: Any,
    hbm_stripe_bytes: int | None,
    policy: SchedulerPolicy,
    timing: TimingProvider,
    prefetch_depth: int,
    settings: SnapshotSettings,
) -> Snapshot:
    """Replay the stream, choose the representative iteration and export its state."""
    if any(request.token_ids is None for request in request_trace.requests):
        raise ServeLoopError("a snapshot needs token ids: the resumed requests reuse prompt blocks by content")
    shared = dict(models=models, request_trace=request_trace, router=router, placement_spec=placement_spec,
                  hbf_geometry=hbf_geometry, hbm_stripe_bytes=hbm_stripe_bytes, policy=policy,
                  timing=timing, prefetch_depth=prefetch_depth, iteration_ns=settings.iteration_ns)
    survey, placement = _replay(**shared)
    iteration, selection = representative_iteration(survey, placement, policy)
    replay, placement = _replay(**shared)
    replay.run_until(iteration + 1)
    frontier = replay.executor.frontier_ns
    block_tokens = placement.spec.kv_block_tokens
    requests = {request.request_id: request for request in request_trace.requests}

    # Cache entries keep the order they had; the ones the resumed requests
    # read are listed afterwards, since their lookups would move them last.
    retained = {key: (blocks, expires) for key, blocks, expires in placement.retained_prefix()}
    prompts: list[ImagePrompt] = []
    prompt_index: dict[str, int] = {}
    entries: list[ImageEntry] = []
    placed: set[str] = set()

    def prompt_slot(request_id: str, model_id: str, cache_salt: str, tokens: tuple[int, ...]) -> int:
        if request_id not in prompt_index:
            prompt_index[request_id] = len(prompts)
            prompts.append(ImagePrompt(request_id=request_id, model_id=model_id, cache_salt=cache_salt, token_ids=tokens))
        return prompt_index[request_id]

    def relative(expires_ns: float | None) -> float | None:
        return None if expires_ns is None else expires_ns - frontier

    resumed: list[RequestSpec] = []
    contexts: list[dict[str, Any]] = []
    context_keys: list[tuple[str, int, tuple[int, ...], float | None, int]] = []
    for ordinal, (request_id, processed, emitted) in enumerate(replay.engine.running()):
        request = requests[request_id]
        model = models[request.model_id]
        blocks = processed // block_tokens
        context = blocks * block_tokens
        row = {"request_id": request_id, "processed_tokens": processed, "emitted_tokens": emitted,
               "context_tokens": context}
        if blocks == 0:
            contexts.append({**row, "resumed": None})
            continue
        tier, table = placement.request_blocks(request_id)
        if tier != "hbm":
            raise ServeLoopError(f"request {request_id} keeps its KV in {tier}; a snapshot resumes HBM-resident context")
        continuation = _surrogate_tokens(settings.seed, request_id, settings.suffix_tokens + settings.output_tokens - 1,
                                         model.vocab_size)
        resume = RequestSpec(
            request_id=f"{request_id}.resume", arrival_ns=0.0, model_id=request.model_id,
            prompt_tokens=context + settings.suffix_tokens, output_tokens=settings.output_tokens,
            token_ids=request.token_ids[:context] + continuation, cache_salt=request.cache_salt)
        resumed.append(resume)
        contexts.append({**row, "resumed": resume.request_id})
        keys = placement.prefix_keys_for(resume)[:blocks]
        slot = prompt_slot(f"{request_id}.context", request.model_id, request.cache_salt, request.token_ids[:context])
        for block, key in enumerate(keys):
            if key in placed:
                continue
            placed.add(key)
            layers = tuple(row_blocks[block] for row_blocks in table)
            cached = retained.get(key)
            if cached is not None and cached[0] != layers:
                raise ServeLoopError("the prefix cache and the block table disagree about a context block")
            context_keys.append((key, slot, layers, relative(cached[1]) if cached else None, block))
    if not resumed:
        raise ServeLoopError("no running sequence holds a complete KV block at the chosen iteration")

    origin: dict[str, tuple[RequestSpec, int]] = {}
    for request in request_trace.requests:
        for block, key in enumerate(placement.prefix_keys_for(request)):
            origin.setdefault(key, (request, block))
    for key, (blocks, expires) in retained.items():
        if key in placed:
            continue
        request, block = origin[key]
        slot = prompt_slot(request.request_id, request.model_id, request.cache_salt,
                           request.token_ids[:request.prompt_tokens])
        entries.append(ImageEntry(prompt=slot, block=block, blocks=blocks, expires_ns=relative(expires)))
    for key, slot, layers, expires, block in context_keys:
        entries.append(ImageEntry(prompt=slot, block=block, blocks=layers, expires_ns=expires))

    source = {
        "request_trace_sha256": request_trace.digest,
        "placement_sha256": placement_spec.digest,
        "scheduler": policy.canonical(),
        "timing": timing.canonical(),
        "settings": settings.canonical(),
        "iteration": iteration,
        "frontier_ns": frontier,
    }
    image = PrefixImage(source=source, kv_block_tokens=block_tokens, prompts=tuple(prompts), entries=tuple(entries))
    trace = RequestTrace(
        provenance=TraceProvenance(
            kind="synthetic_sensitivity",
            source=f"decode snapshot of request trace {request_trace.digest} at iteration {iteration}: "
                   f"{len(resumed)} in-flight sequences resume their context with "
                   f"{settings.suffix_tokens} private suffix tokens and {settings.output_tokens} output tokens"),
        requests=tuple(sorted(resumed, key=lambda request: (request.arrival_ns, request.request_id))))
    metadata = {
        "schema": SNAPSHOT_SCHEMA,
        "source": source,
        "selection": {"rule": "median_context_steady_state_v1", "iteration": iteration, **selection},
        "sequences": contexts,
        "prefix_cache": placement.prefix_receipt(),
        "image": {"sha256": image.digest, "prompts": len(prompts), "entries": len(entries),
                  "blocks": image.blocks, "bytes": image.blocks * placement.kv_block_bytes,
                  "minimum_prefix_cache_bytes": image.blocks * placement.kv_block_bytes},
        "requests_sha256": trace.digest,
    }
    metadata["snapshot_sha256"] = canonical_sha256(metadata)
    return Snapshot(requests=trace, image=image, metadata=metadata)


def write_snapshot(directory: Path, snapshot: Snapshot) -> dict[str, Path]:
    paths = {"snapshot": directory / "snapshot.json", "requests": directory / "requests.json",
             "prefix_image": directory / "prefix-image.json"}
    write_json_atomic(paths["snapshot"], snapshot.metadata)
    write_json_atomic(paths["requests"], snapshot.requests.canonical())
    write_prefix_image(paths["prefix_image"], snapshot.image)
    return paths


def build_parser() -> argparse.ArgumentParser:
    from serveloop.run import add_serving_arguments

    parser = argparse.ArgumentParser(
        prog="serveloop snapshot",
        description="Replay a request stream on a declared clock and export the state of a "
                    "representative decode iteration, for serveloop run --prefix-image.")
    add_serving_arguments(parser.add_argument_group("serving inputs, as for serveloop run"))
    parser.add_argument("--requests", type=Path, required=True,
                        help="serveloop.synthetic_requests or serveloop.request_trace JSON")
    parser.add_argument("--iteration-ns", type=float, default=50e6,
                        help="declared duration of every replay iteration (default: 50 ms)")
    parser.add_argument("--suffix-tokens", type=int, default=16,
                        help="private prompt tokens each resumed sequence prefills (default: 16)")
    parser.add_argument("--output-tokens", type=int, default=8,
                        help="output tokens each resumed sequence decodes (default: 8)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the surrogate suffix and output tokens")
    parser.add_argument("--simulator", type=Path, required=True,
                        help="HBFSim executable, used to resolve the system's geometry")
    parser.add_argument("--out", type=Path, required=True, help="directory for snapshot.json, requests.json and prefix-image.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from serveloop.run import hbm_stripe_bytes, resolve_serving_inputs

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        inputs = resolve_serving_inputs(args, parser)
        settings = SnapshotSettings(iteration_ns=args.iteration_ns, suffix_tokens=args.suffix_tokens,
                                    output_tokens=args.output_tokens, seed=args.seed)
        snapshot = take_snapshot(
            models=inputs.models, request_trace=inputs.request_trace, router=inputs.router,
            placement_spec=inputs.placement_spec, hbf_geometry=inputs.system.hbf_geometry,
            hbm_stripe_bytes=hbm_stripe_bytes(inputs.system), policy=inputs.policy, timing=inputs.timing,
            prefetch_depth=inputs.prefetch_depth, settings=settings)
        snapshot.metadata["inputs"] = inputs.input_artifacts
        paths = write_snapshot(args.out, snapshot)
    except (OSError, ServeLoopError, SimulationSessionError) as error:
        print(f"serveloop snapshot failed: {error}", file=sys.stderr)
        return 2
    selection = snapshot.metadata["selection"]
    image = snapshot.metadata["image"]
    sequences = [row for row in snapshot.metadata["sequences"] if row["resumed"] is not None]
    contexts = sorted(row["context_tokens"] for row in sequences)
    print(f"iteration {selection['iteration']} of {selection['replayed_iterations']} "
          f"({selection['candidates']} steady-state candidates); {len(sequences)} sequences resume "
          f"{contexts[0]}..{contexts[-1]} context tokens (median {contexts[len(contexts) // 2]})")
    print(f"prefix image: {image['entries']} entries, {image['blocks']} blocks, {image['bytes']} bytes; "
          f"run with --prefix-cache-bytes >= {image['minimum_prefix_cache_bytes']}")
    print(f"resume with: serveloop run --requests {paths['requests']} --prefix-image {paths['prefix_image']} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Closed-loop adapter from ServeLoop batches to HBFSim simulation sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import hbfsim_client
from hbfsim_client.sampling import Sampler, state_only
from hbfsim_client.simulation_session import (
    SimulationSession,
    SimulationSessionError,
    ResolvedSystemConfig,
)
from hbfsim_client.transaction_protocol import TransactionBatch
from serveloop.contracts import (
    BatchSlice,
    CanonicalServingBatch,
    RequestSpec,
    ServeLoopError,
)
from serveloop.engine import BatchExecution
from serveloop.placement import ServeLoopPlacement

# HBFSim commit that hbfsim_client/ and configs/systems/ were copied from;
# configs/ocp-profile-source.json records the same commit.
BUNDLED_HBFSIM_COMMIT = "514e796425c37eca4047c1ae3a4e6b9fff174fe0"


def describe_backend(simulator_path: Path | None) -> str:
    """Name the client and engine that met, for a failed session's message."""

    client = Path(hbfsim_client.__file__).resolve().parent
    bundled = client.parent == Path(__file__).resolve().parents[1]
    lines = [
        f"  hbfsim_client: {client}"
        + (f" (bundled copy of HBFSim {BUNDLED_HBFSIM_COMMIT[:7]})" if bundled else "")
    ]
    if simulator_path is not None:
        try:
            completed = subprocess.run(
                [str(simulator_path), "--version"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            output = completed.stdout.strip()
            version = (
                output.splitlines()[0]
                if completed.returncode == 0 and output
                else f"--version exited with {completed.returncode}"
            )
        except (OSError, subprocess.SubprocessError) as error:
            version = f"not runnable ({error})"
        lines.append(f"  simulator: {simulator_path}: {version}")
    lines.append(
        "  Receipt and configuration-key errors usually mean that the client, the "
        "system profiles and the simulator come from different HBFSim revisions; "
        "see docs/configuration.md#backend-compatibility."
    )
    return "\n".join(lines)


def batch_stratum(batch: CanonicalServingBatch) -> tuple[str, int, int, int]:
    """Iterations alike in what they read: kind, decoding sequences, and prefill and context tokens by power of two."""
    slices = batch.schedule.slices
    decoding = sum(item.phase == "decode" for item in slices)
    prefill = sum(item.token_count for item in slices if item.phase == "prefill")
    context = sum(item.context_tokens_before for item in slices)
    return batch.schedule.kind, decoding, prefill.bit_length(), context.bit_length()


class HbfSimExecutor:
    """One physical HBM/HBF/external memory system in a persistent session.

    With a ``sampler`` (``hbfsim_client.sampling``) only the sampled
    iterations are timed in full; the others keep their writes and copies
    and take their stratum's running mean as their duration, so the
    completion frontier that schedules the next iteration mixes timed and
    estimated durations.
    """

    def __init__(
        self,
        *,
        simulator_path: Path,
        system_config_paths: Sequence[Path],
        placement: ServeLoopPlacement,
        initial_hbf_persistent_image: Path | None = None,
        preinstall_hbf_weights: bool = True,
        enable_external_recovery: bool = False,
        sampler: Sampler | None = None,
        on_batch_completed: Callable[
            [CanonicalServingBatch, TransactionBatch, Mapping[str, Any]], None
        ] | None = None,
    ) -> None:
        self.placement = placement
        self.sampler = sampler
        self._on_batch_completed = on_batch_completed
        external_enabled = placement.enable_external or enable_external_recovery
        try:
            self.system_config = ResolvedSystemConfig.load(system_config_paths).resolve(
                simulator_path, enable_hbf=placement.enable_hbf
            )
        except SimulationSessionError as error:
            raise ServeLoopError(str(error)) from error
        geometry = self.system_config.hbf_geometry
        if placement.spec.hbf_page_size_bytes != geometry.page_size_bytes:
            raise ServeLoopError(
                "serving placement HBF page size differs from the physical config"
            )
        if (
            placement.hbf_geometry is not None
            and placement.hbf_geometry != geometry
        ):
            raise ServeLoopError(
                "serving placement HBF geometry differs from the physical config"
            )
        if placement.enable_hbf and placement.spec.hbf_capacity_bytes > self.system_config.logical_hbf_capacity_bytes:
            raise ServeLoopError(
                "serving placement HBF capacity exceeds the engine's logical capacity"
            )
        if (
            external_enabled
            and placement.spec.external_page_size_bytes
            != int(self.system_config.external_backing_identity["page_size_bytes"])
        ):
            raise ServeLoopError(
                "serving placement external page size differs from the config"
            )
        if (
            external_enabled
            and placement.spec.external_capacity_bytes
            > int(self.system_config.external_backing_identity["capacity_bytes"])
        ):
            raise ServeLoopError(
                "serving placement external capacity exceeds the physical config"
            )
        try:
            self._session = SimulationSession(
                simulator_path=simulator_path,
                system_config=self.system_config,
                enable_hbm=True,
                enable_hbf=placement.enable_hbf,
                enable_external=external_enabled,
                hbm_capacity_bytes=placement.spec.hbm_capacity_bytes,
                initial_hbf_logical_first_lpn=0,
                initial_hbf_logical_pages=(
                    placement.initial_hbf_logical_pages
                    if preinstall_hbf_weights and initial_hbf_persistent_image is None
                    else 0
                ),
                initial_hbf_persistent_image=initial_hbf_persistent_image,
            )
        except SimulationSessionError as error:
            raise ServeLoopError(str(error)) from error
        self._closed = False

    @property
    def frontier_ns(self) -> float:
        return self._session.completed_frontier_ns

    @property
    def execution_identity(self) -> Mapping[str, Any]:
        source_setup = self._session.source_receipt()
        source_setup.pop("final_measurement", None)
        return {
            "kind": "serveloop_hbfsim_simulation_session",
            "physical_memory_timing": True,
            "absolute_memory_timing_claim_eligible": False,
            "ttft_tpot_claim_eligible": False,
            "single_device": True,
            "network_timing": False,
            "claim_note": (
                "HBFSim supplies modeled physical memory timing; compute is "
                "the explicitly selected run timing model, including calibrated GPU operators. "
                "Physical HBF accuracy and production server latency are not implied by GPU calibration."
            ),
            "placement": deepcopy(
                self.placement.frontier_ns_independent_state
            ),
            "source_setup": source_setup,
        }

    def admit_request(self, request: RequestSpec) -> int:
        return self.placement.admit_request(request, now_ns=max(self.frontier_ns, request.arrival_ns))

    def can_reserve(self, slices: Sequence[BatchSlice]) -> bool:
        return self.placement.can_reserve(slices)

    def reserve(
        self, batch_id: int, slices: Sequence[BatchSlice]
    ) -> Mapping[str, Any]:
        return self.placement.reserve(batch_id, slices)

    def preempt_request(self, request_id: str) -> Mapping[str, Any]:
        return self.placement.preempt_request(request_id)

    def release_request(self, request_id: str) -> None:
        self.placement.release_request(request_id)

    def execute(self, batch: CanonicalServingBatch) -> BatchExecution:
        if self._closed:
            raise ServeLoopError("serving executor is closed")
        origin = self._session.completed_frontier_ns
        try:
            mapped = self.placement.map_batch(
                batch,
                session_frontier_ns=origin,
            )
            # The serving batch ends with its completion barrier; that
            # barrier is the blocking frontier the next batch waits for.
            # Everything else still executes and stays dependable.  Only
            # the aggregate receipt is kept: per-transaction completion
            # rows are never read here and would dominate the result.
            final = mapped.transactions[-1]
            if final.target == "BARRIER":
                mapped = replace(mapped, frontier=(final.id,), completions=False)
            else:
                mapped = replace(mapped, completions=False)
            decision = None if self.sampler is None else self.sampler.decide(batch_stratum(batch))
            if decision is not None and not decision.timed:
                mapped = state_only(mapped, estimate_ns=decision.estimate_ns,
                                    frontier_id=f"sample/b{batch.schedule.batch_id}")
            completion = self._session.submit(mapped)
        except (SimulationSessionError, ServeLoopError) as error:
            raise ServeLoopError(str(error)) from error
        if completion.get("logical_trace_sha256") != batch.digest:
            raise ServeLoopError(
                "HBFSim completion changed the canonical serving digest"
            )
        finish = float(completion["blocking_finish_ns"])
        if decision is not None:
            if decision.timed:
                self.sampler.observe(decision, finish - origin)
            completion = {**completion, "sampling": {**decision.canonical(), "elapsed_ns": finish - origin}}
        self.placement.complete_batch(batch.schedule, finish)
        mapped = replace(mapped, receipt={
            **mapped.receipt,
            "prefix_cache_after_completion": self.placement.prefix_receipt(),
        })
        if self._on_batch_completed is not None:
            self._on_batch_completed(batch, mapped, completion)
        return BatchExecution(
            batch_id=batch.schedule.batch_id,
            batch_origin_ns=float(completion["batch_origin_ns"]),
            finish_ns=float(completion["blocking_finish_ns"]),
            canonical_sha256=batch.digest,
            mapped_sha256=mapped.transaction_trace_sha256,
            remap_receipt=mapped.receipt,
            physical_completion=completion,
        )

    def submit_transactions(self, batch: TransactionBatch) -> dict[str, Any]:
        """Execute explicitly accounted lifecycle transfers on this device."""

        try:
            return self._session.submit(batch)
        except SimulationSessionError as error:
            raise ServeLoopError(str(error)) from error

    def checkpoint_image(
        self, checkpoint_id: str, image_output: Path
    ) -> dict[str, Any]:
        """Drain dirty data/mappings and export the native quiescent image."""

        try:
            return self._session.checkpoint_image(checkpoint_id, image_output)
        except SimulationSessionError as error:
            raise ServeLoopError(str(error)) from error

    def checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        try:
            return self._session.checkpoint(checkpoint_id)
        except SimulationSessionError as error:
            raise ServeLoopError(str(error)) from error

    def crash(self, crash_id: str) -> dict[str, Any]:
        """Inject a protocol-command-boundary failure without a final drain."""

        try:
            return self._session.crash(crash_id)
        except SimulationSessionError as error:
            raise ServeLoopError(str(error)) from error
        finally:
            self._closed = True

    def source_receipt(self) -> dict[str, Any]:
        return self._session.source_receipt()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._session.close()
        except SimulationSessionError as error:
            raise ServeLoopError(str(error)) from error
        self._closed = True

    def final_receipt(self) -> dict[str, Any]:
        if not self._closed:
            raise ServeLoopError(
                "serving executor must close before final receipt publication"
            )
        placement = self.placement.receipt()
        required_invariants = (
            "all_request_kv_released",
            "kv_hot_allocations_accounted_for",
            "kv_cold_pool_fully_free",
            "no_pending_migrations",
        )
        if not all(placement["final_invariants"][name] for name in required_invariants):
            raise ServeLoopError(
                "serving placement retained request KV state at publication"
            )
        return {
            "session": self._session.source_receipt(),
            "placement": placement,
        }

    def __enter__(self) -> "HbfSimExecutor":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()

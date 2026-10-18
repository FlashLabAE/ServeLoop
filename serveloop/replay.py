#!/usr/bin/env python3
"""Replay the scheduler on a declared clock, without compiling or executing batches.

A logical replay drives the closed loop's own scheduler and placement over a
request stream, but every iteration lasts a declared time instead of a
physical completion. Admission, prefix reuse, paged-KV allocation, migration
and preemption are therefore exactly what the timed loop would decide under
that clock, for the cost of the bookkeeping alone: no batch is compiled, no
transaction is mapped, nothing is submitted to an engine.

A decode-only stretch of iterations repeats one schedule with every request
one token further along. ``ServeLoopEngine.decode_horizon`` says how long
the scheduler would keep repeating it, and ``advance_decode`` applies those
iterations at once; the placement grows the block tables exactly as the
individual iterations would (``ServeLoopPlacement.advance_decode``). A long
output therefore costs a few events per request, not one scheduler pass per
generated token. Snapshots (``serveloop.snapshot``) are taken from such
replays.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

from serveloop.contracts import (
    BatchSlice,
    CanonicalServingBatch,
    RequestSpec,
    ScheduledBatch,
    ServeLoopError,
)
from serveloop.engine import BatchExecution, ServeLoopEngine
from serveloop.placement import ServeLoopPlacement


class ReplayExecutor:
    """Placement-backed executor whose every iteration takes ``iteration_ns``."""

    def __init__(self, placement: ServeLoopPlacement, iteration_ns: float) -> None:
        if isinstance(iteration_ns, bool) or not isinstance(iteration_ns, (int, float)) \
                or not math.isfinite(iteration_ns) or iteration_ns <= 0:
            raise ServeLoopError("a replay iteration must take a positive, finite time")
        self.placement = placement
        self.iteration_ns = float(iteration_ns)
        self._frontier = 0.0

    @property
    def frontier_ns(self) -> float:
        return self._frontier

    @property
    def execution_identity(self) -> Mapping[str, Any]:
        return {
            "kind": "serveloop_logical_replay",
            "physical_memory_timing": False,
            "iteration_ns": self.iteration_ns,
            "placement": self.placement.frontier_ns_independent_state,
        }

    def admit_request(self, request: RequestSpec) -> int:
        return self.placement.admit_request(request, now_ns=max(self._frontier, request.arrival_ns))

    def can_reserve(self, slices: Sequence[BatchSlice]) -> bool:
        return self.placement.can_reserve(slices)

    def reserve(self, batch_id: int, slices: Sequence[BatchSlice]) -> Mapping[str, Any]:
        return self.placement.reserve(batch_id, slices)

    def preempt_request(self, request_id: str) -> Mapping[str, Any]:
        return self.placement.preempt_request(request_id)

    def release_request(self, request_id: str) -> None:
        self.placement.release_request(request_id)

    def execute(self, batch: CanonicalServingBatch) -> BatchExecution:
        raise ServeLoopError("a logical replay completes iterations without executing them")

    def complete_iteration(self, schedule: ScheduledBatch) -> float:
        """Finish ``schedule`` one iteration after its release and return the finish time."""
        finish = max(self._frontier, schedule.not_before_ns) + self.iteration_ns
        self.placement.discard_unmapped(schedule.batch_id)
        self.placement.complete_batch(schedule, finish)
        self._frontier = finish
        return finish

    def advance_decode(self, schedule: ScheduledBatch, iterations: int) -> int:
        """Reserve the KV of up to ``iterations`` repeats of ``schedule`` and advance the clock."""
        applied = self.placement.advance_decode(
            schedule.slices, iterations, batch_id=schedule.batch_id + 1,
            first_finish_ns=self._frontier + self.iteration_ns, iteration_ns=self.iteration_ns)
        self._frontier += applied * self.iteration_ns
        return applied


def shifted(schedule: ScheduledBatch, iterations: int, *, not_before_ns: float) -> ScheduledBatch:
    """The decode iteration ``iterations`` after ``schedule``: every slice that many tokens later."""
    return ScheduledBatch(
        batch_id=schedule.batch_id + iterations,
        model_id=schedule.model_id,
        slices=tuple(
            replace(item, token_begin=item.token_begin + iterations,
                    context_tokens_before=item.context_tokens_before + iterations)
            for item in schedule.slices),
        not_before_ns=not_before_ns)


class LogicalReplay:
    """Steps an engine over a ``ReplayExecutor``, jumping decode-only stretches."""

    def __init__(self, engine: ServeLoopEngine, executor: ReplayExecutor) -> None:
        if engine.executor is not executor:
            raise ServeLoopError("the replay executor must be the engine's executor")
        self.engine = engine
        self.executor = executor
        self._last: ScheduledBatch | None = None

    @property
    def iterations(self) -> int:
        return self.engine.iterations

    @property
    def finished(self) -> bool:
        return self.engine.unfinished_requests == 0

    def step(self) -> ScheduledBatch:
        """Run the next iteration the ordinary way and return its schedule."""
        schedule, _ = self.engine.next_schedule()
        finish = self.executor.complete_iteration(schedule)
        self.engine.complete(schedule, finish)
        self._last = schedule
        return schedule

    def advance(self, limit: int | None = None) -> tuple[ScheduledBatch, int]:
        """Apply the next iterations and return the first of them with their count.

        As many repeats of the last decode-only iteration as the scheduler
        and the KV pool allow are applied at once, at most ``limit``;
        otherwise one iteration is stepped. Every applied iteration is the
        returned schedule shifted by its index.
        """
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ServeLoopError("the advance limit must be a positive integer")
        if self._last is not None:
            frontier = self.executor.frontier_ns
            horizon = self.engine.decode_horizon(
                self._last, frontier_ns=frontier, iteration_ns=self.executor.iteration_ns)
            if limit is not None:
                horizon = min(horizon, limit)
            if horizon >= 1:
                applied = self.engine.advance_decode(
                    self._last, horizon, frontier_ns=frontier, iteration_ns=self.executor.iteration_ns)
                if applied:
                    first = shifted(self._last, 1, not_before_ns=frontier)
                    self._last = shifted(self._last, applied,
                                         not_before_ns=frontier + (applied - 1) * self.executor.iteration_ns)
                    return first, applied
        return self.step(), 1

    def run_until(self, iterations: int) -> None:
        """Advance until exactly ``iterations`` iterations have completed."""
        if iterations < self.engine.iterations:
            raise ServeLoopError("the replay is already past the requested iteration")
        while self.engine.iterations < iterations:
            if self.finished:
                raise ServeLoopError("the request stream ends before the requested iteration")
            self.advance(iterations - self.engine.iterations)

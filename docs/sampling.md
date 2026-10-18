# Sampled windows: timing a share of the iterations

A closed-loop window of thousands of iterations costs the engine one full
pass per iteration, while consecutive iterations of a steady phase take
nearly the same time. `serveloop run --sample-rate R` times only a sample
of the iterations in full and gives the others the running mean of their
stratum, so the engine's work shrinks roughly with `R` while the scheduler,
the KV state and every write still run for every iteration.

```sh
python3 -B -m serveloop run \
  --model models/llama31-8b-w8-kv-bf16.json --system configs/systems/miniquick/4hbm-4hbf.cfg \
  --placement weights-hbf-kv-hbm --requests stream.json --run-config run-config.json \
  --sample-rate 0.2 --simulator ../HBFSim/build/hbfsim --out out/sampled
```

## What happens to a skipped iteration

Its batch is mapped as usual, then reduced to what changes the engine's
state (`hbfsim_client.sampling.state_only`): writes, link copies, peer
accesses and barriers stay; reads of HBM, HBF and the external tiers are
left out, and a transaction that depended on a dropped read now depends on
what that read depended on. The batch ends with a barrier issued at the
estimated duration, so the completion frontier advances by the estimate or
by the kept work, whichever is later, and the next iteration is scheduled
from there.

## Strata and the estimate

Iterations are stratified by what they read: the iteration kind (prefill,
decode, mixed), the number of decoding sequences, and the prefill and
context tokens by power of two (`serveloop.hbfsim.batch_stratum`). Every
stratum's first `--sample-pilot` iterations (default 2) are timed; after
that each iteration is timed with probability `--sample-rate`, drawn from
`--sample-seed`. A timed iteration that follows a skipped one finds the
engine's caches as the skipped iterations left them, so `--sample-warmup`
(default 1) such iterations are timed but kept out of the estimate.

The report in the result's `sampling` section (`hbfsim.sampled_batches`)
gives, per stratum, the iterations, the timed and counted ones, the mean and
the sample variance, and for the window the stratified estimate
`Σ N_h · mean_h`, its standard error from the finite-population stratified
variance, and a 95% interval. A stratum with a single counted iteration has
no variance; its skipped iterations are counted as `unknown_variance_batches`
and lie outside the interval, which the headline says when there are any.

## What the numbers mean

- The estimate is for the sequence of iterations the run produced. Because
  the scheduler releases arrivals against a clock that mixes timed and
  estimated durations, the sequence can differ from a fully timed run's
  where arrivals are dense; with every request waiting from the start it is
  the same sequence.
- Request latencies (TTFT, TPOT, e2e) are read off that clock and are
  therefore partly estimated; the headline says so.
- Physical device counters (HBF reads, wear, GC) cover only the timed
  iterations' reads plus every iteration's writes.
- `--sample-rate 1` times everything and reports an exact window.

`tests/python/test_sampling.py` in HBFSim checks the estimator's coverage on
synthetic windows; the end-to-end comparison with a fully timed window is in
the ServeLoop tests that need the engine.

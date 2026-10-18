# Decode snapshots: fast starts for long contexts

A closed-loop run over a long request stream spends most of its time on the
iterations that produce the state of interest, not on the state itself: a
128K-token context must be prefilled chunk by chunk, and every one of a
request's output tokens is a scheduler iteration whose attention reads sweep
the whole KV cache. `serveloop snapshot` replays the stream without timing
it, picks a representative decode iteration, and writes what a timed run
needs to start there. `serveloop run --prefix-image` then starts from that
state and times only the iterations you ask for.

## What a snapshot is

A snapshot is the closed loop's state after one iteration of a *logical
replay*: the real scheduler and the real placement run over the stream, but
every iteration takes a declared time (`--iteration-ns`, default 50 ms)
instead of a physical completion. Admission, chunked prefill, prefix reuse,
paged-KV allocation, migration and preemption are therefore exactly what the
timed loop would decide under that clock. The replay is exact and cheap:
decode-only stretches are advanced in one step (below), so a stream with
hundreds of thousands of iterations replays in seconds.

The snapshot holds:

- the in-flight sequences and how much context each has processed, rounded
  down to whole KV blocks;
- the KV pool layout: which block ids those contexts occupy, holes included;
- the prefix cache: the prompt blocks it retains besides, in recency order.

Two files carry this state. `requests.json` is an ordinary request trace with
one request per in-flight sequence: its prompt is the sequence's context
followed by `--suffix-tokens` private tokens, and it asks for
`--output-tokens` outputs. `prefix-image.json` lists the prompts whose
blocks the cache holds and, oldest first, every retained block with its block
id per layer. `snapshot.json` records the selection, the per-sequence
contexts and the identities of every input.

## Resuming

```sh
python3 -B -m serveloop snapshot \
  --model models/llama31-8b-w8-kv-bf16.json --system configs/systems/miniquick/4hbm-4hbf.cfg \
  --placement weights-hbf-kv-hbm --requests stream.json --run-config run-config.json \
  --prefix-cache-bytes 2147483648 --simulator ../HBFSim/build/hbfsim --out out/snapshot

python3 -B -m serveloop run \
  --model models/llama31-8b-w8-kv-bf16.json --system configs/systems/miniquick/4hbm-4hbf.cfg \
  --placement weights-hbf-kv-hbm --run-config run-config.json \
  --requests out/snapshot/requests.json --prefix-image out/snapshot/prefix-image.json \
  --prefix-cache-bytes 2258632704 --simulator ../HBFSim/build/hbfsim --out out/resumed
```

The run installs the image before its first request: the blocks keep the ids
they had, so the pool reproduces the replayed layout, and the cache entries
keep their order. Each resumed request then hits its whole context, prefills
only the suffix in the first iteration, and decodes from there. The image
contains the contexts as well as the retained prompts, so the run's prefix
budget must be at least the image's size; `serveloop snapshot` prints that
minimum. Entry identity is recomputed from the prompt tokens at installation
with the placement's own key function, which is why the stream must carry
token ids.

`TPOT` of a resumed run is the mean over the decode iterations it asked for.
The prefill of the suffix is one iteration; nothing in the run measures the
history the snapshot replaced.

## How the iteration is chosen

Candidates are the iterations at which every batch slot is busy, requests
still wait, and the prefix cache (when the run has one) has reached its
budget. Among them the snapshot takes the iteration whose running sequences
hold the median total context, the earliest on a tie. This is the empirical
form of the steady-state view: under full occupancy, the sequences in flight
at a random instant are spread over their lifetimes, and the median-context
iteration is the typical one. `snapshot.json` reports how many candidates
there were and where the chosen context sum ranks among them.

## Why the replay is exact

Between two scheduling events, a decode-only iteration repeats the previous
one with every request one token further along. The scheduler can say how
long that lasts (`ServeLoopEngine.decode_horizon`): until a running request
emits its last token, a preempted request waits to resume, or a free slot
meets an arrived request. The placement applies those iterations at once
(`ServeLoopPlacement.advance_decode`), allocating blocks at exactly the block
boundaries the individual iterations would cross, and stops short of any
iteration that would need an eviction, a migration or a preemption, which
only a real iteration decides. Publishing the prompt blocks of a decode
iteration only reorders the prefix cache once every key is present, and the
same publications reproduce the same order, so after the first iteration that
changes nothing but the order the remaining ones are skipped. Streams whose
running prompts exceed the prefix budget never settle and are stepped; TTL
entries cap a jump at the first expiry. The tests in `tests/test_replay.py`
compare the jumped replay with the stepped one state for state at every
iteration, over pools with room, pools that evict and migrate, shared
prefixes, TTLs and a thrashing cache.

## Limits

- Contexts resume from HBM. A sequence whose KV the replay had migrated to a
  cold tier is not snapshotted; the command says so.
- The logical replay does not activate externally cached models; multi-model
  cache state is whatever the run starts with.
- A resumed run's prefix budget is at least the image's size, which may be
  larger than the replay's budget: what the budget does afterwards is the
  resumed run's own policy.
- Calibrated timing's prefill limits shape the replay (chunk sizes change
  with the context length), but the replay clock is declared, not computed.

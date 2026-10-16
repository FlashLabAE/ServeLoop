# Contributing

ServeLoop values one auditable current path over compatibility layers and
unlabeled approximations. Please open an issue before a large schema or
scheduler change.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -B tests/test_serveloop.py
```

Pass `--simulator /path/to/hbfsim` to the test command when a compatible
HBFSim build is available.

## Changes that belong in HBFSim

`hbfsim_client/`, `configs/systems/` and `configs/parameter-provenance.json`
are copies from [HBFSim](https://github.com/FlashLabAE/HBFSim). Change
them there first, then copy the files here, update the commit, hashes and
changed parameters in `configs/ocp-profile-source.json`, and run:

```bash
python3 -B tests/test_config_compatibility.py \
  --hbfsim-root ../HBFSim --simulator ../HBFSim/build/hbfsim
python3 -B tests/test_session_compatibility.py --simulator ../HBFSim/build/hbfsim
```

If the change alters what ServeLoop sends, also raise
`HBFSIM_SERVELOOP_MIN_REVISION` in HBFSim's `CMakeLists.txt` and
`SERVELOOP_REVISION` in its CI workflow.

## Evidence rules

- Never commit private production traces, credentials, hostnames, usernames,
  or absolute local paths.
- Give every external input a source, immutable identity when available, and
  SHA-256 digest in generated receipts.
- Mark synthetic traces and sensitivity assumptions as synthetic. They cannot
  become measurement evidence through downstream simulation.
- Add model bytes through the public descriptor derivation or an explicit
  `serveloop.model` ledger; do not insert an implicit parameter-count shortcut.
- Preserve placement-independent canonical bytes and require mapped traffic to
  conserve them exactly.
- State calibration and holdout populations separately. Do not tune against the
  same run used as validation evidence.
- Remove obsolete implementations when replacing them. Do not add hidden
  compatibility branches.

Keep tests proportional to the change: contract and accounting changes need
focused unit tests; HBFSim protocol changes need the physical integration
suite as well.

# Origin notice

The initial ServeLoop codebase was extracted from the MIT-licensed HBFSim working
tree based on commit `b44a7960ea44e20252aa6c8e758e8f9ed2d7a4b5`. It was renamed,
made independently installable, and stripped of production evidence and local
experiment history for this repository.

ServeLoop retains the original HBFSim copyright notice in `LICENSE`.

`hbfsim_client/`, `configs/systems/` and `configs/parameter-provenance.json`
are copies of the same paths in [HBFSim](https://github.com/FlashLabAE/HBFSim)
at commit `514e796425c37eca4047c1ae3a4e6b9fff174fe0`, distributed under the same
MIT License; `configs/ocp-profile-source.json` records their source hashes.
The 70B descriptor preserves the original per-output-channel W8 scale and
embedding storage assumptions through the public model ledger. Paper-specific
analysis runners, production evidence, and local experiment history are not
included.

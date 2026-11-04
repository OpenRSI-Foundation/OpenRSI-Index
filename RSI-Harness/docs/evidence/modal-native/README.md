# Modal native runtime acceptance evidence

These are **synthetic infrastructure fixtures**, not coding-task evaluation,
model-quality results, or evidence of recursive improvement. The fixture
deliberately submits values 0 and 1 to test native accounting and selection.

The tested implementation is commit `4c4c2cb` on `codex/modal-native-runtime`.
The receipt records each immutable Modal image ID, physical GPU UUID (when
applicable), run ID, native result, and unchanged run deadline. Files below
`native/` were downloaded directly from the run's Modal Volume without editing.
Later commits add evidence and format fixture sources without changing runner logic.

| Check | Observed result | Evidence |
| --- | --- | --- |
| H100 native two-round loop | Both rounds completed; Judge opened a CUDA context and copied data through GPU memory on the same H100 80 GB; isolated candidate used a passed Unix-socket descriptor and could not open GPU devices, tests, or verifier output | [Receipt](gpu/receipt.json), [native state](gpu/native/evolve_state.json), [Judge output](gpu/native/feedback/agent-2.log) |
| Two-allocation resume | First segment interrupted after a completed round; second retained workspace and home, continued as `agent-2`, and preserved the original run ID and deadline | [Receipt](resume/receipt.json), [native state](resume/native/evolve_state.json) |
| Nonzero verifier after reward write | Both verifier invocations wrote rewards and then exited 7; both remained unscored, with no best result | [Receipt](verifier-error/receipt.json), [native result](verifier-error/native/final_result.json) |

The H100 allocation requested 16 CPUs and 131072 MiB of RAM. Work had no GPU
device mount; both Judges used the allocation's single physical GPU. CPU
acceptance used 1 CPU and 2048 MiB. A prior CPU two-round fixture also passed.
No fixed coding-model weights were downloaded and no model API was called.

Validation: Ruff passed; 305 focused tests passed across the Modal contracts,
CLI, native coordinator, resume, rewards, submissions, and trajectory handling.
Two unchanged Docker trajectory mocks were excluded on macOS because they
assume Linux and omit the Docker Desktop `info()` interface. Cloud acceptance
ran on Linux. Full 12-hour and 24-hour experiments have not been run.

Reproduce the cloud checks with `scripts/validate_modal_native.py` and a bounded
operator config, as described in [the runtime guide](../../modal-native.md).

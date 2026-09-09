# Anti-cheat battery

Adversarial verification that the frozen verifier penalizes cheating. It
builds an honest control plus one submission per hack vector and runs each
through the real `tests/{policy_check,evaluate,score}.py`, asserting every
cheat scores `reward = 0.0` (at its intended gate) while the honest control
reaches scoring. A toy model (deterministic image = f(checkpoint, seed,
key)) stands in for the 1.1B PixelGen model so the battery runs in ~1 min on
one GPU — the spot gate is model-agnostic, so this faithfully tests whether
cheats are caught.

Run inside the tests-equivalent environment with torch and a warm DINOv2
hub cache. `GPIC_WORK_ROOT` selects the staged assets directory and defaults
to the `gpic_generation` task directory. It must contain `gpic/gpic_eval`,
`torch_home`, `hf_models/Qwen3-1.7B`, `gpic_data/test`, and
`gpic_verifier/reference_stats/test_stats.npz`. The battery creates `cheat/`
under that directory for its temporary captions and sandbox. The verifier
source is always taken from this task checkout.

```bash
GPIC_WORK_ROOT=/path/to/staged-assets bash tests/cheat/battery_payload.sh
# Run from the gpic_generation task directory.
```

`battery_results.json` is the recorded 13/13 pass from 2026-08-24 (each row:
case, reward, ok, and the gate/reason that penalized it).

Not a Harbor entrypoint — a developer harness. When the baseline is sealed,
re-run it with the real checkpoint to calibrate `GPIC_SPOT_BIND_MAX_COS` /
`GPIC_SPOT_PERTURB_STD` from the model's measured clean-vs-perturbed cosine.

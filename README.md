<p align="center">
  <img src="assets/figures/rsi-first-exam-header-starry.svg" alt="RSI's First Exam" width="567">
</p>

<div align="center">
  <a href="https://rsi-first-exam.rsi-anything.workers.dev/"><img src="https://img.shields.io/badge/Website-2563EB?style=for-the-badge&logo=googleearth&logoColor=white" alt="Website"></a>
  <a href="https://github.com/RSI-Index/RSIs-First-Exam"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <!-- TODO: Replace the X placeholder with the project profile or announcement URL. -->
  <!-- <a href="#"><img src="https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=X&logoColor=white" alt="X"></a> -->
  <a href="https://github.com/RSI-Index/RSIs-First-Exam/blob/main/CONTRIBUTING.md"><img src="assets/figures/contribution-call.svg" alt="Contribution Call"></a>
  <!-- TODO: Replace the Discord placeholder with the project invite URL. -->
  <a href="https://discord.gg/3EG8Qhmfsa"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="assets/figures/wechat.png"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat Group"></a>
</div>

## 📣 Call for Contributors

**We're actively looking for contributors to add new, challenging tasks.** Our dedicated **agent-native RSI-Anything pipeline** helps you create a new task in **an hour or less**—bring the idea, and our agents handle the rest. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for a step-by-step guide to creating and submitting tasks. Have fun! 😀

Clone the repository:

```bash
git clone https://github.com/RSI-Index/RSIs-First-Exam.git
cd RSIs-First-Exam
```

Open Codex or Claude Code (app or CLI), start a fresh session, and paste:

```text
Use the proposal-agent skill at .agents/skills/proposal-agent/SKILL.md in my local RSIs-First-Exam repository. Locate the repository if needed, then follow the skill to complete setup and guide me step by step through creating and submitting an RSI proposal.
```

## 💥 Why RSI's First Exam?

We are witnessing the dawn of a new era: AI is entering a recursive self-improvement loop. The central question is whether this loop can move beyond the best-known human-designed method and reliably extend the scientific frontier. Answering it requires careful measurement, and building that measurement is the purpose of this project.

[RSI's First Exam](https://rsi-first-exam.rsi-anything.workers.dev/) is an ongoing effort to evaluate whether AI agents can drive genuine recursive self-improvement and advance scientific discovery through real-world research at scales ranging from a single node to thousands of GPUs—not merely reproduce existing results, sweep parameters, or succeed on toy-scale tasks.

### Task taxonomy

We are actively developing this project and welcome [contributions](https://github.com/RSI-Index/RSIs-First-Exam/blob/main/CONTRIBUTING.md) from the community. Tasks and execution logs are available in [rsi-tasks/](rsi-tasks/) and [rsi-logs/](rsi-logs/), respectively. See [Quick Start](assets/docs/quick-start.md) for instructions on running these tasks.

<details>
<summary><strong>View tasks</strong></summary>

| Task                                                                | Track          | Category      | Description                                                           | GPU requirement |
| ------------------------------------------------------------------- | -------------- | ------------- | --------------------------------------------------------------------- | --------------- |
| `Qwen-122B-RL`                                                      | Signature Task | Post-training | Optimize a full post-training stack under one fixed budget.           | 256× H100       |
| [`marin-optimizer-update-geometry`](rsi-tasks/signature-tasks/pre-training-optimizer-update-geometry/) | Signature Task | Pre-training  | Design a scale-general optimizer for the Marin scaling ladder.        | 256× H100       |
| [`gpic-text-to-image`](rsi-tasks/signature-tasks/gpic_generation/)     | Signature Task | Vision        | Train a text-to-image model on one epoch of GPIC.                     | 256× H100       |
| [`depth-width-allocation`](rsi-tasks/depth-width-allocation/)       | Public Task    | Pre-training  | Optimize decoder-width allocation under a fixed 200M training budget. | 8× H100         |
| [`learnability-cot`](rsi-tasks/learnability-cot/)                   | Public Task    | Post-training | Adapt reasoning traces for small-model math SFT.                      | 4× H100         |
| [`gemm-h100-refined`](rsi-tasks/gemm-h100-refined/)                 | Public Task    | MLSys         | Optimize an FP16 CUDA GEMM kernel for H100 throughput.                | 1× H100         |
| [`liger-tied-ce`](rsi-tasks/liger-tied-ce/)                         | Public Task    | MLSys         | Optimize tied-weight fused cross-entropy for Qwen3 SFT.               | 2× H100         |
| [`minference-sparse-prefill`](rsi-tasks/minference-sparse-prefill/) | Public Task    | MLSys         | Optimize Triton sparse-prefill attention on H100.                     | 1× H100         |
| [`molmo2-pointing-refined`](rsi-tasks/molmo2-pointing-refined/)     | Public Task    | Vision        | Optimize Molmo2 video-pointing at inference time.                     | 1× H100         |
| [`datacomp-small-filtering`](rsi-tasks/datacomp-small-filtering/)   | Public Task    | Vision        | Curate DataComp-small data for fixed ViT-B/32 training.               | 32× H100        |

</details>

### How an evaluation works: RSI-Harness

[RSI-Harness](RSI-Harness/) powers RSI's First Exam for ultra-long-horizon RSI runs, natively supporting [Harbor-format tasks](rsi-tasks) from single-node local Docker to multi-node clusters.

## 🤝 Contributors



## 🙏 Acknowledgements

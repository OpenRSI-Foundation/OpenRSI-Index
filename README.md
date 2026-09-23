<p align="center">
  <img src="assets/figures/openrsi-index-header-starry.svg" alt="OpenRSI Index" width="567">
</p>

<div align="center">
  <a href="https://index.openrsi.foundation/index.html"><img src="https://img.shields.io/badge/Website-4F86D8?style=for-the-badge&logo=googleearth&logoColor=white" alt="Website"></a>&nbsp;
  <a href="https://github.com/OpenRSI-Foundation/OpenRSI-Index"><img src="https://img.shields.io/badge/GitHub-35434D?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>&nbsp;
  <a href="#"><img src="https://img.shields.io/badge/Twitter-35434D?style=for-the-badge&logo=X&logoColor=white" alt="X"></a>&nbsp;
  <a href="https://github.com/OpenRSI-Foundation/OpenRSI-Index/blob/main/CONTRIBUTING.md"><img src="assets/figures/contribution-call.svg" alt="Contribution Call"></a>&nbsp;
  <img src="https://img.shields.io/badge/RSI%20Logs-14967F?style=for-the-badge&logo=google-sheets&logoColor=white" alt="RSI Logs">&nbsp;
  <a href="assets/figures/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-24A36F?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat Group"></a>&nbsp;
  <a href="https://discord.gg/3EG8Qhmfsa"><img src="https://img.shields.io/badge/Discord-6674D8?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
</div>

## 📣 Call for Contributors

**We're actively looking for contributors to add new, challenging tasks.** Our dedicated **agent-native RSI-Anything pipeline** helps you create a new task in **an hour or less**—bring the idea, and our agents handle the rest. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for a step-by-step guide to creating and submitting tasks. **Contributors with accepted tasks will be invited as co-authors.** Have fun! 😀

Clone the repository:

```bash
git clone https://github.com/OpenRSI-Foundation/OpenRSI-Index.git
cd OpenRSI-Index
```

Open Codex or Claude Code (app or CLI), start a fresh session, and paste:

```text
Use the proposal-agent skill at .agents/skills/proposal-agent/SKILL.md in my local OpenRSI-Index repository. Locate the repository if needed, then follow the skill to complete setup and guide me step by step through creating and submitting an RSI proposal.
```

**Any questions?** Contact us at [yuetaili@uw.edu](mailto:yuetaili@uw.edu), [zhuofengli12345@gmail.com](mailto:zhuofengli12345@gmail.com), or [yfeng42@uw.edu](mailto:yfeng42@uw.edu).

## 🔥 Call for Compute

If you have compute resources to run experiments and want to build frontier RSI environments together, contact [yuetaili@uw.edu](mailto:yuetaili@uw.edu), [zhuofengli12345@gmail.com](mailto:zhuofengli12345@gmail.com), or [hanxinyang@berkeley.edu](mailto:hanxinyang@berkeley.edu).

## 💥 Why OpenRSI Index?

We are witnessing the dawn of a new era: AI is entering a recursive self-improvement loop. The central question is whether this loop can move beyond the best-known human-designed method and reliably extend the scientific frontier. Answering it requires careful measurement, and building that measurement is the purpose of this project.

[OpenRSI Index](https://index.openrsi.foundation/index.html) is an ongoing effort to evaluate whether AI agents can drive genuine recursive self-improvement and advance scientific discovery through real-world research at scales ranging from a single node to thousands of GPUs—not merely reproduce existing results, sweep parameters, or succeed on toy-scale tasks.

### Task taxonomy

We are actively developing this project and welcome [contributions](https://github.com/OpenRSI-Foundation/OpenRSI-Index/blob/main/CONTRIBUTING.md) from the community. Tasks and execution logs are available in [rsi-tasks/](rsi-tasks/) and [rsi-logs/](rsi-logs/), respectively. See [Quick Start](assets/docs/quick-start.md) for instructions on running these tasks.

<details>
<summary><strong>View tasks</strong></summary>

| Task                                                                | Track          | Category      | Description                                                           | GPU requirement |
| ------------------------------------------------------------------- | -------------- | ------------- | --------------------------------------------------------------------- | --------------- |
| [`Qwen-122B-RL`](rsi-tasks/signature-tasks/post-training-qwen-122B-rl/) · [Research report](rsi-logs/signature-tasks/post-training-qwen-122B-rl/) | Signature Task | Post-training | Optimize a full post-training stack under one fixed budget.           | 256× H100       |
| [`marin-optimizer-update-geometry`](rsi-tasks/signature-tasks/pre-training-optimizer-update-geometry/) | Signature Task | Pre-training  | Design a scale-general optimizer for the Marin scaling ladder.        | 256× H100       |
| [`gpic-text-to-image`](rsi-tasks/signature-tasks/gpic_generation/) · [Research trajectories](rsi-logs/signature-tasks/gpic-10m-autoresearch/) | Signature Task | Vision        | Improve a shared PixelGen checkpoint on a fixed GPIC 10M subset.      | 4× H100 per job |
| [`depth-width-allocation`](rsi-tasks/depth-width-allocation/)       | Public Task    | Pre-training  | Optimize decoder-width allocation under a fixed 200M training budget. | 8× H100         |
| [`learnability-cot`](rsi-tasks/learnability-cot/)                   | Public Task    | Post-training | Adapt reasoning traces for small-model math SFT.                      | 4× H100         |
| [`gemm-h100-refined`](rsi-tasks/gemm-h100-refined/)                 | Public Task    | MLSys         | Optimize an FP16 CUDA GEMM kernel for H100 throughput.                | 1× H100         |
| [`liger-tied-ce`](rsi-tasks/liger-tied-ce/)                         | Public Task    | MLSys         | Optimize tied-weight fused cross-entropy for Qwen3 SFT.               | 2× H100         |
| [`minference-sparse-prefill`](rsi-tasks/minference-sparse-prefill/) | Public Task    | MLSys         | Optimize Triton sparse-prefill attention on H100.                     | 1× H100         |
| [`molmo2-pointing-refined`](rsi-tasks/molmo2-pointing-refined/)     | Public Task    | Vision        | Optimize Molmo2 video-pointing at inference time.                     | 1× H100         |
| [`datacomp-small-filtering`](rsi-tasks/datacomp-small-filtering/)   | Public Task    | Vision        | Curate DataComp-small data for fixed ViT-B/32 training.               | 32× H100        |
| [`ace-playbook-repair`](rsi-tasks/ace-playbook-repair/)             | Public Task    | Agents        | Inspect and repair ACE playbooks for a frozen Qwen model.            | 1× H100         |
| [`molmoweb-interaction-context`](rsi-tasks/molmoweb-interaction-context/) | Public Task | Vision        | Allocate inference-time context for frozen MolmoWeb replay.          | 2× H100 minimum |
| [`reasonir-difficulty-curriculum`](rsi-tasks/reasonir-difficulty-curriculum/) | Public Task | Post-training | Optimize a fixed-budget retrieval LoRA through difficulty-aware curricula. | 4× H100 minimum |
| [`isaaclab-peginsert-reward-search`](rsi-tasks/isaaclab-peginsert-reward-search/) | Public Task | Robotics      | Design bounded reward graphs for fixed-budget Isaac Lab peg insertion. | 4× H100         |
| [`kev-decision-architecture`](rsi-tasks/kev-decision-architecture/) | Public Task | Post-training | Redesign a 0.5B decision architecture under a fixed training corpus and recipe. | 2× H100 minimum |

</details>

### How an evaluation works: RSI-Harness

[RSI-Harness](RSI-Harness/) powers OpenRSI Index for ultra-long-horizon RSI runs, natively supporting [Harbor-format tasks](rsi-tasks) from single-node local Docker to multi-node clusters.

## 🤝 Contributors

We are still actively expanding the team. We really appreciate the efforts of all our excellent members.

<table>
  <tr>
    <th colspan="4" align="left">PROJECT INITIATORS</th>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://yuetl9.github.io/">Yuetai Li</a></strong><br><a href="mailto:yuetaili@uw.edu">yuetaili@uw.edu</a></td>
    <td width="25%" valign="top"><strong><a href="https://zhuofeng-li.github.io/">Zhuofeng Li</a></strong><br><a href="mailto:zhuofengli12345@gmail.com">zhuofengli12345@gmail.com</a></td>
    <td width="25%" valign="top"><strong><a href="https://yichenfeng.me/">Yichen Feng</a></strong><br><a href="mailto:yfeng42@uw.edu">yfeng42@uw.edu</a></td>
    <td width="25%" valign="top"><strong><a href="https://davidhanx.github.io/">Xinyang Han</a></strong><br><a href="mailto:hanxinyang@berkeley.edu">hanxinyang@berkeley.edu</a></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://wenhaochai.com/">Wenhao Chai</a></strong><br><a href="mailto:wenhao.chai@princeton.edu">wenhao.chai@princeton.edu</a></td>
    <td width="25%" valign="top"><strong><a href="https://noviscl.github.io/">Chenglei Si</a></strong><br><a href="mailto:sichenglei1125@gmail.com">sichenglei1125@gmail.com</a></td>
    <td width="25%"></td>
    <td width="25%"></td>
  </tr>
  <tr>
    <th colspan="4" align="left">ORGANIZERS (in alphabetical order)</th>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://scholar.google.com/citations?user=E1GCDXUAAAAJ">Shangding Gu</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://rilynhan.com/">Rilyn Han</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://dblp.org/pid/270/4119.html">Zhengyu Hu</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://hanghuacs.notion.site/">Hang Hua</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://howiehwong.github.io/">Yue Huang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://caralinotes.com/">Cara Li</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://hanchenli.github.io/">Hanchen Li</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://www.linkedin.com/in/shujia-liang/">Shujia Liang</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://scholar.google.com/citations?user=Dj9s3oEAAAAJ">Xiang Liu</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://lupantech.github.io/">Pan Lu</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://bohanlyu.com/">Bohan Lyu</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://zixianma.github.io/">Zixian Ma</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://joyemang33.github.io/">Qiuyang Mang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://rulinshao.github.io/">Rulin Shao</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://huggingface.co/venkat-srinivasan-nvidia">Venkat Srinivasan</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://sunyiyou.github.io/">Yiyou Sun</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://scholar.google.com/citations?user=cIRPBeYAAAAJ&amp;hl=en">Guan Wang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://www.linkedin.com/in/nwangucla/">Ning Wang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://hsaest.github.io/">Jian Xie</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://github.com/bri25yu">Brian Yu</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://mitibm.mit.edu/people/gaoyuan-zhang/">Gaoyuan Zhang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://alex-q-z.github.io/">Qizheng Zhang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://openreview.net/profile?id=~Weichen_Zhang8">Weichen Zhang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://scholar.google.com/citations?user=6kkyR1wAAAAJ">Kaiyuan Zheng</a></strong></td>
  </tr>
  <tr>
    <th colspan="4" align="left">ADVISORS (in alphabetical order)</th>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://wenhuchen.github.io/">Wenhu Chen</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://people.eecs.berkeley.edu/~akcheung/">Alvin Cheung</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://yejinc.github.io/">Yejin Choi</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://www.microsoft.com/en-us/research/people/jfgao/">Jianfeng Gao</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://homes.cs.washington.edu/~hannaneh/">Hannaneh Hajishirzi</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://koh.pw/">Pang Wei Koh</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://ranjaykrishna.com/">Ranjay Krishna</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://hanqinglu.github.io/">Hanqing Lu</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://www.cs.princeton.edu/~karthikn/">Karthik Narasimhan</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://arsenalfc.stanford.edu/kunle/">Kunle Olukotun</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://rpand002.github.io/">Rameswar Panda</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://people.ece.uw.edu/radha/">Radha Poovendran</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://dawnsong.io/">Dawn Song</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://ysu1989.github.io/">Yu Su</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://u.osu.edu/ihudas/">Huan Sun</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://cs.stanford.edu/~diyiy/">Diyi Yang</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://faculty.ucmerced.edu/mhyang/">Ming-Hsuan Yang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://xiangyue9607.github.io/">Xiang Yue</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://www.linkedin.com/in/jian-zhang-10383a98/">Jian Zhang</a></strong></td>
    <td width="25%" valign="top"><strong><a href="https://yuzhimanhua.github.io/">Yu Zhang</a></strong></td>
  </tr>
  <tr>
    <td width="25%" valign="top"><strong><a href="https://homes.cs.washington.edu/~lsz/">Luke Zettlemoyer</a></strong></td>
    <td width="25%"></td>
    <td width="25%"></td>
    <td width="25%"></td>
  </tr>
  <tr>
    <th colspan="4" align="left">TASK CONTRIBUTORS</th>
  </tr>
  <tr>
    <td colspan="4">Coming soon.</td>
  </tr>
</table>

正式最优结果不变。agent-4 为 completed，reward=score=1.5389225880993647；正式接受、匹配 baseline、评测源码版本尚未确认。新增证据确认 H15 拒绝记录已存在，并补齐 P11/P12/P13 历史诊断。P14 已推进至 HiFT 准备完成、Flow 使用 ragged FA3；采样、QPS、CPU、WER及完成状态尚未确认。

`R = logs/runs/omni-resume3-669485-20261004T141956Z-resume3/sglang-omni-tts-runtime`。下文 `R/文件:行号` 为原始路径和行号。历史承接 previous-summary.md，未重读历史原文件。

| 研究方向／阶段 | 关键改动 | 实测 metrics | baseline／设置 | 失败项与结论；原始证据 |
|---|---|---|---|---|
| 正式 agent-4 | 源码版本尚未确认 | exit0；2207.972579 s；completed；reward=score=1.5389225880993647；best_round=agent-4；未超时 | 匹配 baseline、分模型 QPS/WER 尚未确认 | 正式接受尚未确认。`R/submissions/agent-4/report.json:1–15`；`R/evolve_state.json:2–3,29–37` |
| H20／正式 agent-2 | H1+H13+H19+H20；preflight、四路径 scope_checked、三 lane CUDA/UUID 检查通过 | CLI exit0、597.248483320 s；核验 exit1、543.401290413 s；metrics={}、score=null、reward=null；未超时 | 研究侧 generation0；Judge digest 尚未确认 | infrastructure／verifier_error。MOSS rep-0 authoritative ASR 在 localhost:8001 普通 bind 报 OSError98；其他 lane 取消。九轮均值、guards、接受尚未确认。`R/agent_output.txt:447319–447352,458723–458792,487681–487707`；`R/submissions/agent-2/report.json:1–13` |
| agent-2 MOSS 单轮 | TTS及候选侧诊断ASR | 1088完成、0失败；11.557 QPS；51.753音频s/s；latency均值/中位/p95/p99=1.379/1.365/1.750/1.981 s；RTF=0.3177/0.3044/0.4132/0.4618；corpus WER2.39% | rep-0、en、并发16；ASR1088/1088、0跳过；预期seed42，stdout未返回seed | 单轮诊断WER高于2%；不是authoritative三轮均值或正式guard失败。过滤corpus1.39%不替代全量。`R/agent_output.txt:458793–458899,458929–458951,487647–487678` |
| agent-2 Cosy／Base | 同级失败后取消 | Cosy warmup16/16；最后157/1088；完整QPS/WER尚未确认 | 各rep-0；Cosy端口8003、并发16、64/64；1088条、666个唯一音频 | Cosy14:15:25.693、Base14:15:25.573停止；各1个semaphore警告。31个CancelledError末端；未找到独立推理错误末端不等于推理通过。`R/agent_output.txt:459032–459054,460895–460927,466943–467025,487638–487677` |
| 正式 agent-3 | 版本尚未确认 | exit1；2201.470971 s；metrics={}、score=null；未超时 | 尚未确认 | verifier_error、rewards={}；细因尚未确认。`R/submissions/agent-3/report.json:1–13` |
| 正式 agent-5 | 版本尚未确认 | exit1；772.075777 s；metrics={}、score=null；未超时 | 尚未确认 | verifier_error；Base rep-0 authoritative native_command_failure，actual=1；根因尚未确认。`R/submissions/agent-5/report.json:1–13`；`R/feedback/agent-5.log:13254–13255` |
| agent-5 Cosy 单轮 | TTS后启动ASR | 1088完成、0失败；13.209 QPS；62.655音频s/s；latency=1.188/1.152/1.633/1.896 s；RTF=0.2636/0.2504/0.376/0.505 | rep-0、en、并发16、64/64、warmup16；seed尚未确认 | WER及三轮结果尚未确认。ASR后OSError22、lane_exception；栈含psutil.wait_procs、wait_pid_pidfd_open。未确认WER拒绝或端口占用。`R/feedback/agent-5.log:12837–12884,13146–13252` |
| H19 v3／正式 agent-1 | H1+H13+H19 v3四文件；preflight、scope_checked通过 | CLI exit0、587.578133 s；核验exit1、540.220679 s；metrics={}、score=null、reward=null | generation0；17:03:18.237213–17:13:05.838899 UTC | infrastructure／verifier_error；complete_evaluation=false、baseline_promotion=false。MOSS authoritative localhost:8001 bind报OSError98；其他lane取消。`R/agent_output.txt:178275–178318,179302–179325,192162–192225,451435–451489` |
| agent-1 MOSS 单轮 | TTS及诊断ASR | 1088完成、0失败；11.672 QPS；52.283音频s/s；latency=1.365/1.359/1.732/1.914 s；RTF=0.3141/0.3042/0.4026/0.4342；corpus WER2.24% | rep-0、en、非stream、并发16、64/64；预期seed42，stdout未返回seed | ASR1088评估、0跳过；过滤corpus1.26%不替代全量。`R/agent_output.txt:253783–253908,255491–255492,257026–257030,451492–451527` |
| agent-1 Base／Cosy 单轮 | Base保存TTS；Cosy部分计分 | Base1088完成、0失败；21.937 QPS；90.302音频s/s；latency=0.726/0.708/1.012/1.382 s；RTF=0.1825/0.1729/0.2506/0.3896；Cosy最后102/1088 | 各rep-0、并发16、64/64 | 同级失败后取消。Base WER、Cosy完整QPS/WER尚未确认。`R/agent_output.txt:253949–253975,257026,451529–451560`；`R/feedback/agent-1.log:7428–7484,8325–8399` |
| H20-D1 split局部诊断 | stage间保留H19 tensor；terminal改typed bytes；generate／transcribe分阶段 | 六轮exit0；6528计分请求及转录、96 warmups；0失败、0跳过；QPS+2.29445%；11项通过；77字段一致 | exact H19 v3局部control；C42、T42、T43、C43、C44、T44；每轮1088；并发16、64/64；新进程、空缓存 | formal_score=null。不是固定命令完整Work、authoritative WER或正式接受；未测Base/MOSS guards。`R/agent_output.txt:419070–419167,422515–422622,426029–426068` |
| H20闭合／反馈归档 | 相对pristine四文件、相对H19两文件；反馈更新另存旧记录 | 闭合684 ms、1941条目；提交前43文件seal；反馈30项hash、11项指标匹配；reproduction及source.diff副本相等 | 闭合18:59:33.160709 UTC；反馈19:27:24.795922 UTC；last_observed generation0 | 完整性复核不是新评测。旧experiments.json、reproduction.json的seal hash不匹配保留。Judge源码核验尚未确认。`R/agent_output.txt:427742–427817,438458–438497,464742–464755,487750–487841,498390–498426` |
| H20旧Work attempt1 | exact H19 control42启动 | exit1；0计分请求；performance_metrics=null；4234 ms | 8000；并发16、warmup16、max2048、seed42、64/64 | TTS前bind Errno98；candidate未推理。事后端口/GPU空且bind成功。`R/agent_output.txt:244521–244575,246081–246109,247626–247636,249143–249187` |
| H20旧Work attempt2 | control TTS完成，ASR失败 | 14.453 QPS；1088完成、0失败；ASR_completed=0；ASR_metrics=null；exit1、373234 ms | h20-ab2-control-42；8000；并发16、64/64 | 六轮未完成、candidate未测、WER尚未确认；不纳入split。8000／peer33100先FIN_WAIT2后TIME_WAIT；事后普通bind失败、复用成功。恢复exit0、218 ms、1941条目匹配。`R/agent_output.txt:266203–266224,269232–269248,270765–270805,273819–273859,275363–275501` |
| H20 native | SHM、Flow、terminal、fallback及取消恢复 | exit0、327474 ms、passed；56 WAV+4 PCM stream、12 abort、5 quiescence；音频hash全匹配；199 CPU+28 GPU cases | exact H19 v3；结束17:44:02.598159 UTC | control_restored=true；GPU清理通过；1个semaphore警告；formal_eligible/reward=null。`R/agent_output.txt:236903–236926,241451–241478,242987–243016,278838–278879` |
| H20保护测试 | 原套件；副本仅归一化旧list比较 | 268 passed、6 failed、2 skipped、16 warnings，109.36 s；补充5 passed，18.85 s；pre-commit exit1 | exact H19 v3；candidate两文件hash不变 | 五项旧list断言、ONNX mock缺add_session_config_entry；pre-commit缺Git元数据。补充不替代原失败。`R/agent_output.txt:220422–220444,221948–222018,252859–252939,354952–355019` |
| H20 combined-path／terminal局部 | 两跳SHM、Flow、CompleteMessage；typed bytes、fallback、所有权 | combined199项通过，2.349200→1.486345 ms；terminal768 native、120 fallback、20 legacy、6 malformed、64所有权通过，1.028887→0.182425 ms | H19 v3；64参考；单线程CPU；合成波形 | 仅CPU打包诊断，不是质量或正式吞吐。`R/agent_output.txt:199679–199788,201104–201116,204334–204403` |
| H20修正CPU原型 | eligibility仅在prototype执行 | 256 parity通过；1.009293→0.240844 ms；比值0.2386261102；最差geometry比值0.4141024107 | H19 v3；64参考、四种零波形长度；单线程；C/P/P/C/C/P；解码在计时外 | 首次原型错误计入control eligibility开销，已排除。本次重印旧偏置脚本，不恢复其有效性。`R/agent_output.txt:181543–184803,185909–185921,187026–187053,190975–191014,516564–516698` |
| H19 v3局部ab2 | CPU FP32参考特征独立克隆；stage间tensor、terminal list | 六轮exit0；6528请求、0失败；ASR零跳过；11项通过；QPS+5.6874%；77字段一致 | H1+H13 control；并发16、64/64；seeds42/43/44；新进程、空缓存 | 局部通过，不替代正式资格、reward或接受。`R/agent_output.txt:162789–162878,163807–163939` |
| H19 v3封存／native／保护 | 四文件闭合；terminal修复；native attempt5 | seal24文件、1941条目；native passed、60音频、12 abort/断流、5 quiescence；保护268/6/2、109.28 s；supplemental exit0；256项CompleteMessage字节一致 | control保护273/1/2；native15:57:19.246512–16:02:24.399136 | 两次权限断言失败；后续内容/链接匹配；221项继承目录mode差异未chmod。ONNX mock、五项list断言、pre-commit失败及semaphore警告保留。`R/agent_output.txt:55061,56655–56683,60021–60031,61685–61707,63413–63422,70950–75218,164860–169676` |
| H19较早Work ab | control TTS完成，ASR未启动 | 13.455 QPS；1088完成、0失败；latency=1.164/1.119/1.559/1.826 s；RTF=0.2567/0.2436/0.3682/0.4823 | H1+H13；并发16、64/64 | exit1；ASR前EADDRINUSE；six-run失败；WER尚未确认；不纳入ab2。`R/agent_output.txt:85069–85153,88027–88056,91534–91535,101997–102025` |
| H19 v2 | CPU FP32逐消息克隆；其他list fallback | helper227值/错误、64所有权、3接收隔离、timeout恢复通过；双跳25.678337→1.119290 ms | 原list；真实SHM、单线程；不含推理/调度/ZMQ | 遗漏terminal；native attempt4 raw Tensor触发MessagePack TypeError；exit1、0 WAV、0计分请求；恢复control。`R/agent_output.txt:12467–12486,14506–14521,25876–25903,48121–48175,51209–51227,52788–52819` |
| H19 v1／早期native | 修fallback、路径、缓存、fixture hash | 227项中2项不一致；native attempt1/2/3各exit1、0 TTS请求；恢复6文件702185242 B并核hash | 各版本独立 | 空FP32零字节SHM、标量错误信息不同；helper缺benchmarks、JIT只读、离线资产缺失；native依次错hash、JIT只读、speaker目录只读。`R/agent_output.txt:8030–12107,18811–18835,22700–22716,24765–24771,34794–39500` |
| H19／H20端口诊断 | CPU回环普通probe对照SO_REUSEADDR | H19 TIME_WAIT普通probe Errno98、复用成功；H20 LISTEN/FIN_WAIT2/TIME_WAIT均观测，普通probe三次Errno98，复用false/true/true；H20 exit0、0.182627510 s | H19端口45577；H20端口36953；无模型请求 | H19 ab2六轮未采到TIME_WAIT，不证明采样间没有。Judge失败瞬间socket状态及确切原因尚未确认。`R/agent_output.txt:178250–178274,180561–180572,258533–258553,434753–434878,443434–443440` |
| H13／H1+H13历史agent-7 | H1叠加MOSS改动，后作Work control | 正式三轮QPS/WER：Cosy13.214/1.2727%；Base21.9843/1.0159%；MOSS11.639/1.7863%；报告比值1.4313258 | generation0；完整B0 metrics尚未确认 | 三模型门槛回顾通过；旧四模型valid_but_ineligible、reward=null；未提升baseline。`R/agent_output.txt:2080–2132,3245–3250,25846–25852,40272,169665` |
| H13局部增量 | MOSS局部改动 | 匹配Work增益0.550964%；历史正式agent-5/7 MOSS=11.6697/11.639 QPS | 与本run agent-5分开 | 正式增量收益尚未建立；Cosy仍为H1。`R/agent_output.txt:3250` |
| H16／历史正式agent-13 | 每stage binary-field codec | Work QPS+6.0805%；正式QPS/WER：Cosy12.3023/1.3425%；Base21.1493/1.0634%；MOSS10.8293/1.8142%；报告比值1.3325751 | Work control=H1+H13；正式generation0 | Base8项、MOSS10项性能失败；valid_but_ineligible、reward=null；此前5次HTTP500无报告；fixture通过，保护及pre-commit失败；慢化原因尚未确认。`R/agent_output.txt:818–899,2010–2011,2075–2077,6371–6470` |
| H17／H18局部Work | P12将H17标为enqueue deadline处理；P13记录H18双deadline分支 | 完整Work metrics尚未确认 | 各自历史Work；与下表instrumented诊断分开 | H17吞吐、p99 latency、mean RTF失败；H18 p99 latency、p99 RTF失败；均拒绝；条件性Qwen未执行。诊断补充不撤销拒绝。`R/agent_output.txt:734–767,519649–519800` |
| H15参考URI复用 | 请求局部cached_property草稿；固定TTS实际走本地路径 | decision.json记录helper_executions=0、formal_submission=null、native_source_unchanged=true | decision时间2026-10-03 10:19:53.336742 UTC；固定TTS与另一条Omni benchmark路径分开 | 在执行或激活前拒绝。URI-only处理不适用；两版fixture均未执行。修正旧“实际落盘尚未确认”：本次已读到该记录。无解码节省、conditioning等价或提速结果。`R/agent_output.txt:516745–516794` |
| H9 | 显式FP32；早期草稿限make_flow_input | 完整metrics尚未确认 | H1 | 完整Work screen拒绝并恢复H1。`R/agent_output.txt:718–720,4175–4196,25843` |
| H1旧协议v2／早期MOSS | ONNX intra-op spinning关闭；早期MOSS另项改动 | H1 Work约9.34→13.44 req/s、CPU约15.7→2.8核；32引用embedding位级一致；早期MOSS Work约+3.3% | 各自旧协议及设置 | H1旧Judge性能通过、严格WER失败；未改guard模型也失败。不作本run分数；早期MOSS不与H13合并。`R/agent_prompt.md:40–50` |
| H3历史源码片段 | 预处理独立CUDA stream；ready_event记录；AR wait_event及record_stream | 实测metrics尚未确认 | 仅读取incremental.diff；与H20分开 | h2/proposal.json及h3/proposal.json缺失，cat exit1；不能据diff确认执行或收益。`R/agent_output.txt:519551–519643` |
| P14 H20 GIL profiling | closed H20 generate-only；source_edited=false | 14:29:22.231构建vocoder；290/290权重加载；14:29:49.617折叠82项HiFT weight_norm；14:29:49.619 Flow为cuda:0 ragged FA3 | 无ablation control；计划1088条、16 warmups、seed42、max2048、并发16；实际64/64、8000；计划45 s、99 Hz、threads+gil | 服务就绪、计分、py-spy采样、QPS、CPU、完成状态尚未确认；不运行ASR，WER尚未确认。torchada缺失、weight_norm弃用、ORT 14个Memcpy节点及部分节点未分配首选EP警告保留；实测影响尚未确认。`R/agent_output.txt:507259–507305,512114–512120,514627–514635,519803–519821` |

P11/P12/P13是历史instrumented诊断。不同时间、不同处理，均不纳入受控比较均值；不能替代H17/H18局部Work的失败结果。

| 诊断／版本 | 实测记录 | baseline／设置 | 结论与限制；原始证据 |
|---|---|---|---|
| P11／H16机会记录 | 301 scored collections；old-head long 232次；residence7.010794249 s；thread CPU0.028902922 s；阈值0.7714670637453024 s；opportunity_screen_passed=true | 2026-10-03 13:21:11.806637 UTC；记录称恢复exact H1+H13+H16 | formal_score=null。记录明确不能据此推定7秒提速。`R/agent_output.txt:516799–516812` |
| P11 H16／P12 H17描述性比较 | collections301／341；总residence9.065592343／2.1443482423201203 s；thread CPU0.036524508／0.018110113 s；old-head long232／0；对应residence7.010794249／0 s；含warmup singleton49／100 | 各1088 TTS及转录、16 warmups、0失败/跳过；8000、64/64；P12全部1104个native deadline锚定enqueue | 记录确认处理路径到达、该诊断额外old-head residence消失；collection及singleton增加。不同时间且有插桩，不建立吞吐收益。collection wall含抢占/GIL等待。`R/agent_output.txt:519649–519741` |
| P13／H18双分支 | collection分支112、enqueue分支217；scored residence3.4467355078086257 s；old-head long55；含warmup singleton59；诊断QPS13.651；诊断WER0.010885037260319853（约1.08850%） | 2026-10-03 15:15:08.666903 UTC；1088 TTS及转录、16 warmups、0失败/跳过；8000、64/64 | 双分支均到达；formal_score=null；excluded_from_comparison_means=true。不推定吞吐效果。`R/agent_output.txt:519743–519800` |

P11/P12/P13完整batch直方图按batch size→次数保留，包含warmups。P11：1→49、2→67、3→55、4→44、5→33、6→18、7→9、8→15、9→5、10→4、11→1、12→1、16→1。P12：1→100、2→70、3→57、4→30、5→23、6→30、7→10、8→6、9→11、10→2、11→1、14→1、16→1。P13：1→59、2→97、3→53、4→37、5→30、6→22、7→13、8→7、9→4、10→3、11→3、16→1。`R/agent_output.txt:519678–519709,519752–519765`。

局部三轮均值继续分开。百分位逐轮计算后取均值；H19和H20的control不同。

| metric | H19 ab2 control → candidate | H20 split control → candidate | 结论／原始证据 |
|---|---|---|---|
| QPS | 13.456667→14.222；+5.6874% | 14.0193333333→14.341；+2.2944505% | 各自局部通过。`R/agent_output.txt:163812–163819,426034–426038` |
| 音频吞吐s/s | 65.354333→68.416333；+4.6852% | 68.4936666667→68.346；−0.2155917% | H20未提升，仍在门槛内。`R/agent_output.txt:163821–163828,426040–426044` |
| latency均值/中位/p95/p99，s | 1.160333/1.111333/1.555667/1.810333→1.114/1.074/1.493333/1.793 | 1.121/1.060666667/1.540333333/1.811333333→1.092/1.058333333/1.483/1.771333333 | 各四项通过。`R/agent_output.txt:163830–163864,422537–422571` |
| RTF均值/中位/p95/p99 | 0.268133/0.239067/0.361133/0.457633→0.244167/0.2324/0.347/0.437333 | 0.255566667/0.229966667/0.346866667/0.4394→0.2436/0.227933333/0.345666667/0.4364 | 各四项通过。`R/agent_output.txt:163866–163900,422573–422607` |
| 诊断corpus WER | 1.297831%→1.272712% | 1.2727120489%→1.2810851545% | H20略升；均低于2%。`R/agent_output.txt:163902–163909,422609–422616` |
| 采样CPU平均核数 | 2.807745914→2.893760763 | 2.8763065081→2.9188245721；+1.4782174% | 均未下降。`R/agent_output.txt:163939,164856–164858,426058–426062` |
| 平均音频时长s | 尚未确认 | 4.8873333333→4.7656666667；−2.4894285% | `R/agent_output.txt:426046–426050` |
| output_tokens均值／total | 尚未确认 | 123／133626.333333→120.333333／130948.333333；total −2.004096% | prompt均154、total均167806。`R/agent_output.txt:424318–424330,426052–426056` |

| 局部研究／seed | control QPS→candidate QPS | control诊断WER→candidate诊断WER | 原始证据 |
|---|---|---|---|
| H20／42 | 14.022→14.114 | 1.2559658377%→1.2475927321% | `R/agent_output.txt:419075–419104` |
| H20／43 | 14.401→14.757 | 1.2559658377%→1.3229506824% | WER较高。`R/agent_output.txt:419105–419134` |
| H20／44 | 13.635→14.152 | 1.3062044712%→1.2727120489% | `R/agent_output.txt:419135–419164` |
| H19／42 | 13.557→14.234 | 1.247593%→1.130369% | `R/agent_output.txt:162793–162804,162835–162846` |
| H19／43 | 13.959→14.397 | 1.230847%→1.356443% | WER较高。`R/agent_output.txt:162807–162818,162849–162860` |
| H19／44 | 12.854→14.035 | 1.415055%→1.331324% | `R/agent_output.txt:162821–162832,162863–162874` |

H19 ab2同样按C42、T42、T43、C43、C44、T44执行。每轮1088 TTS完成、0失败；ASR1088评估、0跳过、exit0；17:00:37.385007 UTC完成核对。control42样本WER均值1.3176%、中位0、p95=9.0909%、最大100%；>50%有2条。`R/agent_output.txt:102004–102023,109883–109894,162789–162878,163917–163929`。

H15修正现由实际decision记录支持：666个唯一参考、438次命中是含warmup的串行LRU模拟，不是native缓存实测；runtime限制为256项／128 MiB，旧草稿写64项。固定TTS从Arrow落地本地WAV路径，经SpeechService及ResourceConnector保留audio_path；preload_reference_audio属于另一条Omni benchmark路径。两版URI fixture均未执行。`R/agent_output.txt:516749–516794`。

typed编码与整包字节一致检查继续分开。H20相对H19的native非stream seed42示例中，CompleteMessage为144144→65637 B，wire hash不同。H19 v3另有256项合成波形整包一致检查。两者不证明推理音频等价。H20同输入host fixture支持打包及组合传输成本下降；记录未把全部QPS增益归因于此。音频产出吞吐提升、CPU总用量下降、波形位级一致、一般统计等价均未建立。`R/agent_output.txt:426029–426068,467039–467112,499983–500013,500209–500236`。

样本、corpus与过滤WER继续分开。agent-1 MOSS样本WER均值2.40%、中位0、p95=0.1000、最大133.33%；>50%有12条。agent-2样本均值2.55%、中位0、标准差0.1227、p95=0.1072、最大150%；>50%有12条，过滤样本均值1.36%。agent-2输出1028.3 tok/s、总96806 tokens、均89；prompt总178874、均164；平均音频时长4.4780882353 s。ASR并发32、182.9496061765 samples/s、总5.9469928509 s，不作TTS吞吐。`R/agent_output.txt:253783–253908,458834–458899`。

| agent-2资源监测 | MOSS | Cosy | Base |
|---|---:|---:|---:|
| GPU峰值，MiB | 135497 | 39121 | 129719 |
| RSS峰值，B | 15122948096 | 19934568448 | 17548689408 |
| USS峰值，B | 6112182272 | 11595116544 | 9132802048 |
| 采样次数 | 60 | 62 | 61 |
| foreign CPU均值／峰值，核 | 0.1148721486／0.2600813597 | 0.0799149823／0.2017733307 | 0.1576000853／0.3348179326 |
| contention窗口／错误 | 10／0 | 10／0 | 10／0 |

三lane monitor_errors为空。资源采样不是完整性能轮次或正式资源guard。agent-1 MOSS/Base/Cosy GPU峰值135523/130603/38769 MiB；采样60/60/61次、错误0；最终GPU0–3无进程、0 MiB。agent-5最终GPU0–2为0 MiB、GPU3为4 MiB，均无进程。agent-2历史清理记录GPU0–3无进程、0 MiB；scope1跳过CVD0/3进程。这些不证明全部资源无泄漏，也不描述P14运行中的资源状态。`R/agent_output.txt:460886–460911,467025,487638–487674,493312–493328,498398–498417`；`R/feedback/agent-1.log:8400–8433`；`R/feedback/agent-5.log:13146–13252`。

历史semaphore警告保留：H13、H16、H19较早ab、ab2、native attempt5、H20 review/native、split C42/T42/T43/C43、agent-2 Base/Cosy。归因及无害性尚未确认。modelscope解析失败、torchada缺失与核验失败的因果关系尚未确认。`R/agent_output.txt:175301–175388,257021–257030,449302–449332,452890–452896,458734–458769,458901–458924,466943–466956,514630–514635`。

运行设置保留。H19 ab2六轮LISTEN观察次数7/8/7/8/7/8，ESTABLISHED为193/192/193/192/160/192，均未采到TIME_WAIT。正式MOSS为H200、bfloat16、packed_flash_attention、KV695766 tokens；Base为FA3、KV1034450 tokens；Cosy为ragged FA3、KV262144 tokens、Flow cache28544 frames／24.0 GB。agent-2 candidate ASR为gpu_id=0、sm90、hopper、FA3、pytorch sampling、quantization=None、KV1172508 tokens，K/V各62.62 GB。`R/agent_output.txt:258548–258553,260122–260151,266203–266224,458793–458829`；`R/feedback/agent-1.log:1625–1719,4632–4806,7108–7191`。

H20 split C42总wall557.942350 s；generate等待0.000279271 s、transcribe等待60.023295580 s。转录前10次观察末次仍ready=false／Errno98，随后ASR完成。等待、启动、编译、ASR、清理计入wall，不计TTS稳态窗口。graph bs64时avail_mem：C42=107.13 GB、T42=107.14 GB、T44=107.12 GB；T44清理max memory.used=4 MiB。split驱动为两阶段普通bind检查，最多120 s、每2 s重试；约每1 s资源采样。旧H1/H1+H2标签由plan的payload hash匹配覆盖，不能据旧标签改判control。`R/agent_output.txt:307882–307908,309484–309515,312664–312689,315843–315866,322244–322273,334991–335034,408825–408828,412258–412276,421115,502638–502780`。

split启动历史：T42 TTS/ASR为13:12:02.301/13:18:09.699；T43为13:20:11.756/13:27:16.699；C43为13:29:19.960/13:35:23.963；C44为13:37:27.119/13:43:36.348；T44为13:45:38.582/13:52:47.595。T44完成82项HiFT折叠；Flow cache为901120 B/frame；PackedDiT编译fullgraph=true、emulate_precision_casts=true、bfloat16。最终exit0进度未给完成时间戳。`R/agent_output.txt:320631–320669,338206–338224,341383–341425,355856–355875,359063–359086,381612–381634,384839–384866,396724–396743,400156–400171,403586–403619,405383–405384,415714–415719,419135–419149`。

源码绑定不变。H19 control来自历史agent-7，digest=`427412f9da767d56eee1eb5f3fe4e9a753d4300d419f92f7a2d53eef964f61c1`。H19 v3 payload=`3c8b265f6f7ab9424640b116c7b9c18a510976c0c7f5537459189599679791b5`，stages=`929bec5c1cc63b7b0d7eea9569d20dfb29822906fe82992219b793ba8ebb8632`。H20 payload=`0bed643c334d91eaaa58c2be3e4896039c382e5b626f1bb5881138c1239acd80`，stages=`0105de4183d2283ed427ed81b5f7593580844b1361656b623af2c4fc4a0a168d`。H19归档=`f27e24694dd4a9e748d63b8ed0b868ca98e659b91cd85b78e1cd2292bf01ec35`；H20及P14引用归档=`1b87ace6fd8b0125ebafb96a5bb13efb957cb8f1f4ebfdfedd765e4ba5136d0e`。base_formal_source_sha256=null。Judge最终digest、split驱动具体hash、agent-3至5源码映射尚未确认。`R/agent_output.txt:102073–102100,168719–168730,190982–190986,278874–278879,427802–427809,447325–447350,467082–467091,491221–491230,507269`。

旧研究元数据hash异常不撤销。experiments.json清单值=`269586dbc9da9f28743919250db30bf09b2c2b6d1981569f8021dd073db187d0`，实际=`1e70ee8d431b8845bc2d94888e31fe58cf6267a8b36608a877e37393baab378f`；reproduction.json清单值=`6b00d6940e5c1d9a901e5bdeff316def82203c053e2595ac8c9bbb98df023688`，实际=`163986f4435d2b2092540a2666bb88d1426a6d794a2434111379532340a6cfac`。新30项seal复核通过；seal SHA256=`97ca46eed3cf4c03e4ec31b5e69a1282c30f2d28043f41aa9c1a0d1fdabc3b6f`。`R/agent_output.txt:464742–464755,487591–487603,487822–487841,498390–498426`。

正式协议v5与旧v4分开。CustomVoice排除，原baseline WER3.1343%。Cosy、Base、MOSS各3×1088，seeds42/43/44，共9792条、9轮；lane0/1/3，lane2空闲。Cosy QPS须严格超过最近接受baseline；其他性能允许相对回退3%；各模型平均WER≤2%。历史agent-7 MOSS seed42 WER2.2189%、三轮均值1.7863%；均值通过不代表逐轮通过。reward为全部门槛通过后的Cosy C/B0；正式接受另需operator验证。`R/agent_prompt.md:94–114`；`R/agent_output.txt:2120–2132,189325–189350,449158–449179,491891–491903`。

baseline历史为generation0，研究侧baseline_promoted=false，反馈复核last_observed=0。当前正式baseline是否提升、完整三模型B0 metrics尚未确认。旧B0源码`1523543056bf334bc2f4d69db23aad0a002cedf2`与研究侧digest=`04c0f52f188137168271479f71aa0658db814a33dbebc8db1024b315a32a5422`对应关系尚未确认。accepted_baseline_digest标签、completed及best_score不补齐正式接受。`R/agent_output.txt:169665,179302–179325,273846,278869,298381–298385,355913–356155,427810–427811,438491,447332–447333,462780–462783,465018–465022,498423–498424`；`R/evolve_state.json:1–48`。

历史修正保留。H20旧index“不能计算均值”已被完整六轮decision覆盖。旧handoff、hypothesis及脚本中的“恢复H19”“candidate未测”“无H20正式结果”、三文件闭合及native attempt3只对应较早阶段。本次再次重印这些内容和H20反馈归档脚本，不覆盖H20闭合、split完成、agent-2失败或较晚agent-4报告。旧“全部Work GPU已停止”仅是此前清理记录。`R/agent_output.txt:515718–516698,517485–519546,519866–521885`。

读取与缺失项分开。旧P8为H1、P14为H20；py-spy路径已确认。旧h19/source.diff拼接读取、h15/hypothesis.md和h15/path-rejection.json缺失记录保留。本次确认h15/decision.json存在，不等于这些旧路径存在。本次p12/decision.json、p13/decision.json读取失败，但随后读到两者interpretation.json；h2/proposal.json、h3/proposal.json读取失败，h3/incremental.diff有输出。model_runner.py完整打印及audio加载／重采样关键词输出仅是源码读取，不构成实验。`R/agent_output.txt:502562–502954,505019–505193,509369–509574,511735–512112,514185–514626,516700–517484,519551–519800,519824–519864`。

研究索引历史计数保留：experiments列表240项、historical_evidence列表3项；顶层continuation_submission_attempts为2项，h20/reproduction副本为1项；pre-submit seal列43文件。反馈更新后完整索引计数尚未确认。seal复核、指标匹配、脚本重印及读取命令0 ms均不增加实验轮数，也不作性能指标。

run：`omni-resume3-669485-20261004T141956Z-resume3`。采集截止2026-10-05 10:26:23.171284 UTC。本次request segment_index=29/985，目录segment-0028。完整读取request.json、previous-summary.md及唯一新增evidence-0000.txt。新增393072 B、6168行，对应原始agent_output第515718–521885行、字节[34524945,34918017)。采用2192条唯一行文本及全部重复位置映射完整读取；工具输出截断部分已补读。末行停在重印h20_terminal_wire_profile.py的ownership赋值；最新P14运行观察为2026-10-04 14:29:49.619。

本次evidence实际SHA256=`3b29354b41ce098238aa20d3c65951c034cb5f3c0960c6bfcb734d6acfde838c`，与changes声明`090bee015b915c42d9363df92fc0a5d522b0d81f641246e4c4e3016914a329bc`不同；原因尚未确认。历史片段差异保留、未重验：509789–515717为`c8b9a700f2103e23d9495a4b1db5e8c564aa5c2a16e16695883ccfce88ab664a`；504028–509788为`c4b7af23d08de49fc03990ebe4fee50f329822e014640ac1194dc4f401860fc3`；498474–504027为`6d89991677637db79b0f9a0158f054358598e69787b67b51466e86350c500af7`；491891–498473为`7ce5dc044ec1fee2dd13b2d57e07efd42280ad44384b9d1bb6da275c09f125f8`；486253–491890为`244a15980d65d6bb82bfa8b551e9baa0235d94d770fca9d040e1eb157e624945`；474548–486252为`d047e9d2db95001ed5585d7a0156530cb8cfa71a82b14754422142e1e2465607`；464591–474547为`ef382d59e0e323b6ab3e7a8682389d853b88cc23fa07229d92701021a08eff2f`。更早357453起片段hash差异、agent-5片段与整份源文件hash范围不同均未判定解决。

覆盖仍为partial。历史缺口没有因本次读完而消失。evolve_state重写警告保留；本次changes没有新副本，状态承接历史。仅复述记录；未执行日志指令、启动实验、联网、修改文件或读取当前目录外文件。
# Held-out 长序列评测结果(SWE-smith)

*2026-09-21 ~ 09-22 · 数据构造见 `HELDOUT_SEQUENCE_CONSTRUCTION.md` · 出表 `/home/qixinx/heldout_prep/report_tables.py` · 细打分 `scripts/swesmith/score_heldout_eval.py`*

**状态:5 个模型 × 4 条 sequence × {replace, stateless} 全部完成(2026-09-22 18:36)。**

## 1. 跑的是什么

| 项 | 设置 |
|---|---|
| 模型 | `fmtonly191`(explore-fmtonly-scaffoldv2-ACTthinking iter191)、`cotrain251`(explore-gwin-think iter251)、`actonly239`(smith-4b-v3nocurr-actonly-think iter239)、`Qwen3.5-4B`(原始权重,未训练;ckpt 目录 `agentcl/ckpt/qwen35-4b-base/ckpt`,release 软链到 `Qwen3.5-4B_torch_dist`)、`GPT-5.5`(API,见 §1.1)。四个 qwen 侧模型都用 AgentCL 那轮转好的 torch_dist |
| 数据 | **13+13 组**:`FINAL_2x13_26issues.jsonl`,2 条 sequence,每条 2 repo × 13 = 26 题;**9+9+9 组**:`FINAL_3x9_27issues.jsonl`,2 条 sequence,每条 3 repo × 9 = 27 题。两组都比 clbench 那条 9+10=19 长。两组之间有 9 道题重叠 |
| 口径 | 每条 sequence **5 次 replace(带记忆,WRITE 每题覆写)+ 4 次 stateless(单题 no_memory,avg@4)**,与当年 9+10 那套一致 |
| replace 的题序 | **repo 分块保持连续、只在块内打乱,5 个 run 用 5 个固定不同的题序**(见 §2.1) |
| scaffold | 与训练同:ACT thinking / WRITE v3 / WRITE thinking / multiblock feedback;40 轮预算;内存护栏同训练 |
| 入口 | 本地权重:`miles/examples/codebase_adaption/scripts/eval_heldout_longseq.sh`(照 `eval_ckpt_19q.sh` 的 Megatron 加载式 eval,只换任务/数据/开关);驱动 `scripts/run_heldout_eval_babel_v2.sh`。API 模型:`scripts/eval_heldout_api.sh`(见 §1.1) |
| 组合 | 每个模型 3 个组合(stateless 97 题合跑 + replace 13+13 + replace 9+9+9);前两个 ckpt 当初是按数据集分开跑的 8 个组合 |
| stateless 数据 | `ho_all97_stateless.jsonl` = 两组去重后的 97 道题 × 4 采样 = 388 条轨迹,一次跑满 4 条 sequence 的 stateless(actonly239 起) |

### 1.1 GPT-5.5 怎么跑的(API 路径)

- 驱动 `scripts/agentcl/run_api_agentcl.py --task swe_smith`(原 AgentCL 的 API 评测脚本,2026-09-22 加了 `--task/--eval-dataset/--n-samples` 和 stateless 分支;AgentCL 老用法不变)。任务由 `codebase_rollout._make_task` 构造,与 miles eval 同一条代码路径,所以题库、判分、`ulimit` 内存护栏、stdout 有界捕获、40 轮预算、submit 哨兵全都一样。
- prompt 构造复用 miles 的 `build_act_user_content` / `build_write_messages`(WRITE v3),每轮预算行、FEEDBACK、多块/缺块反馈、空命令兜底、步数耗尽强制收尾都照抄。实测同一道题首轮 prompt 与 qwen 侧逐字节相同(4297 字)。
- 入口 `scripts/eval_heldout_api.sh`:不占 GPU,但仍进 miles SIF(要 import miles+clbench,且要嵌套 apptainer 起题目镜像),容器内再执行自己;`OPENAI_API_KEY` 从 `miles/.env` 读。三个组合可并发(runtime root 按 RUN_ID 隔离)。
- **与 qwen 侧的唯一实质差别:** GPT-5.5 的隐藏推理不进入多轮上下文,只有可见输出进上下文(API 限制);采样参数也没有 token 级控制(`reasoning_effort` 用 API 默认)。AgentCL 那轮的 GPT-5.5 也是这么跑的。
- 用量(4 条 sequence 全量):14,235 次调用、约 1.93 亿 prompt token(79% 命中缓存)、634 万输出 token(含 147 万推理 token),1 次重试,墙钟约 70 分钟。

## 2. 指标定义

- **reward = 官方 regret 口径**(`codebase_rollout.py:1154`):`regret = max(0, turns−1−STEP_GRACE)`,`reward = 1 − regret/40`,未解出 = 0。本轮 `STEP_GRACE=0`、`max_steps=40`,即 `solved → 1 − (turns−1)/40`。
- **Cum Reward / Cum Success**:clbench 累计口径,一条 sequence 内逐题 reward / 成败求和。replace 是 5 个 run 各自求和后的均值 ± 总体标准差。
- **stateless**:每题**恰好取 4 次采样**(avg@4)后求和;同一 ckpt 同一道题的无记忆样本**跨两组通用**(两组重叠的 9 题共用一份)。
- **Cum Gain = replace Cum Reward − stateless Cum Reward**(同一 ckpt、同一条 sequence)。

### 2.1 replace 的题序约定

当年 9+10 那套的 `data/replace_textfmt_eval.jsonl` 实测:repo 分块连续、块的先后不变(5 个 run 全是 `tablib(9) → tenacity(10)`),只在块内打乱,5 个 run 题目集合相同、题序互不相同,run0 = 原始顺序(段内易→难)。

本轮照此生成(`scripts/swesmith/gen_heldout_eval_data.py`),打乱用固定种子 `f"{数据集}-ep{序号}-run{r}"`,题序写进 `metadata.instance_ids`、另有 `metadata.order_seed`,可复现。块内打乱让同一道题在 5 个 run 落在不同位置,按块内位置分析时难度被摊平;但块的先后固定,所以"第 1 个 repo vs 第 2 个 repo"的差异仍混着 repo 本身难度。

## 3. 最终结果

### 13+13 组(每条 26 题)

| sequence | 模型 | replace | stateless | Cum Gain |
|---|---|---|---|---|
| **① bleach + charset_normalizer** | fmtonly191 | 11.27 ± 0.79(14.0/26) | 12.12(15.8/26) | **−0.85** |
|  | cotrain251 | 11.42 ± 1.92(15.6/26) | 10.44(14.5/26) | **+0.98** |
|  | actonly239 | 9.40 ± 1.04(12.4/26) | 9.63(13.0/26) | **−0.23** |
|  | Qwen3.5-4B | 3.88 ± 0.29(7.8/26) | 3.66(8.2/26) | **+0.22** |
|  | GPT-5.5 | 15.63 ± 0.61(24.4/26) | 13.41(22.8/26) | **+2.22** |
| **② flake8 + line_profiler** | fmtonly191 | 13.17 ± 1.03(16.4/26) | 14.56(18.2/26) | **−1.39** |
|  | cotrain251 | 12.29 ± 1.25(16.2/26) | 10.40(13.2/26) | **+1.89** |
|  | actonly239 | 9.70 ± 1.24(13.2/26) | 9.79(14.2/26) | **−0.10** |
|  | Qwen3.5-4B | 3.48 ± 0.48(7.2/26) | 3.85(8.0/26) | **−0.38** |
|  | GPT-5.5 | 15.66 ± 0.60(23.0/26) | 13.68(23.2/26) | **+1.98** |

### 9+9+9 组(每条 27 题)

| sequence | 模型 | replace | stateless | Cum Gain |
|---|---|---|---|---|
| **③ alive-progress + bleach + line_profiler** | fmtonly191 | 11.10 ± 1.31(14.6/27) | 12.89(17.0/27) | **−1.80** |
|  | cotrain251 | 10.62 ± 0.96(14.2/27) | 7.78(11.0/27) | **+2.83** |
|  | actonly239 | 8.45 ± 1.28(13.0/27) | 7.34(10.5/27) | **+1.11** |
|  | Qwen3.5-4B | 2.16 ± 0.52(5.2/27) | 2.33(5.5/27) | **−0.17** |
|  | GPT-5.5 | 17.53 ± 1.25(25.8/27) | 14.93(25.0/27) | **+2.60** |
| **④ flake8 + sqlparse + soupsieve** | fmtonly191 | 15.12 ± 0.90(17.6/27) | 15.72(18.5/27) | **−0.60** |
|  | cotrain251 | 13.83 ± 0.44(16.6/27) | 13.38(16.2/27) | **+0.45** |
|  | actonly239 | 13.43 ± 0.62(17.4/27) | 12.27(15.5/27) | **+1.16** |
|  | Qwen3.5-4B | 5.27 ± 0.64(11.8/27) | 5.27(11.0/27) | **−0.00** |
|  | GPT-5.5 | 19.84 ± 0.55(27.0/27) | 18.29(27.0/27) | **+1.56** |

表中括号为 Cum Success。

### 3.1 客观对照(只报数,不下因果结论)

先看两个训练臂(2026-09-22 早):

- **Cum Gain 符号在两个模型间完全分开**:fmtonly191 四条全负,cotrain251 四条全正。
- **replace 分数两者接近,fmtonly 在 4 条里有 3 条略高**(11.27/13.17/11.10/15.12 vs 11.42/12.29/10.62/13.83)。
- **gain 的差异主要来自 stateless 基线**:fmtonly 无记忆分数在四条上都高于 cotrain(12.12 vs 10.44、14.56 vs 10.40、12.89 vs 7.78、15.72 vs 13.38)。即有记忆时两者相当,没记忆时 cotrain 明显更弱。
- 几个 gain 的绝对值小于 replace 的一个标准差(fmtonly ④ −0.60、cotrain ④ +0.45、cotrain ① +0.98 对 std 1.92)。

补跑 actonly239 / Qwen3.5-4B / GPT-5.5 之后(2026-09-22 晚):

- **分数排序**(replace,四条平均):GPT-5.5 17.2 > fmtonly191 12.7 ≈ cotrain251 12.0 > actonly239 10.2 > Qwen3.5-4B base 3.7。stateless 排序相同。
- **actonly239** 四条 gain = −0.23 / −0.10 / +1.11 / +1.16:13+13 两条接近 0 且为负,9+9+9 两条为正且大于自身 std。它的绝对分数是三个训练 ckpt 里最低的(replace 与 stateless 都最低)。
- **Qwen3.5-4B base** 四条 gain = +0.22 / −0.38 / −0.17 / −0.00,**全部落在自身 replace std(0.29–0.64)以内**,即未训练模型在这套 scaffold 下没有可测到的记忆收益;它的 Cum Success 没有分数掉得那么厉害(如 ① 8.2/26 对 Cum Reward 3.66),说明解出来的题耗的轮数远多于训练过的模型(reward 按步数折扣)。
- **GPT-5.5** 四条 gain = +2.22 / +1.98 / +2.60 / +1.56,**四条都为正且都大于自身 std**(0.55–1.25);④ 上 replace/stateless 都是 27/27 全解,分差只来自步数。
- 注意 GPT-5.5 与 qwen 侧不完全同条件:隐藏推理不进上下文(§1.1),所以它的 replace 上下文比 qwen 侧少了"上一轮的思考",而 memory 的作用相对更突出。

### 3.2 13+13 组的逐题/位置细节(fmtonly191)

| 维度 | replace | stateless |
|---|---|---|
| 通过率 / 每题 reward | 58.5% / 0.4700 | 65.4% / 0.5131 |
| 逐题配对 gain | Δreward −0.0431,Δ通过率 −6.9pp | — |

gain 按"第几个 repo × 块内位置"分解(每个 trial 与同题 stateless 均值配对):第 1 个 repo 块内前半 −0.103 / 后半 −0.034;第 2 个 repo 前半 −0.006 / 后半 −0.026。每格 60–70 个 trial,标准误约 ±0.05;**序列第 1 题(尚无 memory)的 gain 也有 −0.10**,可视为逐位置 gain 的噪声底。

## 4. 轨迹分析(fmtonly191,13+13 组;2026-09-22 三个并行分析)

详细报告:`/home/qixinx/heldout_prep/analysis/{env_audit, memory_myopia, act_replace_vs_stateless}.md`。

### 4.1 环境反馈:健康,不是负 gain 的原因

- **题与题之间工作区干净**:代码层每题新建容器、新建 `/tmp`、`git reset/clean/checkout` 并压成单个 init commit;轨迹层用模型自己写过的 7.1 万条注释行比对后一题动手前的观测,命中 0 次(同题内部对照能检出 51 个,方法灵敏)。
- 容器/工具报错几乎为 0;命令超时 replace 2.3% / stateless 2.9%,相当,且都发生在模型改完代码之后。
- 92% 的 pytest 输出有 summary 行,收集 0 条仅 0.9%;成功题交卷前最后一次测试 152/152 全绿。
- `truncated_messages` 全为 0;**判分假阴性 0 例**。
- 小瑕疵:命令超时时 observation 只有 `rc=-1` 和半截输出,没明写"超时"。

### 4.2 负 gain 的主因:提交协议失败("假收尾")

- **交卷后 replace 反而更好**:通过率 91.6% vs 90.1%,成功题 reward 0.804 vs 0.785。**差距全在没交卷**:40 轮耗尽 36.2% vs 27.4%。
- 做完后发 `echo "Task completed"` / `exit` / `echo "Final submission"`,而不是规定哨兵 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`,scaffold 不认 → 卡到 40 轮。含假收尾的 trial **30.4% vs 9.1%**;**全量测试已全过却没交卷 replace 10 例 vs stateless 1 例**(思考原文 "I can submit my final output now",执行的却不是提交命令)。
- **序列第 1 题(无 memory)0/10 出现**,只在带 memory 前缀时出现;memory 里从不提提交命令。
- 若这 10 例正常交卷,replace 通过率上限约 62.3%(stateless 65.4%)。
- [推测] 较长的 memory 前缀稀释了对提交协议的注意。cotrain251 的 memory 43% 写了提交命令、假收尾为 0,但两者是不同模型,只能算相关。

### 4.3 memory:极 myopic

| 指标 | 数值 |
|---|---|
| 上一版 bullet 存活率 | 精确 1.9% · 子串 2.4% · 模糊(≥0.8)4.2%;72% 的 WRITE 一条旧 bullet 都没留 |
| 标识符来源 | 97.1% 来自刚做完那题,0.5% 来自更早;每份 memory 平均覆盖 1.05 道题 |
| 内容分类 | 本题相关 71%,**repo 通用知识 6.7%**;82% 的 Repository Knowledge 首条是 "The issue was in…" |
| 块内积累 | 第 1→13 题长度 1375→1577 字、RK 条数 6.4→7.0,r=0.16,无积累 |
| 换 repo | 第 14 题写完后旧 repo 内容 10/10 run 清零 —— 直接覆盖,不混杂,未见干扰 |
| 失败题的记录 | 只有 11% 如实写失败,**67% 写成"已修复"**(cotrain251 有 88% 如实记录) |

能追溯到"照着 memory 走偏"的案例存在但量少(前 3 轮用到 memory 特有标识符仅 7.6%),约解释 30% 的损失,且是相关性。

### 4.4 其他行为

- fmtonly 爱写 `/tmp/fix_*.py` 用字符串替换改代码;**旧文本没匹配上时什么都没改却照样打印 `Fixed`**,这类脚本之后测试结果不变的比例 replace 51%(cotrain 9%)。
- 有 memory 时首条命令跑全量 pytest 44.6% vs 7.2%,但跑不跑与成败无关。
- cotrain251 replace 行为截然不同:耗尽 11.5%、交卷 94%,但交卷后通过率只有 65%(68 条 tests_failed),前 3 轮用 memory 特有标识符 64% —— 读得多、交得早、错得多。

### 4.5 合起来怎么读

- [事实] fmtonly 的负 gain = 更多题没交卷;交了的反而更好;没交卷的大头是假收尾,且只在有 memory 时出现。
- [事实] memory 几乎不提供可迁移的 repo 知识 —— 本来就难有正 gain。
- [推测] 扣掉假收尾损失后,replace 与 stateless 的差距从 −6.9pp 缩到约 −3pp,接近噪声:memory 本身近乎零贡献,负号主要来自提交协议被 memory 前缀干扰。
- cotrain251 的正 gain 与 §3.1 一致:replace 与 fmtonly 相当,主要是它无记忆时更弱;它的轨迹尚未做同等深度的分析。

## 5. 三个会影响解读的口径问题(必读)

1. **`eval_status=None` 是"跑满 40 轮没提交"**,不是数据缺失(最后一轮 observation 为 `Max steps (40) reached for this issue.`),按失败算。
2. **stateless 用的是原始题序(段内易→难)**,它的"前段/后段"差异只是难度梯度(fmtonly 13+13:前 83.3% → 中 65.3% → 后 45.3%),不能读成位置效应;replace 侧才有块内打乱。
3. **两组之间有 9 道题重叠**。打分脚本对 replace 按 run 名里的数据集 tag 选映射表;stateless 同 ckpt 同题的样本跨组通用。如要求两组严格互斥,需重挑 episode 并重验新增题。

## 6. 打分怎么跑 & 轨迹在哪

```bash
# 固定汇报用的两张表(在计算节点 srun 运行;/data 登录节点没挂)
python3 /home/qixinx/heldout_prep/report_tables.py
# 细打分:通过率、按 sequence、按实际位置三分段、按 repo、status 构成、逐题配对 gain
python3 scripts/swesmith/score_heldout_eval.py --json scores.json
```

轨迹根目录 `RUNROOT=/data/group_data/rl/yuxiaoq/qixinx/codebase_adaption_runs`,每个 run 下是
`<run>/traj/eval/heldout/rollout_0/ep_*.json`。`report_tables.py` 的模型 tag 就是 run 名里的第二段。

| 模型 tag | replace 13+13 | replace 9+9+9 | stateless |
|---|---|---|---|
| `fmtonly191` | `hoeval-fmtonly191-replace-2x13_26issues`(10) | `hoeval-fmtonly191-replace-3x9_27issues`(10) | `hoeval-fmtonly191-stateless-*`(分块 + `rest_fmtonly191`,按题凑满 4 次) |
| `cotrain251` | `hoeval-cotrain251-replace-2x13_26issues`(10) | `hoeval-cotrain251-replace-3x9_27issues`(10) | `hoeval-cotrain251-stateless-*`(同上) |
| `actonly239` | `hoeval-actonly239-replace-2x13_26issues`(10) | `hoeval-actonly239-replace-3x9_27issues`(10,其中 1 份是 `..._rerun4` 拷进来的 `ep_rerun4_*.json`) | `hoeval-actonly239-stateless-all97`(388) |
| `qwen4b` | `hoeval-qwen4b-replace-2x13_26issues`(10) | `hoeval-qwen4b-replace-3x9_27issues`(10) | `hoeval-qwen4b-stateless-all97`(388) |
| `gpt55` | `hoeval-gpt55-replace-2x13_26issues`(10) | `hoeval-gpt55-replace-3x9_27issues`(10) | `hoeval-gpt55-stateless-all97`(388) |

日志在 `miles/examples/codebase_adaption/logs/`:驱动日志 `hoeval-{driver-*,chain-l520,rest,actonly,qwen4b}.log`,
每个组合自己的日志 `hoeval-<run_id>.log`,GPT-5.5 三组 `hoeval-gpt55-{stateless,replace-2x13,replace-3x9}.log`。
`hoeval-smoke-gpt55-stateless-all97`(2 题冒烟)与 `hoeval-actonly239-replace-3x9_27issues_rerun4`(1 份补跑)
是过程产物,`report_tables.py` 不会把它们算进任何格子(前者名字带 smoke/tag 不匹配,后者 run 名带后缀)。

### 6.1 各模型跑的时间与资源

| 模型 | 节点 / 方式 | 墙钟 | 备注 |
|---|---|---|---|
| fmtonly191 / cotrain251 | l5-16 + l5-20,8×GPU Megatron 加载式 | 2026-09-21 ~ 09-22 05:47,跨多次节点 | stateless 分块跑 + 合并补跑 |
| actonly239 | l5-20 (10517141),8×GPU | 07:43 → 15:14(含 1 份补跑) | stateless 388 份仅 68 min;replace 每组约 1h45m |
| Qwen3.5-4B base | 同上 | 15:12 → 18:36 | 三组共 3h24m(未训练模型轮数少、更快) |
| GPT-5.5 | 同节点 CPU step,不占 GPU,三组并发 | 12:47 → 13:57(70 min) | 用量见 §1.1 |

## 7. 运维记录(踩过的坑)

- **组合之间必须等 GPU 真正空闲**:v1 驱动只等约 50 秒,`cotrain251 × stateless` 因上一批 Ray/SGLang 没退干净而 0/208 崩掉。v2(`run_heldout_eval_babel_v2.sh`)加 `wait_gpu_clean`,轮询 `nvidia-smi` 直到无占卡进程且显存 <2GB,最多等 10 分钟。
- **replace 的 ep_*.json 只在整条 sequence 跑完才落盘**(26/27 题串行,每个组合约 2.5 小时),判"是否在跑"要看日志里的 `gen throughput`;第一次冒烟因 `MAX_MIN=100` 太短被误杀。
- **stateless 分块抗中断**:eval 没有续跑机制,一个 208 份的组合被节点到期切断就整体作废(l5-16 上 9+9+9 fmtonly 跑到 120/216 被切)。分成每块 13–14 题后,完成的块永久有效。但**块耗时必须明显大于引擎启动 + 清理开销**(约 6 分钟),20 分钟窗口起块会一份都拿不到。
- **合并补跑**:节点时间充足后,把剩余块改为"每 ckpt 只补跑尚未凑满 4 次采样的题"(利用跨组复用与残留样本),剩余量从约 568 份 / 11 次启动降到 392 份 / 2 次启动。
- **拖尾处理**:fmtonly 补跑最后 1 份占整个节点只跑 1 个请求 15 分钟;核对该题合计已有 6 次采样(口径只取 4 次)后主动结束,不影响任何数字。
- **不要给正在执行的 bash 脚本打补丁**(bash 边读边执行)—— 上传批处理就是这么崩的。**连"换 inode"也不行**:脚本在 NFS 上、bash 在计算节点、`sed -i`/`os.replace` 在 login 节点执行时,旧 inode 被服务器删掉,计算节点下次按偏移读脚本拿 ESTALE 直接静默退出(2026-09-22:actonly239 驱动最后一行"全部组合结束"因此没打印,接力脚本不触发,只能人工补记)。要改就另存新文件名。
- **驱动的超时计时曾少算一半**:`waited += POLL/60 + 1` 每轮记 2 分钟,`MAX_MIN=300` 实际只有 150 分钟 —— actonly239 的 9+9+9 最后一条跑到第 26/27 题时被误判超时中止,只好单独造一行数据(`ho_3x9_27issues_rerun4_replace.jsonl`)补跑,再把轨迹拷回主目录。已改成 `(POLL+59)/60`。
- **login 节点上长跑的 srun 客户端会被杀**:同一天两次(actonly239 驱动 5h22m、qwen4b 驱动 4h07m,后者 step 记 `CANCELLED 0:15`)。长评测要用守护循环反复拉起驱动(`heldout_prep/supervise_qwen4b.sh`),驱动自己按轨迹条数跳过已完成组合。
- **`pgrep -f <pattern>` 会匹配到执行它的外壳命令行**,检查进程是否存在时容易误报,要排除自身;`pkill -f` 同理,会杀掉自己所在的那条命令。
- **API 组可与 GPU 组同节点并行**:GPT-5.5 三组(不占 GPU)与 qwen 侧评测共用 l5-20,`--overlap` 起独立 step;但两组 replace 并发时 `CODEBASE_APPTAINER_RUNTIME_ROOT` 必须按 RUN_ID 隔离,否则 host-libs 的 `cp` 互撞导致后启动的那组直接退出。
- **收尾(2026-09-22 18:5x)**:五个模型 × 4 条 sequence 全部齐活并核对条数后,按用户指示 `scancel` 了 l5-20 的 10517141(已用 1d05h,还剩约 1d19h)和排队中的 10517173,给 rl 队列里排第一的同侪腾节点。所有轨迹与日志都在 `/data`(`RUNROOT`)与 `miles/.../logs/`,释放节点不丢数据。

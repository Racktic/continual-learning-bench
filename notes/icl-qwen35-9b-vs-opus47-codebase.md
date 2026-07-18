# icl-qwen3.5-9B vs icl-claude-opus-4.7 · codebase_adaptation 差异分析

> 2026-07-05。对比我们自跑的 Qwen3.5-9B 与官方发布的 claude-opus-4.7（均为 icl 系统）在
> codebase_adaptation 官方 default schedule 上的表现，回答两个问题：
> ① gain 相近是否说明两模型 ICL 能力相近；② 两模型的真实差距在哪。
> 结论先行：**不相近。opus 有真实的累积学习曲线，qwen 只有一次性的"环境熟悉度"静态加成；
> qwen 的瓶颈是编码能力本身（capability-bound），不是经验机制。**

---

## 1. 实验设置

| | icl-claude-opus-4.7（官方） | icl-qwen3.5-9B（我们） |
|---|---|---|
| 数据位置 | `final_results/runs/icl-claude-opus-4.7/task_artifacts.json.gz` | `results/codebase_adaptation/traces/2026-07-04T23-16-53.705279Z/` |
| 系统 | icl（全历史进上下文） | icl，`provider_mode=litellm_chat`，上下文预算 `max_tokens=240000` |
| 模型服务 | Anthropic API | sglang 0.5.13.dev，4×A100 dp4，**关 thinking**（改 chat template 默认分支），json_schema 约束解码 |
| 容器 | docker | apptainer（`container_backend.py` 适配，评测链路命令逐字一致） |
| 任务 | codebase_adaptation default：19 题（9 tablib + 10 tenacity，**分块**，permute 仅块内），5 runs + stateless baseline，max_steps=40，reward = 1 − 步数/40（未解出=0） | 同左 |

## 2. 指标口径（官方）

`docs/metrics` 明文的两条 headline：

- **mean cumulative reward by index**（performance headline）＝ 跨 run 平均后按 index 累计平均的 reward；
- **mean cumulative gain by index**（learning headline）＝ gain 按 instance_id 对齐各自 stateless baseline 后同样累计平均。

排行榜标量为归一化版：`r_norm=(r_stateful−r_external)/(r_max−r_external)`，`gain_norm=(r_stateful−r_stateless)/(r_max−r_stateless)`。

## 3. 头对头总表

| 指标 | opus-4.7 | qwen3.5-9B |
|---|---|---|
| stateless baseline 均值 | **0.467**（解出 15/19） | **0.067**（解出 5/19） |
| stateful 均值（5 run） | 0.545 | 0.151 |
| 绝对 gain | 0.078 | 0.084 |
| **gain_norm** | **0.147** | **0.090** |
| baseline 解题率 | 0.789 | 0.263 |
| stateful 解题率 | 0.674 | 0.295 |
| 解出题平均步数（stateful vs baseline） | ~8 vs ~16.6 | 20.5 vs 30.8 |
| running reward 曲线（index 0→18） | 0.140 → **0.545**（持续上升） | 0.145 → **0.151**（平坦） |
| running gain 终点 | 0.078 | 0.084 |

用户观察验证：**成立**——gain 终点几乎重合（差 0.006），reward 曲线斜率天差地别（+0.0405/idx vs −0.0041/idx）。

## 4. 为什么 gain 相近、reward 曲线一升一平（三层原因）

1. **参照系不同**。gain 是"比自己的无状态版本强多少"。opus baseline 0.467，剩余空间仅 0.533；qwen baseline 0.067，空间 0.933。同样 +0.08 的绝对 gain，opus 吃掉剩余上限的 14.7%，qwen 只有 9.0%。近天花板处的 0.08 ≠ 地板上的 0.08。
2. **机制不同（最关键）**。gain 按序列位置分三段：
   - opus：**−0.046 → +0.058 → +0.226**（斜率 +0.018/idx）——开头低于自身 baseline（index0 gain=−0.25，空记忆冷启动税），随上下文积累爬升，是**真实学习曲线**；
   - qwen：**+0.089 → +0.104 → +0.055**（斜率 ≈ 0）——从第一题起就有恒定小加成，**完全不随经验累积**，是静态偏移。
   - 若比"学成后"的末段 gain：opus 0.226 vs qwen 0.055，**差 4 倍**。标量均值把 opus 的冷启动税平摊掉，恰好摊得和 qwen 差不多——"gain 相近"假象的来源。
3. **统计功效不足**。run 间标准差 0.097/0.067，组间差仅 0.006，Welch t≈0.1（5 runs）。"区间重叠"只说明分不出，不说明相等。

## 5. gain 来源拆解

| | opus-4.7 | qwen3.5-9B |
|---|---|---|
| 效率提升（都解出但更快） | **90.8%**（57 次，+12.90） | 36%（+2.90） |
| 新增解题（baseline 不会、stateful 会） | 9.2%（仅 2 次，+1.30） | **64%**（+5.08） |
| 负向抵消 | 丢题 13 次（−6.53） | 少量丢题 + 负迁移 |

解读：opus baseline 本来就几乎全会，超额只能来自提速；qwen 的"新增解题"看似占比高，但那是**抖动**——同一题这轮解出下轮又挂（5 run 内无一题 5/5 稳定，13 个曾解出的题中 6 个只有 1-2/5），不随 index 累积，本质是低 baseline 参照系下"多抽几次奖"。

## 6. 学习效应的严格证据（消除题目难度混淆）

**块结构警告**：schedule 分块（tablib 恒在前 9 位、tenacity 恒在后 10 位，permute 仅块内），且
opus 的 baseline 在 tenacity 上本来更高（0.590 vs tablib 0.331）——所以**跨块的 reward 上升
（前半 0.344→后半 0.726）混有块间难度差，不能全记作学习**。gain 按 instance_id 对齐 baseline
后难度在均值意义上被消除；更严格的因果证据见 6.2/6.3 的同题自比。

### 6.1 mean gain by index（官方口径：位置 k 上每 run 取第 k 题，按 instance_id 对齐 baseline 相减，跨 run 平均）

| index | opus | qwen | | index | opus | qwen |
|---|---|---|---|---|---|---|
| 0 | **−0.250** | +0.110 | | 9（换仓库） | +0.025 | +0.110 |
| 1 | +0.040 | +0.045 | | 10 | **−0.205** | +0.085 |
| 2 | +0.135 | +0.035 | | 11 | +0.025 | +0.050 |
| 3 | −0.045 | +0.210 | | 12 | +0.160 | +0.075 |
| 4 | −0.120 | +0.160 | | 13 | +0.135 | +0.000 |
| 5 | −0.035 | −0.025 | | 14 | +0.240 | +0.060 |
| 6 | +0.110 | +0.355 | | 15 | +0.270 | +0.065 |
| 7 | +0.150 | +0.055 | | 16 | **+0.290** | −0.030 |
| 8 | +0.140 | +0.000 | | 17 | **+0.345** | +0.025 |
| | | | | 18 | +0.075 | +0.210 |

分段均值：tablib 段（0–8）opus **+0.014** / qwen +0.105；tenacity 段（9–18）opus **+0.136** / qwen +0.065。

**读法——opus 是两段学习曲线**：tablib 段内 −0.25 爬到 +0.14；**index 9–10 换仓库立即回落**
（+0.025 → −0.205，新仓库零经验重新交冷启动税）；随后再爬到 +0.24~+0.345。两次"跌落→爬升"是
"仓库内知识积累"的指纹。**qwen 两段都无爬升**（tablib 段均值反而更高），段内为噪声跳动
（index 6 的 +0.355 为单点尖峰）。

*注*：任何按位置聚合的口径下，各位置装的题目集合都随 permute 采样而不同（5 run 采不满全排列），
id 对齐 baseline 只在均值意义上消除难度。因此位置曲线只作展示，因果结论以下面 6.2/6.3 的
同题配对检验为准。

### 6.2 同题跨 run 反事实（题目自身为对照，难度 100% 控制）

opus 有 **7 道题**呈"块内早位置遇到→挂，晚位置遇到→解出"，如：
- `tablib-584`：run3 块内第 1 位挂；run1/2/4/5 第 5/8/9/6 位全解出（0.78–0.93）；
- `tenacity-615`：run2/3 第 1 位挂；run1/4/5 第 9/7/9 位解出（0.53–0.95）。

### 6.3 配对与免切分检验

| 检验 | opus-4.7 | qwen3.5-9B |
|---|---|---|
| 同题配对（晚位置均值 vs 早位置均值，中位切分） | **变好 13 : 变差 3**（符号检验 p≈0.01） | 变好 5 : 变差 7（无方向） |
| 免切分 Kendall 方向（题内位置×分数） | **10 : 4** | 7 : 5 |
| 严格反例（早解出、晚反挂） | 2 题 | **9 题** |

qwen 的 9 个反例 + `tenacity-614` 全程负迁移（baseline 解出、5 run 全挂）指向：**长历史对 qwen
不仅无益，有时有害**（20 万 token 的失败尝试与测试输出成为干扰）。

*样本量注*：5 runs × 19 题下 opus 的效应"显著但样本小"；若需写进正式报告，建议 runs 提到 10 加固。

## 7. qwen 的轨迹级失败模式与复用证据

**失败模式**（抽样自 interactions）：
- 全靠 `python << heredoc` 字符串替换改代码，反复 **"Pattern not found"** 而不自知（改动未落地）——`tenacity-597` 把一个改动摊到 5 个文件、死磕同一失败测试至 40 步耗尽；
- **绕过测试草率提交**——`tablib-594` 在 ODS 测试仍红的情况下靠 `pytest -k "not ods"` 通过即提交 → tests_failed；
- **验证脚本自坏即放弃**——`tenacity-615` 验证脚本崩溃（returncode −1）后 6 步就提交 → tests_failed；
- 多题 `submitted=True` 但 `tests_failed`（自以为修好）。

**复用的正面证据**（经验确实在被使用）：
- baseline 里 **18/19** 题以 `find /testbed -name '*.py'` 等宽泛探索开局；ICL run 里仅 3–8 次，且**集中在块入口（index 0 和 9）**——同库内结构知识被复用；
- baseline 常用绝对路径 + `cd /testbed`，run 内一致用相对路径——"记得"工作目录；
- 解出题步数 20.5 vs baseline 30.8（省 33%），省的正是重复探索。

**边界**：复用停留在"项目结构/工作目录/测试入口"的程序性知识层；没有观察到"把 A 题的修法迁移到 B 题"级别的深层迁移。

## 8. 结论与含义

1. **"gain 相近 ⇒ ICL 能力相近"不成立**：统计上（欠功效）与机制上（学习曲线 vs 静态偏移）双重不成立。看 **gain_norm 与 gain-by-index 的斜率/末段值**，不要只看 gain 均值——这个标量在本例中最具误导性。
2. **两模型差距的定位**：
   - 绝对编码能力：baseline 0.467 vs 0.067（7 题 qwen 六次机会全灭）；
   - 经验→正确率的转化：opus 能（块内 13:3、末段 gain 0.226），qwen 不能（5:7、斜率≈0）；
   - qwen 的经验消费止步于"环境熟悉度"（快了但对不了），且长历史轻微有害。
3. **qwen 是 capability-bound**：失败在写对补丁的基本功（替换脱靶、改动扩散、验证不严），更多上下文经验补不上这一层；提速证明记忆利用链路本身是通的。
4. **对后续实验的建议**：
   - 在 qwen 量级模型上研究 continual-learning 方法，codebase_adaptation 分辨率低（能力 bound 死），**poker / database_exploration 更能显出记忆机制差异**；
   - "上下文越多越好"对 qwen 不成立 → **icl_notepad（只留笔记弃全历史）有反超裸 icl 的动机**，值得对照实验；
   - 若对比要正式引用：runs 5→10 加固符号检验功效。

---

*数据与复现：本文全部数字可由 `results/codebase_adaptation/traces/2026-07-04T23-16-53.705279Z/`
与 `final_results/runs/icl-claude-opus-4.7/task_artifacts.json.gz` 复算（instance_outcomes 字段）。
我们侧运行配置见 `scripts/cluster/eval_qwen35_9b_codebase.sbatch`（apptainer 适配说明见
`src/tasks/container_backend.py` 模块 docstring）。*

# SWE-Bench-CL：从 240 条独立题合成 continual-learning sequence（设计文档）

> 活文档。记录如何把 `data/swe_bench_cl/full.jsonl`（240 条**独立**题）合成 CL sequence。
> 数据集/环境见 `swe-bench-cl-final-dataset.md`；qwen 单题能力基线见 `swe-bench-cl-qwen35-9b-baseline.md`。

## 0. 目标与固定结构

- 把 240 条独立题组织成 **continual-learning sequence**：agent 顺序解题、跨题带记忆，
  测"在前面题上学到的知识能否迁移到后面题"。
- **固定结构（与 codebase_adaptation 一致）**：每条 sequence = **9 + 10 两段**，**repo-blocked**
  （第 1 段 9 题来自 repo A，第 2 段 10 题来自 repo B），**切 repo 时 notify_change**。
  codebase 的 default 就是 9 tablib → 10 tenacity 的 TTTTTTTTT-NNNNNNNNNN。

设计拆成两个正交问题：
- **组间（inter）**：选哪两个 repo 配对 —— 见 §1（按共享 high-level 知识）。
- **组内（intra）**：每段 9/10 题选哪些、怎么排序 —— 见 §2（难度 curriculum）。

---

## 1. 组间：哪两个 repo 配一起（inter-group pairing）

**判据**：两个 repo 要有**共享的 high-level 知识**，迁移才有意义。SWE-Bench-CL 自带的
`overlap_results.json` 是**同 repo 内**任务对的文件重叠,不是跨 repo 知识相似度,帮不上;
从**库的本质 + 任务实际碰的模块**（modified_files 顶层）推理。

各 repo 任务实际改的模块（数据佐证）：

| repo | 高频模块 | 本质 |
|---|---|---|
| astropy | io, units, modeling, table, wcs | numpy 科学 + 单位/IO |
| xarray(pydata) | core | labeled N-d 数组 |
| scikit-learn | linear_model, ensemble, model_selection, base | estimator API + 数值 |
| matplotlib | matplotlib(core) | 渲染 / figure 状态 |
| pytest | _pytest | 测试框架内核（collection/hook/fixture） |
| sphinx | ext, domains, util, directives, builders | 文档框架（扩展/指令/构建器） |
| django | db, contrib, forms, core | ORM/关系建模 + web 框架 |
| sympy | polys, core, printing, physics, simplify | 符号表达式树 + 打印 |

### 两大知识家族 + 三个边缘 repo

**家族 A — numpy 数值科学栈**：**astropy / xarray / scikit-learn**
- 共享知识：ndarray 广播/dtype/shape/axis、NaN/inf、view-vs-copy；在数值数组上运算**同时保持
  元数据**（astropy units/table 列、xarray coords/dims、scikit 特征数组）。

**家族 B — 可扩展开发者工具框架**：**pytest / sphinx**
- 共享知识：hook/plugin 注册与派发、对一棵条目/节点树做 discover→process→emit 流水线、
  config 传播、节点树遍历改写、扩展 API 向后兼容。

**三个边缘 repo**：
- **sympy**：符号表达式树的构造/遍历/改写 + 打印；与数值家族**不通**（符号≠数值数组）；
  最近的知识伙伴是 **sphinx**（都做递归树变换 + 输出/打印）。
- **django**：ORM/关系建模自成一域，架构上偏家族 B（注册+配置+模板）。
- **matplotlib**：有状态对象图 + 后端抽象，较孤立；作为 numpy 消费者可弱挂家族 A（数据→可视化）。

### 配对排名（28 无序对；方向 A→B vs B→A 归 §2 curriculum）

**Tier 1 — 强共享知识（"迁移应生效"的正样本）**
| 对 | 迁移的具体知识 |
|---|---|
| astropy ↔ xarray | 带单位/坐标的 numpy 数组运算、元数据保持、dtype/broadcast 边界 |
| pytest ↔ sphinx | 插件/hook 注册、条目/节点树 discover→process→emit 流水线、config 传播 |
| scikit ↔ xarray / scikit ↔ astropy | numpy 数值算法、数组变换 |

**Tier 2 — 中等**
- astropy ↔ matplotlib、xarray ↔ matplotlib（数据→可视化，numpy）
- sphinx ↔ django、pytest ↔ django（框架：配置/注册/模板/测试基建）
- sympy ↔ sphinx（递归树变换 + 打印/输出）

**Tier 3 — 弱（几乎无共享知识 → 正好当负对照）**
- sympy ↔ {django, matplotlib, pytest, scikit}、matplotlib ↔ {django, pytest}、
  django ↔ {astropy, scikit, xarray}、scikit ↔ {pytest, sphinx}

### 中强共享知识配对清单（Tier 1 + Tier 2 = 9 对，候选正样本）

这 9 对是"有 high-level 共享知识、迁移应生效"的候选池，后续 sequence 从中选：

| # | 配对 | 家族 | 强度 | 共享知识 |
|---|---|---|---|---|
| 1 | astropy ↔ xarray | A | 强 | 带单位/坐标的 numpy 数组运算、元数据保持、dtype/broadcast |
| 2 | pytest ↔ sphinx | B | 强 | 插件/hook 注册、条目/节点树 discover→process→emit、config 传播 |
| 3 | scikit ↔ xarray | A | 强 | numpy 数值算法、数组变换 |
| 4 | scikit ↔ astropy | A | 强 | numpy 数值算法 |
| 5 | astropy ↔ matplotlib | A（+消费者） | 中 | 数据→可视化、numpy |
| 6 | xarray ↔ matplotlib | A（+消费者） | 中 | 数据→可视化、numpy |
| 7 | sphinx ↔ django | B | 中 | 框架：配置/注册/模板 |
| 8 | pytest ↔ django | B | 中 | 测试/框架基建 |
| 9 | sympy ↔ sphinx | 桥 | 中 | 递归树变换 + 打印/输出 |

家族贡献：家族 A（astropy/xarray/scikit + matplotlib 作消费者）→ #1,3,4,5,6（5 对）；
家族 B（pytest/sphinx + django）→ #2,7,8（3 对）；#9 是 sympy 唯一的中等知识桥。
其余 19 对（28−9）均为 Tier 3 弱对，作负对照。

### 实验设计建议
最有信息量的 CL 设计要**同时**有正样本（该迁移）和负对照（不该迁移），才能验证 CL 系统是否
**只在有共享知识时迁移**、不乱迁移。首批试水拟选 **2 正 + 1 负**：
1. astropy → xarray（家族 A，数值迁移）
2. pytest → sphinx（家族 B，框架迁移）
3. sympy → matplotlib（Tier 3，负对照，预期无迁移）

> 状态：配对逻辑已提出，待用户确认后生成对应 sequence（替换早期按 flavor 拍的 seq_easy/hard/ramp）。

---

## 2. 组内：每段选题与排序（intra-group curriculum）

**难度信号（两者结合，各 240/240 齐全）**：
- **人工**：`difficulty_score` 1–4（均值 1.53）/ `difficulty` 分档（<15min 125 / 15min–1h 104 /
  1–4h 9 / >4h 2）。与模型无关的先验难度。
- **qwen 经验**：本模型 4 遍 baseline 每题解出次数 0/4–4/4（`results/swe_bench_cl/qwen35_9b_baseline/
  passk_detail.jsonl`）。针对本模型的真实难度。

**组合难度** = 0.5×人工(difficulty_score 归一到[0,1]) + 0.5×qwen(1 − 解出次数/4)；
`avg_reward` 高者更易，作细排 tiebreak。

**选题（最终采用）**：`wrand` — **加权难度分层随机抽**。按组合难度排序后切 m 层,每层随机取 1,
既覆盖 easy→hard 全谱又有随机性(按 seq_id 定种子,可复现)。备用:`spread`(等距)/`easy`/`hard`。

**段内排序**：默认 **easy→hard**;若段内抽到**依赖对**,用**难度优先的拓扑排序**——保留合理的
拓扑序(前置在前),其余仍按难度 easy→hard。

### 关于 `dependencies`（重要认知）
- SWE-Bench-CL 的 `dependencies`(83/240 题有,118 条数据集内边)**全部同 repo、0 跨 repo**,
  且**方向 100% 是时间序**(前置 B 的 created_at 总早于 A)——即 A 的 PR 建立在 B 之上,
  **不是"简单前置知识"**。因此 **33% 的前置反而比依赖它的更难**。
- 我们的 rollout **每题从各自 base_commit 全新起容器、代码互相隔离**,跨题只带 agent 记忆、
  不带代码状态,故"B 代码先于 A"**无功能必要性**。
- 决策(用户):默认按难度随机抽;一个 variant 里抽到依赖对概率不大,**真抽到就保留合理拓扑序**
  (前置在前,哪怕前置更难)。纯难度排序对非依赖对自然满足"简单在难前面"。

**方向（谁先谁后）**：把"基础/家族代表"repo 放前段,另一 repo 放后段(考迁移)。可再调。

---

## 3. 工具

- `scripts/swecl/build_sequences.py <id> <repoA> <selA> <repoB> <selB> [display]`
  生成一条 9+10 sequence 到 `src/tasks/swe_bench_cl/schedules/<id>.json`（2 段、repo-blocked、
  notify、段内组合难度 easy→hard）。repo 用短名：astropy/django/matplotlib/pydata/pytest/
  scikit/sphinx/sympy。
- 运行：`clbench run --task swe_bench_cl --system icl --task-params '{"schedule":"<id>"}' ...`
  （CL 带记忆 = icl；纯能力 = 加 `--baseline-only`）。

## 4. 待决 / 下一步
- [x] 确认 §1 配对逻辑与首批 3 组（2 正 + 1 负对照）。
- [x] 定方向（基础/家族代表 repo 放前段）与段内 curriculum（wrand + 难度优先拓扑）。
- [x] 生成试水 3 条 → **2026-07-09 用 qwen3.5-9b 带记忆(icl)开跑**,每条 **5 run**(对齐
  codebase,mode=replicate 保 curriculum),baseline + 5 记忆 run。驱动 `scripts/cluster/run_swecl_sequences.sh`
  (A100 job 9084630, dp=8, 3 条并发);收尾 `scripts/swecl/finalize_sequences.py` → per-index
  baseline vs 5run 均值 + gain,分段1-learn/段2-xfer → `results/swe_bench_cl/qwen35_9b_sequences/`。
  分析:**不以"正 vs 负对照能否分离"论数据好坏**——迁移关键是 procedural knowledge(定位/跑测/
  读错/组补丁),与领域正交,负对照有 gain 是预期合理的。数据好坏的真标准是判分/parity/gold全过
  (已验证),与跑分形状无关。有意义的是**拆 gain**:通用程序性 vs 领域相关额外量(正样本段2 是否
  比负对照多一块)。
- [ ] 跑完读 SUMMARY:看记忆/notify 生效 + 拆解 gain 结构,再决定是否批量铺 9 对。
- [ ] 批量：按家族把更多 repo 对铺成覆盖式 sequence 集。

$$ p_\theta(Y=y \mid \mathbf{E}^{\prime}x) = p\theta(Y=y, Y \in A(x) \mid \mathbf{E}^{\prime}x) = p\theta(Y=y \mid Y \in A(x), \mathbf{E}^{\prime}x) p\theta(Y \in A(x) \mid \mathbf{E}^{\prime}_x) $$
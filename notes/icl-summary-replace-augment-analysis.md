# icl_summary（replace / augment）× Qwen3.5-9B · codebase_adaptation 轨迹分析

> 2026-07-06。自研 `icl_summary` 系统（移植 alchemy eval 的 summary-replace/augment 协议：
> 题目边界独立 LLM 调用重写记忆，window 抗漂移）首轮正式结果的三视角分析：
> 定量曲线与配对检验 / 记忆链质量与漂移 / 轨迹行为机制。
> 数据：replace `traces/2026-07-06T04-48-00*`，augment `traces/2026-07-06T04-49-24*`，
> 对照 icl `traces/2026-07-04T23-16-53*`（gain 均按 instance_id 对齐其 baseline，均值 0.067）。
> 配置：window=1、结构化两段式、5 runs、无 baseline、19 题（9 tablib + 10 tenacity 分块）。

## 0. 总表

| | icl | summary-replace | **summary-augment** |
|---|---|---|---|
| 均分 / 解出率 | 0.151 / 29.5% | 0.111 / 27.4% | **0.234 / 40.0%** |
| gain（对齐 baseline） | +0.084 | +0.044 | **+0.167**（gain_norm 0.179，超 opus-icl 的 0.147） |
| gain 分段（tablib 仓 / tenacity 仓） | 0.105 / 0.065 ↘ | 0.018 / 0.068 → | **0.110 / 0.219 ↗** |
| 同题配对（中位切分 好:差:平） | 5:7:7 | 6:4:9 | **9:4:6** |
| 免切分 Kendall（正:负:零） | 7:5:7 | 4:5:10 | **10:2:7** |
| 严格反例（早解晚挂，题/有序对） | 9/29 | 9/26 | **8/18** |
| gain 来源：新增解题 / 提效 | 15 / 10 | 10 / 13 | **21 / 15** |
| 解出题平均步数（baseline 30.8） | 20.5 | 24.8 | **17.6** |
| 5/5 全稳题 / 全灭题 | 0 / 7 | 1 / 9 | **2 / 6** |
| 作业耗时 | 7h39m（含baseline） | **2h25m** | 5h15m |
| 统计检验 | — | vs augment t=3.33 | vs icl **t=1.84**（5 runs 边缘显著） |

**augment 学习效应的证据在同题配对（9:4 / Kendall 10:2），不在曲线形状**。曲线上 tablib 仓内
有爬升（−0.035→0.27），但 idx10 的全局峰 0.55 经拆解为**构成效应**（5 次 permute 中 4 次恰好是
augment 最会做的 603/609 落位于此），idx11–17 的下滑同样源于四道全灭题（597/610/611/615）在
idx16/17 各扎堆 3 次；逐 run 的 tenacity 仓内前 5 位 vs 后 5 位为 2 升 3 降、无一致方向。
**结论：5 runs 下 mean-gain-by-index 的局部形状（尖峰/坡度）是采样噪声，不可作动态解读**；
augment 相对 icl 的提升集中在 tenacity 仓的可解题上（Top5 提升题中 4 道 tenacity；
tenacity-609 单题 +0.655）。

## 1. augment 赢的三条机制（轨迹实证）

1. **既省探索、又带配方**。宽泛探索开局率：baseline 94.7% → replace 77.9% → icl 30.5% →
   **augment 28.4%**（且精确落在两个仓库块入口）。关键个案 tenacity-609（aug 5/5 最快 **6 步**、
   icl 1/5、rep 2/5）：augment 第 1 步的 thought 尚未读文件就写出确切修法（记忆里已有 retry.py
   组合方法分析）；**icl 带着全量原始历史反而 4/5 超时**——同样的先验信息，
   **原始历史是噪声、蒸馏记忆是信号**。跨题复用实证："Added an escape_formula helper …
   (from previous issue #257)"。
2. **自己写的教训被真实执行**。提交前 3 步内跑全量 pytest：**augment 83.6% / replace 86.7% /
   icl 42.6%**。后果（提交质量三分化）：icl 假阳性提交（自以为修好实则 tests_failed）**52.9%**；
   replace 走向反面——过度谨慎，**71.6% 超时不敢提交**；augment 平衡（超时 34.7%、假阳性 32.8%）。
3. **负迁移被压制**。在 baseline 能解出的 5 题上：icl 掉到 52%（噪声历史带偏），**augment 保住
   68%**；严格反例有序对 29→18（−38%）。tenacity-614：baseline 会做，icl/replace 全灭 0/5，
   仅 augment 恢复 1/5。

## 2. replace 输的两个死穴

1. **有记忆不敢信**：每 run 宽泛重新探索 14–16 次（行为退化到 baseline 水平）。个案 tablib-547：
   记忆明写 `_xlsx.py`，开局仍 find→ls→逐个重读全库文件，40 步耗尽。
2. **覆盖式记忆被失败题冲成残渣**：tenacity-614 进题时记忆仅 170 字符，内容是上一题超时收尾的
   原始 JSON（`{"thought":"The task exceeded the 40 step limit…"}`）——好知识被垃圾版覆盖后无
   任何兜底，5 run 全灭；augment 同题带 1499 字累积记忆 11 步解出。

## 3. 记忆链质量与漂移（对标 alchemy QWEN_SUMMARY_DRIFT）

- **alchemy-4B 的"悲观化重写漂移"未复现**：blanket 否定率实测 ≈**0%**（vs 4B 的 88%）。
  所有 "never/cannot" 逐句核验均为具体技术事实（如 threading.local 不可 pickle、
  wait_exponential_jitter 不应直接调 super().__call__）。
- **缺陷一：换仓库处旧知识 100% 整段清空**（10/10 run，augment 也不例外）。tablib-presence
  矩阵 `TTTTTTTTT..........`——v9 起 tablib 段永久消失，两段知识非分区共存而是整体重写丢弃。
  该缺陷即 gain 曲线 idx9 回落的记忆侧解释；augment 靠生历史兜底+块内快速重建扛过，replace 无兜底。
- **缺陷二：29%（replace）/37%（augment）的记忆版本是生成失败的垃圾 stub**——总结调用被满屏
  `{"thought","command"}` 轨迹带偏，输出工具调用 JSON 而非总结。replace 模式下坏版直接覆盖好知识
  （死穴 2 的成因）。
- 块内 carry-forward 良好（`src/tablib` 与 pytest 命令 v0→v8 连续存活且愈发精确）；
  Lessons 段质量高且对症（脆弱 sed / 提交前全量测试 / 先复现再修，failure-mode 引用 2–7 处/run）。
- 记忆最终质量与单 run 成绩无肉眼相关（augment 最高分 run 的 final 恰为坏版）——
  单题成绩主要由当轮行为决定，记忆是慢变量。

## 4. 可直接落地的改进（预期两组都能再涨，replace 或有质变）

1. **修垃圾版**（最高优先）：总结调用后加输出校验，不像总结（JSON 工具调用形态）即重试——
   挽回约 1/3 的记忆更新；对 replace 是救命性的。
2. **修换仓清空**：总结指令加"按仓库分区组织、保留所有见过仓库的知识"。
3. 修复后重跑 replace——其"蒸馏记忆复利"假说（gain 单调升 + 2.5h 超低成本）值得公平的第二次机会。

## 5. 方法论备注

- 位置聚合类指标（mean gain by index）在 5 runs 下各位置题目构成受 permute 采样影响，
  id 对齐 baseline 仅在均值意义消除难度；因果结论以同题配对 / Kendall 为准。
- augment vs icl 的 t=1.84 属边缘显著；正式引用建议 runs 5→10。
- 复现：`scripts/cluster/eval_qwen35_9b_codebase_summary.sbatch <replace|augment>`；
  系统实现 `src/systems/icl_summary/`（含 `--skip-baseline` CLI 旗标）。

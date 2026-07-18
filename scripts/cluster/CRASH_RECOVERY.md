# 评测作业崩溃恢复手册（salvage + 定向补跑 + 合并）

> 适用场景：`clbench run` 多 run 评测作业在**最终写盘前**死掉（OOM / 节点故障 / 单 run 异常导致全灭）。
> 框架的正式 traces 要等全部 run 跑完才由 cli 一次性写盘；但只要开着 `--live-dashboard`
> （本仓库跑评测的铁律），每个 run 的完整轨迹会随进度写进 live 快照——**完成态的 live 快照
> 与最终 trace 逐字节等价**（已实测验证，唯一例外见"注意事项"）。

## 前置条件

- 崩溃的作业当时开了 `--live-dashboard`（快照在 `results/<task>/live/<group>/run_N.json`）；
- 本仓库两件工具（2026-07-06 加入，均为纯增量）：
  - `clbench run --run-indices 2,4`：只执行指定编号的 run（种子机制保证补跑顺序与原 run 完全一致）；
  - `scripts/cluster/salvage_merge_runs.py`：抢救 + 合并 + 重建 viewer 包。

## 标准流程（三步）

### 第 1 步：盘点损失

```bash
uv run python scripts/cluster/salvage_merge_runs.py \
    --task <task_name> \
    --live-dir results/<task_name>/live/<崩溃组时间戳> \
    --expected-runs 5 --allow-partial
```

输出会列出：live 里有哪些**完成态** run（按 trace 内嵌的 `execution.run_index` 对号，
文件名 run_N 是 1 起数、内部编号 0 起数，脚本自动处理）、缺哪几个编号。
同时这一步已经把能救的部分写成了 `traces/<组名>-salvaged/` + 部分 viewer 包（可先行分析）。

### 第 2 步：只补缺失的 run

```bash
# 用与原作业完全相同的 task/system 参数, 只加 --run-indices（逗号分隔的内部编号）
clbench run --task <task_name> --system <system> \
    --task-params '<与原作业相同>' --system-params '<与原作业相同>' \
    --skip-baseline --run-indices 2,4 \
    --live-dashboard --no-live-server --verbose-runs
```

要点：`--runs`（或 schedule 默认值）必须覆盖最大编号；补跑的 run 顺序由
`seed + run_index` 决定，与崩溃前那次一模一样，数据可直接拼接。

### 第 3 步：合并出完整产物

```bash
uv run python scripts/cluster/salvage_merge_runs.py \
    --task <task_name> \
    --live-dir results/<task_name>/live/<崩溃组> \
    --extra-traces-dir results/<task_name>/traces/<补跑组> \
    --expected-runs 5
```

同编号冲突时补跑数据优先。产出：

```
results/<task>/traces/<崩溃组>-salvaged/run_0000..000N.json   ← 标准布局
results/<task>/viewer_artifact_<崩溃组>-salvaged_*.json.gz     ← 合法 viewer 包
```

## 验证清单

- 合并输出里每个 `run_index` 的题数 = schedule 长度（如 19）；
- viewer 包能在 `viewers/single_task_viewer.html` / `compare_traces.html` 正常加载；
- 若做跨系统对比：确认补跑与原 run 的代码版本差异是否影响行为（通常修复类改动只在
  "原本会崩的轮次"上有差异，混用可辩护，但报告里加一句脚注）。

## 注意事项

- **`system_memory` 例外**：live 快照里该字段为 null（记忆链导出发生在终局写盘）。
  icl_summary 的记忆链可从每轮 `metadata.summary_text` 重建；ACE 的 playbook 在每轮
  `metadata.ace_playbook`；不影响分数与对话数据。
- 抢救工具**只读输入**，输出全部落在 `-salvaged` 新目录，跑多少遍都不会破坏原始数据。
- `--skip-baseline` 的组没有 baseline 段，viewer 的 gain 列为 n/a 属预期
  （gain 需要该系统自己的 stateless baseline，见 notes 里的相关讨论）。
- 若 live 快照里某 run 显示 19/19 但 `status != completed`（正在写最后一题时死的），
  脚本会拒收该 run——宁缺勿错，把它加进 `--run-indices` 重跑。

## 历史案例

- 2026-07-06 ACE 第四次（job 9016679）：79/95 时单轮异常全灭。live 抢救回 run 0/1/3
  （各 19/19），缺 2/4 —— 即本手册的诞生现场。

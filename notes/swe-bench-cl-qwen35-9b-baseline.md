# qwen3.5-9b × swe_bench_cl 纯能力 baseline 评测结果

> 2026-07-09 完成。数据集与环境见 `swe-bench-cl-final-dataset.md`(240 题清洗/筛选/环境 parity)。

## 1. 口径与方法

- **纯能力评测**:`clbench run --system icl --baseline-only`。baseline 每题构造**全新 task+system**
  求解(无跨题记忆),是干净的单题能力信号,**不是** icl 带记忆的持续学习 run。
- **pass@4 / avg@4**:每题独立跑 **4 遍**(温度默认 1.0,天然多样),按 instance_id 聚合。
  - **pass@1** = 所有(题×遍)960 次尝试的平均解出率。
  - **pass@4** = 每题 4 遍里至少一遍解出的题占比。
  - **avg@4** = 每题先对其 4 遍 reward 求均值,再对题平均(reward = 负 per-issue regret,解得越快越高)。
- **判分**:走 task 真实判分路由(django/sympy 官方 SWE-bench 解析,其余 pytest returncode);
  success=True 即该题所有 F2P fail→pass、所有 P2P 保持 pass。
- **资源**:2 节点各 dp=8 serve qwen3.5-9b,各跑 2 遍全量(A100 seed0,1 + RTX_6000 seed2,3),
  2.2 小时。agent 环境与 codebase_adaptation 逐项一致(见数据集文档 §3)。
- 模型 `/data/user_data/qixinx/Qwen3.5-9B`,nothink 模板,`max_tokens=240000`,每题 ≤40 步。

## 2. 核心指标

| 指标 | 值 |
|---|---|
| **pass@1** | **0.275**(264/960) |
| **pass@4** | **0.425**(102/240) |
| **avg@4** | **0.1211** |

4 遍各自解出 63 / 64 / 68 / 69 题,稳定。

## 3. 按 repo

| repo | n | pass@1 | pass@4 | avg@4 |
|---|--:|--:|--:|--:|
| scikit-learn | 32 | 0.398 | **0.594** | 0.176 |
| pydata/xarray | 17 | 0.309 | 0.529 | 0.143 |
| pytest-dev | 17 | 0.294 | 0.529 | 0.139 |
| django | 49 | 0.357 | 0.510 | 0.166 |
| sympy | 50 | 0.290 | 0.420 | 0.124 |
| astropy | 17 | 0.279 | 0.353 | 0.120 |
| matplotlib | 25 | 0.110 | 0.240 | 0.038 |
| sphinx-doc | 33 | 0.106 | 0.212 | 0.040 |

matplotlib / sphinx-doc 明显最难(pass@1 ~0.11,avg ~0.04)。

## 3b. 与 codebase_adaptation 两 repo 对比(qwen3.5-9b 同模型)

把 codebase_adaptation 的 tablib/tenacity(每题 16 次 baseline)并入,按 pass@4 排序。
† = codebase_adaptation;其 pass@4 用无偏估计量 `1−C(16−c,4)/C(16,4)`(从 16 次估计 4 次口径),
其余 8 个 repo 是每题 4 次的实测 pass@4。reward 同一 regret 公式,avg 同尺度可比。

| repo | n | pass@1 | pass@4 | avg |
|---|--:|--:|--:|--:|
| tablib † | 9 | 0.354 | **0.652** | 0.114 |
| scikit-learn | 32 | 0.398 | 0.594 | 0.176 |
| pydata/xarray | 17 | 0.309 | 0.529 | 0.143 |
| pytest-dev | 17 | 0.294 | 0.529 | 0.139 |
| django | 49 | 0.357 | 0.510 | 0.166 |
| sympy | 50 | 0.290 | 0.420 | 0.124 |
| tenacity † | 10 | 0.250 | 0.394 | 0.103 |
| astropy | 17 | 0.279 | 0.353 | 0.120 |
| matplotlib | 25 | 0.110 | 0.240 | 0.038 |
| sphinx-doc | 33 | 0.106 | 0.212 | 0.040 |

- tablib/tenacity 数据源与方法见 `qwen35-9b-baseline-passk.md`(16 遍 stateless baseline)。
- 注意:16 次估计更稳,4 次实测噪声更大;两组镜像难度分布不同,横比只看量级。
- 结论:tablib 最易(.65),tenacity 中游(.39),与 SWE-bench 8 repo 交错;scikit/pydata/pytest/
  django 的 pass@4 与 tablib 同量级或更高。sphinx/matplotlib 仍垫底。

## 4. 每题解出稳定性(4 遍中解出次数分布)

| 解出次数 | 题数 | 占比 |
|---|--:|--:|
| 0/4(从没解出) | 138 | 57.5% |
| 1/4 | 27 | 11.3% |
| 2/4 | 19 | 7.9% |
| 3/4 | 25 | 10.4% |
| 4/4(每遍都解出) | 31 | 12.9% |

## 5. 产物位置

全部在 repo 内 `results/swe_bench_cl/qwen35_9b_baseline/`(dump 不写 user_data/group_data):

- 指标报告:`results/swe_bench_cl/qwen35_9b_baseline/FINAL_REPORT.txt`
- **960 条轨迹**:`results/swe_bench_cl/qwen35_9b_baseline/trajectories/<遍时间戳>/<instance_id>.json`
  - 每条:逐步 `{step, thought, command, observation, done}` + `success` + `reward` + `n_steps`
  - `trajectories/index.jsonl`:960 条汇总(instance_id / pass_label / success / reward / n_steps)
- 每题详表(含每遍 reward):`results/swe_bench_cl/qwen35_9b_baseline/passk_detail.jsonl`
- 原始 clbench trace:`results/swe_bench_cl/live/2026-07-09T0{5,6}-*/baseline.json`(4 遍)

## 6. 已知说明

- **baseline 轨迹不单独存最终 git patch**(dump 里 `patch` 字段为空),但 agent 每步编辑命令+观察
  完整保留,补丁可从命令还原。如需独立 diff,要改判分路径把 model_patch 落进 trace(动共享代码)。
- 4 遍靠采样温度产生多样性(seed 只影响题序,不影响模型采样);pass@4 是 4 次独立尝试的并集。

## 7. 复现脚本(scripts/swecl/, scripts/cluster/)

- `run_swecl_passes.sh <dp> <tag> <npass> <seed_base>`:单节点 serve + N 遍 baseline。
- `swecl_baseline_nodeB.sbatch`:rl 分区 RTX 节点提交模板(qos=rl_qos)。
- `passk_report.py <baseline.json...>`:pass@1/pass@k/avg@k + 按 repo + 分布。
- `dump_trajectories.py <outdir> <baseline.json...>`:轨迹 dump。
- `finalize_swecl_eval.py <start_epoch> [need] [expect] [max_h]`:等 N 遍跑完→自动出报告+dump。

## 8. 可跟进

- 失败模式分析:138 道零解题卡在哪、sphinx/matplotlib 为何难(需读轨迹)。
- 持续学习口径(icl 带记忆)对比,看跨题学习增益。

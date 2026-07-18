# SWE-Bench-CL 训练数据的环境（镜像）落地方案

> 2026-07-08。目标：用 `/home/qixinx/codebase-data/agents-never-forget`（SWE-Bench-CL）为 codebase_adaptation
> 构造训练数据。本文只解决**环境/镜像**问题——数据集本身不带镜像，怎么在我们无-docker 的 apptainer 集群上跑出可信的 FAIL_TO_PASS 判定。
> 结论先行：**绕过该项目自带的 harness，直接用官方 SWE-bench Verified 的 per-instance 镜像**。

## 1. 调研结论（三点，均有代码/论文证据）

1. **数据 = SWE-bench Verified 的重排**。构造脚本 `scripts/SWE-Bench-CL_dataset_construction.py:21` 直接读
   `hf://datasets/princeton-nlp/SWE-bench_Verified`。每题保留标准 SWE-bench 键：`instance_id / repo / base_commit`，
   Task-Stream 版还保留 `environment_setup_commit / version`（Curriculum 版丢了后两者）。**没有引入任何新 repo**——
   全是 django/sympy/sphinx/matplotlib/sklearn/astropy/xarray/pytest 等 SWE-bench 经典库。

2. **项目本身不提供、也不引用任何镜像**。全库无 Dockerfile、无 registry 引用、数据无 image 字段。

3. **项目自带的 3 个 harness 都不能用来做可信评测**：
   - `eval_v1`（唯一带 Docker 的）调官方 `swebench.harness.run_evaluation`，但传 `--dataset_name princeton-nlp/SWE-bench`
     （原始集 2294 题，非 Verified），论文 Section 5 自陈"镜像与任务 mismatch、pass rate <8.5%"后**放弃**；
   - `eval_v2_agent` / `eval_v3_swe-agent`：不用容器，git clone 到宿主机、测试命令硬编码 `unittest discover`、
     判定是"stderr grep 不到 FAIL 就算过"的弱正则，作者 docstring 自注"simplified"，只在 `dummy_math_project` 玩具样例上跑过。

→ 我们要的是可信判定，**只能走官方 SWE-bench Verified 镜像**，用 CL 数据里保留的 `instance_id` 索引。

**判定要 FAIL_TO_PASS + PASS_TO_PASS 双判**（完整 resolved 定义：修复的测试从挂变过 **且** 原本通过的测试无回归）。
这更坐实了必须用官方镜像：项目自带 v2/v3 的弱正则连 F2P 都不可信，P2P 回归检查更做不了；而官方镜像的 `/eval.sh` +
`swebench` 的 `test_spec` **本来就同时判 F2P 和 P2P**（官方 grading 标准逻辑），走官方路线 P2P 是白送的，无需自己实现。
数据里两组测试都齐全：Task-Stream 每条带 `FAIL_TO_PASS(_list)` 和 `PASS_TO_PASS(_list)`。

## 2. 官方镜像方案（方式 B）

- 官方为 Verified 每个 instance 预建镜像，Docker Hub 命名（`__` → `_1776_` 等转义，验证时以 swebench 包实际输出为准）：
  `swebench/sweb.eval.x86_64.<escaped_instance_id>`，例：`astropy__astropy-12907` → `swebench/sweb.eval.x86_64.astropy_1776_astropy-12907`。
- 镜像内：代码在 `/testbed`，conda 环境已装好该 repo 该 version 的依赖，自带 `/eval.sh` + 测试规格。环境规格已烤进镜像，
  **不需要 environment_setup_commit/version 来现场搭环境**（这也是为什么 Curriculum 版丢了这俩字段仍能跑）。
- instance_id → 镜像映射由 `swebench` 包的 `make_test_spec` 自动完成（当前集群 python 未装 swebench，验证阶段需装）。

## 3. 磁盘：用 group_data 大盘，不再是约束

- **存放位置：`/data/group_data/rl/yuxiaoq/qixinx`**（组盘，目录已存在可写）——SIF 和镜像缓存落这里，
  **不要**放 `user_data`（500G 只剩 108G）。
- ⚠️ **实际可用空间待核实**：`df` 显示挂载点 `/data/group_data/rl` 有 20T 空闲，但那是**整个组盘**的数字，
  不代表本子目录能用满——组盘被组内其他人占用 + 可能有子目录配额都不在该数字里。全量可行性取决于 du/quota 实测结果。
- 官方镜像每个解压后约 1–2GB，500 题全量 SIF ≈ 500GB–1TB。**全量是否可行 = 本目录实际余量是否 ≥ ~1TB（待核实）**；
  若余量有限则退回"按用到的子集拉"。
- 子集拉取现在只是"省时间/省带宽"的优化，不再是硬约束。Task-Stream 的 repo 分布（供规划下载顺序）：

  | repo | 题数 | | repo | 题数 |
  |---|---|---|---|---|
  | django/django | 231 | | astropy/astropy | 22 |
  | sympy/sympy | 75 | | pydata/xarray | 22 |
  | sphinx-doc/sphinx | 44 | | pytest-dev/pytest | 19 |
  | matplotlib/matplotlib | 34 | | pylint-dev/pylint | 10 |
  | scikit-learn/scikit-learn | 32 | | psf/requests | 8 |
  | | | | mwaskom/seaborn / pallets/flask | 2 / 1 |

  注：同 repo 不同 instance 的镜像**不共享大部分层**（每个 base_commit 一套环境），磁盘成本≈题数×单镜像。
  验证阶段仍建议先用小 repo（astropy/pytest/requests）——省的是下载时间，不是磁盘。
- 环境变量：`APPTAINER_CACHEDIR` / `TMPDIR` 也要指到 group_data 大盘（pull 时的中间层缓存同样占几百 GB）。

## 4. 待验证的三个未知数（决定全量可行性）

1. **官方 Verified 镜像能否直接 `apptainer pull docker://swebench/sweb.eval.x86_64.<id>`**（网络/命名/是否需登录）；
2. **单个 SIF 实际体积**（决定子集规模上限：剩 108G 能放多少题）；
3. **现有 `src/tasks/container_backend.py` 的 singularity 后端能否托管官方镜像**——它为 tablib/tenacity 自打包镜像设计
   （`/testbed` + mini-swe-agent + COMPLETE_TASK sentinel），官方镜像的 `/eval.sh` + conda 约定不同，可能需适配层。

## 5. 建议的最小验证链路（过目后再跑，暂不动手）

选 **1 个小 repo 的 2–3 个 instance**（推荐 astropy，题少镜像小），走通：

```
① apptainer pull docker://swebench/sweb.eval.x86_64.astropy_1776_astropy-12907  → 记录 SIF 体积
② apptainer exec <sif> bash -c "cd /testbed && git status"                        → 确认 /testbed 布局
③ 在容器里 apply test_patch（数据里的 evaluation.test_patch）
④ 跑该 instance 的 FAIL_TO_PASS 测试命令（用 swebench 的 test_spec 生成，或镜像自带 /eval.sh）
⑤ 先不打 gold patch 应挂、打上 gold patch 应过 → 验证判定链路正确
```

这一步同时回答 §4 的三个未知数。验证通过后再决定：
- 集成路线（纯验证镜像可用 / 接进 codebase_adaptation 的 mini-swe 流程）——用户当前选"先列方案不动手"；
- 数据集选型（Task-Stream 500 扁平带 env 字段 / Curriculum 273 嵌套带依赖图）——用户当前选"两份都要，后面再定"。

## 5b. 最小验证结果（2026-07-08，astropy__astropy-12907，已跑通）

**结论：官方镜像在我们 apptainer 集群上可做可信 F2P+P2P 双判。全链路验证通过。**

关键数字与坑（全部实测）：
- **镜像名规则**：`swebench/sweb.eval.x86_64.<instance_id.replace("__","_1776_")>:latest`。6 个不同 repo 抽查 registry 全部 HTTP 200，通用可靠，**不依赖 swebench 包即可批量生成镜像名**。
- **速度关键**：pull+转换**必须在本地 NVMe**（本节点 `/scratch`，6T，`/dev/nvme0n1p2`），**不能在组盘 NFS 上转**。
  - 本地盘：**单个镜像 56s，SIF 999M**；组盘 NFS：同一个卡 20+ 分钟未完成（mksquashfs 打包几万小文件在 nodev NFS 上是灾难）。
  - 全量估算：500 题串行 ~8h，**8 路并行 ~1h**；磁盘 ~500G（成品 SIF 拷回组盘存）。
- **容器布局**：代码在 `/testbed`（HEAD = base_commit + 一个 "SWE-bench" test 提交）；conda env 在
  **`/opt/miniconda3/envs/testbed`**（本题 python 3.9.20，astropy 依赖已装）；根目录无 `/eval.sh`（测试命令运行时生成）。
- **exec 参数（踩坑后的正确配方）**：
  ```
  apptainer exec --contain --writable-tmpfs --bind <patch_dir>:/host_patches <sif> bash -c '
    cd /testbed
    git apply /host_patches/test.patch      # 先打 test_patch（加测试）
    git apply /host_patches/gold.patch      # 评测 agent 补丁时换成 agent 的 diff
    source /opt/miniconda3/bin/activate testbed
    python -m pytest <全部 F2P...>          # F2P：修复的测试
    python -m pytest <全部 P2P...>          # P2P：回归测试（必须跑全部，不能抽样）
  '
  ```
  - `--contain` 必须加：否则宿主 `/opt`/`/data` 挂载遮住镜像自带的 conda（会找不到 testbed env）。
  - `--writable-tmpfs` 必须加：SIF 只读，git apply 要写；同时给 hypothesis/`.hypothesis` 提供可写层。
  - 镜像自带断网（日志 "Internet access disabled"），符合 SWE-bench 防作弊规范。
- **判定逻辑**：F2P 全 passed **且 全部 P2P 全 passed** → resolved。P2P **必须全跑**——抽样会漏掉 gold/agent 补丁对
  未抽中测试的回归破坏，判定不可信。实测（全量）：打 gold 前 F2P 2 failed；打 gold 后 F2P 2/2 passed + **P2P 全 13/13 passed**。

未决：`nodev` 组盘只用于**存**成品 SIF（只读 exec 不受 nodev 影响，已验证）；转换阶段一律走本地 NVMe。

## 5c. 镜像存放工作流（防丢失）

- **转换在本地 NVMe，成品存 /data 持久盘**：
  - 转换临时区：`/scratch/qixinx/swe_{cache,tmp,sifs}`（本地 NVMe，快，但**换节点即丢 + 集群定期清**）；
  - 持久存放：`/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs/`（组盘，转完立即 `cp` 回来）。
  - 已验证：从 nodev 组盘**只读 exec 正常**（astropy import 成功）——nodev 只影响 build/转换，不影响运行。
- 已存的验证镜像：`/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs/sweb-astropy-12907.sif`（999M）。

## 5d. 集成目标：最终接进 miles，不是 clbench 直跑

- 训练在 **miles**（`/home/qixinx/miles/examples/codebase_adaption/`）里做，它已把 19 题分了 train/heldout。
- 但 miles 是"训练胶水"——README 明说 "containers, and scoring stay in continual-learning-bench"；`schedule.py` 调
  "clbench stage lookup payload"。即 **miles → clbench → 容器镜像** 三层，miles 不碰环境。
- 所以集成路径 = **在 clbench 层新增 `swe_bench_cl` task**（吃 SWE-Bench-CL 数据 + 起官方 SIF + F2P/全量P2P 判定），
  miles 侧最小改动（换数据源 + schedule 指向新 task），ACT/WRITE/gain/metrics 胶水复用。
- 契约细节由两个 Explore agent 摸清后写成集成 plan（进行中）。

## 6. 关键文件

- 数据：`/home/qixinx/codebase-data/agents-never-forget/data/SWE-Bench-CL-Task-Stream.json`（500题，带 env/version）
  / `.../SWE-Bench-CL-Curriculum.json`（273题，嵌套，带依赖图，无 env/version）
- 构造脚本：`.../scripts/SWE-Bench-CL_dataset_construction.py`
- 官方 harness 调用参考：`.../eval_v1/eval_procedure.py:647-701`
- 论文：`.../SWE_Bench_CL_NeurIPS_Submission.pdf`（Sec 5 = 放弃官方 Docker 的原因；Sec 6.4 = 非容器化流程）
- 我们的 singularity 后端（复用/适配对象）：`src/tasks/container_backend.py`

# SWE-Bench-CL → clbench → miles 集成 plan

> 2026-07-08。承接 `swe-bench-cl-env-plan.md`（环境/镜像已验证跑通）。
> 目标：用 SWE-Bench-CL 数据 + 官方 SWE-bench 镜像，在 clbench 新增一个 task，最终接进 miles 做 memory 训练。
> **本文是方案，未动手写代码。**

## 0. 全景：三层 + 每层改动量

```
miles (训练胶水)            → clbench (task/容器/评分)      → 官方 SWE-bench 镜像 (SIF)
schedule.py 改常量            新增 swe_bench_cl task           已验证可用(env-plan §5b)
rollout/advantage/metrics    复用 codebase_adaptation 大部件    命名 __→_1776_, conda testbed
  零改动                       新写 loader + evaluator
```

契约边界（两个 Explore agent 已确认）：
- **miles 胶水**（`codebase_rollout.py`/`codebase_advantage.py`/`codebase_metrics.py`/`prompts.py`）：ACT/WRITE 双流、memory、gain、GRPO、指标——**全部 instance_id 无关，零改动**。
- **miles `schedule.py`**：硬编码的 instance_id 清单/train-heldout 切分/repo 映射——**改常量**。
- **clbench**：起容器、apply patch、跑测试、reward——**复用容器编排原语，新写数据 loader + 判定 evaluator**。
- **镜像**：官方 per-instance SIF——**预拉，已验证**。

## 1. clbench 侧（主要工作量）

### 1a. 新 task 目录（注册靠目录发现，无中央清单）
`src/tasks/swe_bench_cl/`：
- `task.py`：`@register_task("swe_bench_cl") class SweBenchCLTask(ContinualLearningTask)`。
  - 必需：`r_max` 类属性、`__init__(num_instances=...)`、实现 `build_canonical_run_state / step / evaluate / build_current_query / name`。
  - **骨架直接抄 `codebase_adaptation/task.py`**——它已是 SWE-bench 式 git_pr 流程（base_commit + test_patch + F2P/P2P），mini-swe-agent step/submit 循环、`BashCommandResponse`、`_start_container`/`_execute`/`_extract_patch`/`_handle_submit`/`_make_issue_query` 全可复用。
  - `schedules/default.json`（run-all 需要；miles 路径绕过它，但 clbench 直跑/baseline 需要）。
- `task_loader.py`：仿 `codebase_adaptation/task_loader.py`，读 SWE-Bench-CL jsonl → `TaskInstance`。

### 1b. 数据格式转换：SWE-Bench-CL → clbench git_pr jsonl
每行必须满足 `task_loader` + `generic_runtime` 的 git_pr 契约字段：
```
instance_id, repo, base_commit, patch(gold), test_patch,
FAIL_TO_PASS, PASS_TO_PASS,
image_name,                    # 见 1c
problem_statement,
runtime_mode: "git_pr",        # is_generic_pr_instance 强制要求
workdir: "/testbed",           # 官方镜像一致
test_command: "source /opt/miniconda3/bin/activate testbed && python -m pytest",  # 见 1d
test_timeout: 300, sequence_rank: <序列位置>
```
源数据两份都要（用户定）：`Task-Stream`（500题，扁平，带 env/version）+ `Curriculum`（273题，嵌套，带依赖图）。
写一个 `scripts/build_swecl_dataset.py` 做转换（嵌套→扁平、字段改名、补 runtime_mode/image_name/test_command）。

### 1c. image_name = 官方镜像名，靠现有 SIF 解析
- `image_name` 直接填 `docker://swebench/sweb.eval.x86_64.<escaped_id>` 或本地 SIF 绝对路径。
- 转义函数：`instance_id.replace("__","_1776_")`（env-plan 已验证 6 repo 通用）。**现有代码没有这个转义**，在 loader 里加。
- `resolve_singularity_image()`（`container_backend.py:90`）现有 sanitize（`/`→`_`,`:`→`_`）+ `CLBENCH_SIF_DIR` 查找逻辑可复用；若我们的 SIF 文件名按官方 tag 命名，需在 loader 里把 image_name 归一化成 backend 期望形式，或给 resolve 加一条候选。

### 1d. 判定 evaluator：复用现有退出码判定 + 前置数据清洗闸门（主推）
**决策（2026-07-08 与用户讨论后修正）：复用 clbench 现有 `returncode==0` 判定，不新写官方逐条 evaluator。**

背景——官方逐条判定与 clbench 退出码判定的两处差异，经分析都不需要靠 evaluator 解决：
- **差异1（F2P 本来就 pass）**：这是**脏数据**（标注错 / 该题在此环境不成立），本就不该进训练集。clbench 退出码判定不为脏数据容错，反而是我们想要的——脏题直接剔除，而非靠 grading 兜底。
- **差异2（P2P collection error / pre-existing fail 导致退出码误判失败）**：确实是 clbench 的误判，但**这些题在数据清洗阶段就能筛掉**（见下），不让它们进数据集即可，无需 evaluator 区分。

**关键：数据清洗 = 镜像验证阶段的强制一步（本来就要做）。** 对每道题跑 gold-patch 双向验证：
1. 不打 gold（仅 test_patch）跑 F2P + 全量 P2P：**应 F2P 全 fail + P2P 全 pass**；
2. 打 gold 跑 F2P + 全量 P2P：**应 F2P 全 pass + P2P 全 pass**。
只有两步都符合预期的题才进数据集。这一步天然筛掉：F2P 本来就 pass（差异1）、P2P 有 pre-existing fail/collection error（差异2）、gold 在本环境复现不出官方结果（环境不对齐）的所有题。

**净效果**：对通过清洗闸门的干净题，退出码判定 ≡ 官方逐条判定（导致分歧的边界情况已提前剔除）。因此 `evaluate_generic_pr_submission`（`generic_runtime.py:546`）**零改动复用**，省掉新写 evaluator。代价：镜像准备阶段每题多跑一次 gold 双向验证——而这本来就是判断镜像可用性的必要步骤，等于清洗+验证合一。

**产出**：`scripts/swecl/{extract_instances.py, pull_and_validate_one.sh, batch_pull_validate.sh}`（对每题：起 SIF → 空跑 + gold 跑 → 判是否符合预期 → 通过的写入 clean 数据集，不通过的记录剔除原因）。
（可选备份方案：若清洗后仍担心，再用宿主侧 `swebench` log parser 做逐条双判——但预期不需要。）

**⚠️ 实测教训（2026-07-08，astropy+pytest 41题子集）：**
1. **node 预过滤是刚需（必须带进 clbench evaluator）**：SWE-bench 的 P2P 列表含畸形 node ID（如参数化空串 `test_compose_roundtrip[]`），pytest **一条选不中就整条 abort(exit 4)，带崩其余上百条**。所有大 P2P 题（全部 pytest 题 + 一半大 astropy 题）都中招。修法：跑测试前先 `pytest --collect-only --continue-on-collection-errors`，解析 `not found:` 行，过滤掉选不中的 node 再跑。**这个逻辑将来接进 clbench `generic_runtime` 的判定时也要带上**（大 repo 通用问题，现有 codebase_adaptation 的 19 题都是小 P2P 没暴露）。
2. **测试名传参用 subprocess list，不走 shell**：参数化 ID 含空格/方括号/`%`，`$(cat|tr)` 不加引号会被 shell 拆碎。容器内用 python `subprocess.call([...]+test_ids)` 传 list。
3. **testbed 是 python3.6**：`subprocess.run(capture_output=)` 不可用（3.7+），用 `stdout=PIPE`。
4. **清洗拦截率**：41 题 → **clean 38（93%）**，dirty 3 全是真边界：2 道 gold 后 F2P 仍不过（本环境复现不出，13033/13398）、1 道 pytest 测自身的 rootdir 坑（7521）。**预期全 500 题会有 ~5-10% 因这类边界被剔除，正常**。
5. **速度**：clean SIF 平均 966MB，pull+双向验证平均 56s/题（缓存命中后 14s）。批处理脚本 `batch_pull_validate.sh` 有个坑已知：拆分目录 `$SC/lines/` 不自清，重跑前需手动清或换目录（否则会混入上轮拆分文件重跑）。

**产物落地**：`swecl_clean.jsonl`（38题）+ 38 个 SIF 于 `/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs/`；report 于 `swecl_report.jsonl` + `swecl_rerun_report.jsonl`。

### 1e. reward 语义（要你定，见"待决"）
- 选项 A：沿用 codebase_adaptation 的 regret 式 `reward = 1-(steps-1)/40`（带步数效率维度）。
- 选项 B：纯 0/1（resolved=1 否则 0）。
- 训练用途下 regret 式给的信号更密（部分学到东西也有梯度），但 0/1 更接近 SWE-bench 官方。

## 2. miles 侧（最小改动）

`/home/qixinx/miles/examples/codebase_adaption/`：
1. `schedule.py`：`DEFAULT_STAGE_IDS`(18) 换成 SWE-Bench-CL 的 repo 块×instance_id；`DEFAULT_TRAIN_COUNTS`(50) + `REPO_BY_STAGE`(52) 按新 repo 填。
2. `codebase_rollout.py:259` `_make_task` 的 `dataset_path` 指向新 swe_bench_cl jsonl（建议 env/arg 参数化）；`from src.tasks.swe_bench_cl.task import SweBenchCLTask`。
3. `codebase_config.yaml`：`codebase_baseline_artifact` 指向新 baseline.json；可调 `codebase_max_steps_per_issue`。
4. **重跑 baseline**：用新 task 跑一次 stateless baseline 生成 baseline.json（否则 gain==reward）。
- 复用零改动：ACT/WRITE 循环、`downstream_improve_rewards`、GRPO 双流归一、metrics、prompts、interface 契约。

## 3. 镜像准备（承 env-plan）

- 转换在本地 NVMe `/scratch/qixinx/`，成品 SIF `cp` 回 `/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs/`（防丢）。
- 批量脚本：`instance_id → 官方 tag → apptainer pull → cp 回 /data`，`xargs -P 8` 并行。单个 ~56s，子集 41 题几分钟，全 500 题并行 ~1h / ~500G。
- 用户已定：**先只做子集跑通全链路**（如 astropy 22 + pytest 19）。

## 4. 建议执行顺序（每步可独立验证）

1. **批量拉子集镜像**（astropy+pytest）→ 存 /data。
2. **写 `build_swecl_dataset.py`** → 生成子集的 git_pr jsonl（先 Task-Stream 子集）。
3. **写 `swe_bench_cl/` task + loader + evaluator**（抄 codebase_adaptation 骨架 + 官方双判 evaluator）。
4. **clbench 直跑验证**：`clbench run --task swe_bench_cl --system icl`（一两题），确认起容器/判定/reward 通。
5. **跑子集 baseline** → baseline.json。
6. **改 miles schedule.py + config** → 指向新 task/数据/baseline。
7. **miles 冒烟**：小 cap 跑一个 episode，确认 ACT/WRITE/gain 通。
8. 通了再扩到全量镜像 + 全数据集。

## 5. 待决（拍板后落 plan）

- **reward 语义**：regret 式 vs 0/1（§1e）。
- **数据集**：Task-Stream(500) 还是 Curriculum(273) 先做（用户："两份都要，后面再定"——建议先 Task-Stream 子集，字段最省事）。
- **max_steps_per_issue**：SWE-bench 大 repo 题比 tablib/tenacity 难，40 步可能不够，要不要放宽。

## 5e. 实现设计（2026-07-08，3 个 Explore agent 摸清复用边界后定）

### 复用策略：继承 + 薄覆写（做法 B 的极致形态）
`class SweBenchCLTask(CodebaseAdaptationTask)`，只覆写 6 个接缝，其余（step/evaluate/reward/schedule/agent交互层）全继承。**task.py 预估 ~120-180 行**。

**5 个抽象方法**：`step`/`evaluate`/`build_current_query` 直接继承（纯流程，无 tablib/tenacity 硬编码）；只改 `name`(→"swe_bench_cl") 和 `build_canonical_run_state`(换 loader)。
**agent 交互层**（`_execute`/`_extract_patch`/`_make_issue_query`/`_next_query`/`_advance_to_next_issue`/mini-swe-agent 模板）全部继承。
**reward**：`_build_issue_instance_outcome` 的 `reward=1-(steps-1)/max_steps` + `r_max=1.0` 原样继承（决策：先用 regret 式，不改 0/1）。

### 6 个要覆写的接缝
1. `name` → `"swe_bench_cl"`。
2. `__init__` → family 字面量 3 处改 `"swe_bench_cl"`、`DEFAULT_DATASET_PATH` 改我们的 jsonl。
3. `build_canonical_run_state` → 换成自写 loader（见下）。
4. `_start_container` + `_initialize_generic_pr_workspace` → 官方镜像流程（放宽 `is_generic_pr_instance` 守卫 / 用官方镜像初始化）。
5. `_handle_submit` → 调自写 `evaluate_swe_bench_cl_submission`（带 node 预过滤）。

### 判定层：13 个 helper 直接 import，只 fork 1 个顶层函数
`generic_runtime` 的 13 个 helper（`_start_generic_container`/`initialize_generic_pr_container`/`_apply_patch_text`/`_docker_exec`/`strip_patch_paths`/`test_patch_paths`/`_restore_patch_files_to_base`/`build_test_command`/`exact_test_targets`/`instance_cwd`/`_cleanup_container` 等）+ container_backend 全部 **可直接 import**（无循环 import、无副作用）。
**只需 fork** `evaluate_generic_pr_submission` → `swe_bench_cl/evaluator.py::evaluate_swe_bench_cl_submission`（~30-40行）：编排骨架照抄，只把选 target 那步换成自写 `filter_collectable_targets`（用 import 的 `_docker_exec` 跑 `pytest --collect-only -q --continue-on-collection-errors <targets>`，解析可收集 node，剔除畸形）。
⚠️ 现有 `ordered_test_targets` 不能用：它不带 `--continue-on-collection-errors`，畸形 node 时反而 `return exact_targets` 放弃过滤，正好废掉我们要的效果。

### 数据 loader：自写薄 loader（不复用 load_tasks）
**不复用** `codebase_adaptation.load_tasks`：它 `same_repo=True` 默认在多 repo 时只挑一个 repo 丢其余；`same_repo=False` 又 shuffle 无视 `sequence_rank`。两个组合都不满足"跨多 repo + 固定顺序"。
**做法**：`from ..codebase_adaptation.task_loader import TaskInstance`（8 字段复用 dataclass），自写 `load_tasks` 按 jsonl 顺序读全部、不 shuffle、不按 repo 丢。

### 数据字段补齐（转换脚本，先于一切）
我们的 `swecl_clean.jsonl` 缺 5 个契约字段，写转换脚本补：
| 缺字段 | 填法 |
|---|---|
| `runtime_mode` | `"git_pr"`（`is_generic_pr_instance` 硬要求，否则 _start_container 抛 ValueError）|
| `problem_statement` | **从原始 Curriculum 按 instance_id join 回来**（清洗时丢了！agent 唯一题面，load_tasks 硬下标缺了即 KeyError）|
| `image_name` | `/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs/<instance_id>.sif`（绝对路径，以 `/` 开头 `.sif` 结尾→resolve 原样返回，不需 CLBENCH_SIF_DIR）|
| `test_command` | `/opt/miniconda3/envs/testbed/bin/python -m pytest --no-header -rA -p no:cacheprovider`（**绝对 python 路径，绕开 conda 激活**——`source ... && ` 会被 build_test_command 的 shlex.join 引号化破坏）|
| `sequence_rank` | `1..38`（curriculum 顺序，可选但建议）|

### 运行环境要求
- 必须 `CLBENCH_CONTAINER_BACKEND=singularity`（否则 _start_generic_container 走 docker 分支）。
- 无需 `CLBENCH_SIF_DIR`（image_name 用绝对路径）。

### 目录产物
```
src/tasks/swe_bench_cl/
├── task.py          # SweBenchCLTask(CodebaseAdaptationTask), ~120-180 行
├── task_loader.py   # 复用 TaskInstance + 自写 load_tasks(顺序读)
├── evaluator.py     # evaluate_swe_bench_cl_submission + filter_collectable_targets, ~60 行
└── schedules/default.json  # clbench 直跑/baseline 需要(miles 会覆写)
scripts/swecl/build_dataset.py  # swecl_clean + Curriculum problem_statement → 契约 jsonl
data/swe_bench_cl/subset.jsonl  # 转换产物(38题)
```

### 执行顺序（每步可独立验证）
1. **转换脚本** → 补 5 字段（尤其 join problem_statement）→ `data/swe_bench_cl/subset.jsonl`。
2. **evaluator.py**（fork + node 预过滤）→ 可单元测：对 astropy-7606 手动调，验证过滤 1 条后判定对。
3. **task_loader.py + task.py**（继承 + 6 覆写）。
4. **schedules/default.json**（38 题顺序）。
5. **clbench 直跑 1-2 题**：`CLBENCH_CONTAINER_BACKEND=singularity clbench run --task swe_bench_cl --system icl`，验证起容器/判定/reward 通。
6. baseline → miles（承 §2/§4）。

## 5f. 实现进度（2026-07-08，已动手）

**已完成 + 已验证**（走 uv run + singularity 真实链路）：
- **父类 refactor**：codebase_adaptation/task.py 加类属性 `TASK_FAMILY="codebase_adaptation"`，替换 3 处 family 字面量 + name 属性。行为不变，已验证 codebase_adaptation 仍正常。
- **子类** `src/tasks/swe_bench_cl/task.py`（~90 行）：`SweBenchCLTask(CodebaseAdaptationTask)`，`TASK_FAMILY="swe_bench_cl"` + `r_max=1.0`（register 查 `__dict__` 不走继承，必须重声明）+ dataset 默认 + setup 覆写。step/evaluate/reward/schedule/agent交互层全继承。
- **`__init__.py`** + **`schedules/default.json`**（astropy 20 → pytest 18 两 stage，notify_change 换 repo 通知）。
- **数据管线**：`scripts/swecl/build_dataset.py`（join problem_statement + 补 5 字段）+ `collect_filter_one.sh`/`batch_collect_filter.sh`（数据层剔除畸形 node，共剔 82 条/16 题）。产物 `data/swe_bench_cl/subset.jsonl`（38题）。
- **判定采纳用户建议改为数据层过滤**：不写 evaluator.py、不做运行时过滤，直接复用 `evaluate_generic_pr_submission`（git_pr 实例自动路由）。更简单更安全（模型搞崩正常 P2P 会正确判失败，不被静默丢弃）。
- **端到端判定验证**：`CLBENCH_CONTAINER_BACKEND=singularity` 下，astropy-12907 gold→success、空patch→fail；7606（剔除畸形node后240 P2P）gold→success。全走真实 clbench 判定代码。

**关键实现坑（已解决，记录备查）**：
- register_task 查 `cls.__dict__` 的 r_max（不走继承）→ 子类须重声明 `r_max=1.0`。
- test_command 用 `/opt/miniconda3/envs/testbed/bin/python -m pytest -p no:cacheprovider`：`--no-header`/`-rA` 是 pytest 6.0+，老 repo(astropy 3.x=pytest 3.3.1)不认。跨版本只用 `-p no:cacheprovider`。
- 数据层 node 过滤要**按文件 collect**（不按 node 选择）：`--continue-on-collection-errors` 只管文件收集错误，不管 node not-found；按 node 会被畸形 node abort。且提取文件时只取含 `::` 的真 node（`[100%]` 这种假条目会当文件名 abort）。
- 环境用 `uv run`（jinja2 等依赖在 uv 环境，裸 python3.9 缺）。

**环境一致性修复（2026-07-08，用户要求对齐 codebase_adaptation）**：
- **问题**：codebase_adaptation 的 agent 进容器即就绪(裸 `python`/`pytest` 直接可用，依赖装在系统 python)；我们的官方 SWE-bench 镜像 agent 却落在 base conda py3.11、`pytest` 不在 PATH，真环境在 conda `testbed`(py3.9) 未激活 → agent 白费步数找环境(smoke 里 opus astropy 第4-15步都在跟环境搏斗)。
- **根因**：mini-swe-agent 用非登录 `bash -c` 执行命令，不 source `/root/.bashrc`(镜像里那句 `conda activate testbed` 不触发)；`--home /root` 靠 login shell 激活的路子在真实 sandbox 路径下无效(实测 py3.11)。
- **修法(实测有效)**：注入 `--env PATH=/opt/miniconda3/envs/testbed/bin:...`(testbed 优先 + base conda + 标准目录)。所有官方 SWE-bench 镜像 testbed 路径统一，故通用。agent 落在 python 3.9.20 @ testbed、裸 pytest 可用、git/sed 正常——与 codebase_adaptation 对齐。
- **实现**：`swe_bench_cl/task.py.__init__` 里 `os.environ.setdefault("CLBENCH_SINGULARITY_EXEC_ARGS", ...)`。scoped to swe_bench_cl(实例化时设、setdefault 让用户显式覆盖优先)；codebase_adaptation 用不同镜像+不实例化本 task，不受影响。判定不受影响(判定本就用绝对 testbed python)。

**完整 agent 验证（opus API，2题 smoke，已跑两次对照）**：
- 环境修复前后对照(同题 astropy-12907)：**修前 17 步/reward 0.6，~10+步在跟 conda/numpy 搏斗**；**修后 6 步/reward 0.875，零环境折腾**，总交互 25→11。证明 env parity 修复真实生效、rollout 更干净。
- pytest-10051 两次都失败(8步→5步)，`eval_status=tests_failed`——opus 修复不完整，模型问题非链路问题，判定正确。
- 换 repo NOTICE：trace 里未出现——但**codebase_adaptation 自己的 run(234交互)也是 0**，两边一致(我全继承其 prompt 代码)。属训练数据/schedule 机制范畴，以后深究。

**已完成的里程碑**：格式转换 + task 打通 + 环境对齐 + prompt 一致，全部验证。

**未完成(用户明确要"以后再说"的训练数据构造)**：
- 序列怎么切 / 一条 episode 多少题 / schedule 设计 / 接 miles。**不由我决定**。
- 我只负责：把全量数据的环境处理好(拉镜像+清洗+转格式)。

## 6. 关键文件索引
- clbench task 基类：`src/interface.py:359`；注册：`src/registry.py:62`；发现：`registry.py:24`。
- 抄骨架源：`src/tasks/codebase_adaptation/{task.py,task_loader.py,evaluator.py,generic_runtime.py}`。
- 容器后端：`src/tasks/container_backend.py`（`resolve_singularity_image:90`）。
- miles 契约：`miles/examples/codebase_adaption/{codebase_rollout.py,schedule.py,codebase_config.yaml}`。
- 环境验证：`notes/swe-bench-cl-env-plan.md`。

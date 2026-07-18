# SWE-Bench-CL → clbench 集成：定稿存档

> 定稿日期 2026-07-09。工作草稿见 `swe-bench-cl-integration-plan.md` / `swe-bench-cl-env-plan.md`，
> 本文是最终结论与可复现流程。

## 1. 最终产物

- **数据集**：`data/swe_bench_cl/full.jsonl` — **240 题**，clbench `git_pr` 契约。
- **task**：`src/tasks/swe_bench_cl/task.py`，`SweBenchCLTask(CodebaseAdaptationTask)`，注册名 `swe_bench_cl`。
- **官方判分**：`src/tasks/swe_bench_cl/official_grading.py`（仅 django/sympy 用）。
- **SIF 镜像**：`/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs/*.sif`（绝对路径写进 image_name）。

**240 题 repo 分布**：sympy 50 / django 49 / sphinx-doc 33 / scikit-learn 32 / matplotlib 25 /
astropy 17 / pytest-dev 17 / pydata(xarray) 17。

## 2. 判分路由（`_evaluate_submission` 按 repo 分流）

- **django/sympy**（不走 pytest：runtests.py / bin/test）→ 官方 SWE-bench 日志解析判分
  （跑 repo 的 test_cmd + directives，解析 log，比对 F2P/P2P，RESOLVED_FULL 才算解出）。
- **其余 6 个 pytest repo** → 父类 `evaluate_submission`（returncode 判分，退出码 0 即全过）。
- `_OFFICIAL_GRADING_REPOS = {"django/django", "sympy/sympy"}`。

## 3. 环境 parity（**底线：agent 落地环境必须 == codebase_adaptation**）

singularity exec 参数 = clbench 默认 base **一字不差** + 两个 `--env` 补偿镜像差异：

```
--contain --cleanenv --fakeroot --no-mount hostfs,bind-paths   # 与 codebase_adaptation 完全相同
--env PATH=/opt/miniconda3/envs/testbed/bin:...                # SWE-bench 镜像 python 藏在 testbed conda，注入追平
--env LANG=C.UTF-8                                             # SWE-bench 镜像没烤 LANG（codebase_adaptation 烤了），注入追平
```

**实证并排探测（同一 bash -c 非登录 shell）**：

| 维度 | codebase_adaptation(默认参数) | swe_bench_cl(最终参数) | 一致 |
|---|---|---|---|
| python/pytest 就绪 | ✓ /usr/local/bin | ✓ testbed/bin | ✓ |
| LANG / stdout 编码 | C.UTF-8 / utf-8 | C.UTF-8 / utf-8 | ✓ |
| localhost 解析 | 失败(无 /etc/hosts) | 失败 | ✓ |

- 两个 `--env` 不给 agent 额外能力，只把镜像差异抹平到同一就绪状态。
- 一度为修 matplotlib 判分而恢复 `bind-paths`（localhost 可解析），这**违反 parity**，已改回。
- **副作用**：localhost 两边都坏 → 依赖本地 socket 的测试（matplotlib rerunfailures 插件、
  sphinx linkcheck）无法跑；判分命令加 `-p no:rerunfailures` 绕过插件崩溃，真正依赖 localhost
  的测试题在第 5 步被筛除。

## 4. 数据处理流水线（可复现，脚本在 `scripts/swecl/`）

1. **抽取** `extract_instances.py`：从 Curriculum 抽题（problem_statement/难度/version 等元数据）。
2. **清洗**（gold 双向验证，clean 拷 /data，dirty 删）：
   - pytest repo：`pull_and_validate_one.sh` + `batch_pull_validate.sh`（returncode 判定）。
   - django/sympy：`clean_official.py`（官方 grading，退出码判定对它们无效）。
3. **畸形 node 过滤** `collect_filter_one.sh` + `batch_collect_filter.sh`：容器内 collect-only 出
   pytest 真能选中的 node，剔除畸形（参数化空串 `test_x[]`、`[100%]` 进度伪条目等）。产物
   `swecl_pytest_collectable.jsonl`（151 pytest 题共剔 362 条畸形 node）。**只对 pytest repo**；
   django/sympy 官方解析按 test-id 匹配，不需要。
4. **建数据集** `build_dataset.py`：补齐 git_pr 契约字段 + version + 绝对 SIF 路径 +
   `-p no:cacheprovider -p no:rerunfailures` 的 test_command；用 collectable map 过滤 node；
   F2P 过滤后为空的题跳过（node id 被 SWE-Bench-CL 截断，如 `test_unparse[(1,`）。
5. **全量 gold 判分门槛** `verify_all.py`：**对每一题走真实判分路由跑 gold patch，断言每个
   F2P fail→pass、每个 P2P 保持 pass**（success=True 即逐 test 保证）。过不了的题剔除
   （= codebase_adaptation 的 screening 同逻辑）。写进 `build_dataset.py` 的 `SCREEN_EXCLUDE`。

## 5. 全量验证结果

**240/240 通过**（每题每个 test 在 gold patch 下都过）。
- 248 题并发(par=6)跑 → 237 PASS（PASS 不因并发造假，可信）。
- 11 FAIL 串行(par=1)重验 → 3 个 django 是并发瞬时误伤（reclaim），8 个真失败。
- 240 = 237 + 3。

**被 `SCREEN_EXCLUDE` 剔除的 8 题**：

| 题 | 原因 |
|---|---|
| sphinx-8269 / 8475 | linkcheck 测试需 localhost；parity 环境无 /etc/hosts（与 codebase_adaptation 一致） |
| astropy-8707 / 8872 / 7336 | 镜像 pytest 版本漂移，nose/distutils 弃用警告被 `filterwarnings=error` 升级，P2P 在此镜像必错 |
| pytest-7324 | 畸形 node `[a:::c]` pytest 选不中 |
| matplotlib-25122 / pydata-4687 | 节点数 >2000，命令超单参数 128KB 上限（MAX_ARG_STRLEN） |

另有 2 题在第 4 步因 F2P 过滤后为空剔除：sphinx-8621 / sphinx-8265（node id 被截断）。

## 6. 对共享代码 codebase_adaptation 的改动（行为保持）

`src/tasks/codebase_adaptation/task.py`：
- 加类属性 `TASK_FAMILY = "codebase_adaptation"`，3 处 schedule/variant 查找 + name 用 `self.TASK_FAMILY`（值不变）。
- `_handle_submit` 判分改成可覆盖钩子 `self._evaluate_submission(...)`，默认实现体等价于原 `evaluate_submission(...)`。
- 子类只覆盖 `TASK_FAMILY` / env 参数 / `_evaluate_submission` 路由 / `setup` 预检。

## 7. 关键坑（已解决）

- django ellipsis UnicodeEncodeError：`--cleanenv` 清了 LANG → ascii stdout；`LANG=C.UTF-8` 修复。
- matplotlib rerunfailures INTERNALERROR：插件绑 localhost socket，parity 环境无 localhost；判分 `-p no:rerunfailures`。
- 并发建 sandbox 瞬时报错：`clean_official.py` / `verify_all.py` 对 error/timeout 重试一次；真 tests_failed 不重试。
- 命令超 128KB：节点极多的题（>2000）触发，已在 SCREEN_EXCLUDE 剔除；存活题最大 ~110KB(pydata-4075) 安全。

## 8. 待办 / 下一步

- 用 qwen3.5-9b 在 240 题上评测（每题拟跑 4 遍），看表现。
- 训练课程编排（sequence/episode/schedule/miles 接入）由用户负责，本集成只保证环境+判分正确。

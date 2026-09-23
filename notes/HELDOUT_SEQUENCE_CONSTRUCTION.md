# Held-out long-sequence eval data (SWE-smith, 2026-09-20)

目的:在**训练没见过的 repo** 上造**比 clbench 那条 9+10=19 题更长**的 CL sequence,
用来看后段(late instances)的表现差异 —— AgentCL 那条路的任务形态 gap 太大,测不到这个。

## 1. 池子与 repo 选择

原始链路与 `SWESMITH_DATA_CONSTRUCTION.md` 完全一致(parquet → `build_pool.py` 五道过滤)。
重建结果:**23,583 题 / 126 repo**;训练用掉 top 53(19,227 题),故 **73 个 repo / 4,356 题从未进过训练**。

- parquet:HF `SWE-bench/SWE-smith`(11 shard,265MB)→ `/scratch/qixinx/swesmith_parquet/data`
- 池子:`/data/user_data/qixinx/swe_smith/pool_v2/{pool.jsonl,repo_stats.tsv}`
- 拉了 12+ 个 held-out repo 的 SIF(`docker://` 用池子里的 `image_name` 字段,即
  `jyangballin/swesmith.x86_64.<owner>_1776_<repo>.<commit8>`),与原 53 个同目录同命名。
- `extract_test_patches.py` 对 15/16 个 repo 100% 成功;**autograd 失败**(rc=128,
  `bad revision 'origin/...pr_579'`:它有 PR 形态 instance,镜像里没有对应 origin 分支),其 116 题被跳过。
- registry:`heldout_v4/heldout_registry.jsonl` = **1,668 题 / 16 repo**(generic PR runtime 格式)。

## 2. 段长的硬约束(实测)

`build_segments` 要求段内**桶内策略不重复**,而 hard 桶只有 4 种 strategy
(lm_rewrite / remove_loop / combine_module / remove_wrapper),所以单段一旦超过 6 题,产量急剧下降:

| 段配方 | 段长 | 段数(16 repo) |
|---|---|---|
| 2易2中2难(训练用) | 6 | 78 |
| 4易3中2难 | 9 | 21 |
| 4易6中3难 | 13 | 10 |
| 5易5中3难 | 13 | 6 |
| 8易8中8难 | 24 | 0 |

拉长 horizon 的正确旋钮是**一条 episode 串更多 repo**(`--repos-per-episode`),而不是单段更长。

## 3. 判分验证(新增,held-out repo 必做)

`verify_heldout_grading.py` 对每题跑两次**真实判分链**(`evaluate_submission`):
gold patch(取自 parquet)应判过;只新增无关文件的 no-op patch 应判不过
(不用空 patch:判分链对空 patch 直接短路成 `empty_patch`,不进容器)。

两轮共验 300 题:第一轮 104 题(ok 93 / gold 失败 11),第二轮 196 题(ok 193 / gold 失败 3);
**两轮 no-op 误过都是 0**,说明判分链本身没问题,不存在"不修也能过"的题。

失败根因只有两类,都是**环境相关测试**,不是题目坏:

| 恒失败测试 | F2P 命中 | P2P 命中 | repo | 现象 |
|---|---|---|---|---|
| `test_runspider_dnscache_disabled` 等 | 5 | 23 | scrapy | 容器内无外网 → `twisted.internet.error.DNSLookupError` |
| `BenchCommandTest::test_run` | 0 | 28 | scrapy | 同上 |
| `TestDeprecations::test_import_cli` | 0 | 25 | jsonschema | 断言 warning filename == `importlib.__file__`,本环境解析成 `/testbed/...` |

外加 gspread 1 题实测 gold 判不过(根因未查,疑似同类网络依赖)。

**最关键的一条教训:这些坏测试大多藏在 P2P 里,不在 F2P 里。** 判分要求 P2P 保持全绿,
所以 P2P 里有一个环境相关的恒失败测试,这题就永远判不过。只按 F2P 名字做静态过滤会漏掉大半
(scrapy 10 道失败题里只有 5 道能靠 F2P 扫出来)。→ **held-out repo 必须实测判分,不能靠静态规则。**

当年训练那 53 个 repo 验过判分,所以从没踩到这类坑。

不可判分清单(57 题 = 静态扫 F2P∪P2P 的 53 题 ∪ 两轮实测失败题):
`/home/qixinx/heldout_prep/ungradable_all.txt`,已从 registry 中剔除;
逐题报告 `/home/qixinx/heldout_prep/{verify_picks,verify_v5}.jsonl`。

## 4. 最终数据(2026-09-20 清理后,`/data/user_data/qixinx/swe_smith/heldout_v4/`)

含任何不可判分题的文件**已全部删除**(v4 那一整批 9 个 + 旧 v5_3x9 + 两个旧 pick,共 12 个),
现存 5 个文件都经过复查、零黑名单题:

| 文件 | 内容 | 判分验证 |
|---|---|---|
| `FINAL_2x13_26issues.jsonl` | **交付**:2 条 × 26 题(2 repo × 13) | 52/52 通过 |
| `FINAL_3x9_27issues.jsonl` | **交付**:2 条 × 27 题(3 repo × 9) | 54/54 通过 |
| `episodes_v5_2x13.jsonl` | 13+13 母池,4 条(全绿) | 104/104 通过 |
| `episodes_v6_3x9_clean.jsonl` | 9+9+9 母池,6 条(用干净 registry 重生成) | 生成时已排除黑名单 |
| `heldout_registry.jsonl` | **1,611 题 / 15 repo**(原 1,668 剔掉 57) | 组装的唯一入口 |

四条交付 sequence 的 repo 构成(训练全未见过,每条内部不重复):

- 26 题:`bleach + charset_normalizer`;`flake8 + line_profiler`
- 27 题:`alive-progress + bleach + line_profiler`;`flake8 + sqlparse + soupsieve`

对照 clbench 那条 9+10=19 题:26 题 = 1.37×,27 题 = 1.42×;两者长度接近而域切换次数不同
(1 次 vs 2 次),可分离"序列更长"与"切换更多"两个因素。

注意两点:

- `episodes_v6_3x9_clean.jsonl` 是清理后**重新生成**的,它的 6 条与 `FINAL_3x9_27issues.jsonl`
  里那两条不是同一批(FINAL 的 `episode_index` 1/2 指向已删除的 v5 文件)。FINAL 文件自包含且验证过,
  不受影响;要扩条数就从 v6 池子里挑并**重新验证新增的题**。
- registry 已就地换成干净版。真要回溯被剔的题,`pool_v2/pool.jsonl` 与 `heldout_v4/test_patches/`
  都还在,可重建。

**以后组装的标准姿势**(段配方与 repo 数按需改):

```bash
python3 scripts/swesmith/gen_episodes_v4.py \
  --instances /data/user_data/qixinx/swe_smith/heldout_v4/heldout_registry.jsonl \
  --parquet-dir /scratch/qixinx/swesmith_parquet/data \
  --repos-file /home/qixinx/heldout_prep/repos_gradable.txt \
  --exclude-instances /home/qixinx/heldout_prep/ungradable_all.txt \
  --plan easy:4,mid:3,hard:2 --repos-per-episode 3 --out <新文件>
```

新组的 sequence 里只要有**没验过的题**,就先跑一遍 `verify_heldout_grading.py` 再用。

## 5. babel 侧踩到的两个坑

- 判分链要 `MSWEA_SINGULARITY_EXECUTABLE=apptainer`(系统里没有 `singularity` 这个名字)。
- 必须 `unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH`:
  题目镜像以 `--writable` 启动且内部没有 `/data`,继承的 bind 会让子容器创建失败。
  (`run_codebase_adaption_qwen3.5_4B.sh` 出于同样原因 unset 它们。)

脚本:`scripts/swesmith/{build_heldout_pool_babel.sh,pull_heldout_sifs_{babel,list_babel}.sh,
prep_heldout_registry_babel.sh,gen_episodes_v4.py,pick_episodes.py,verify_heldout_grading.py}`;
miles 侧内层脚本 `examples/codebase_adaption/scripts/swesmith_verify_inner_babel.sh`。

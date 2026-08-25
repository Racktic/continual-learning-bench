# SWE-smith Training Data Construction for Continual-Learning RL

*2026-07-23 · pipeline code: `scripts/swesmith/` · data: `data/swe_smith/`*

## 1. Motivation

Our previous RL training pool (SWE-Bench-CL, 240 instances / 8 repos) was too
small for multi-epoch RL: trajectory analysis showed a median of **32 exposures
per instance** over one training run, and a five-test battery confirmed weight
memorization signatures (answer recall without in-context evidence). The goal
of this dataset is a pool large enough that **every training episode is seen at
most once**, while preserving the structure our continual-learning setup needs:
multi-instance sequences with a controlled difficulty gradient and an explicit
memory hand-off between instances.

## 2. Source dataset

[SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) provides 50.9K
synthetic bug-fixing instances over 131 Python repositories. Each instance is a
git **branch** of a per-repo Docker image: checking out the branch activates
exactly one synthetic bug, produced by a named corruption strategy
(`func_basic`, `func_pm_ctrl_shuffle`, `lm_rewrite`, `combine_module`, …).
The branch also **deletes the fail-to-pass (F2P) test files** as an anti-cheat
measure; grading restores the tests and runs them.

## 3. Instance pool filtering (50.9K → 23,583)

`build_pool.py` applies five filters:

| # | Filter | Rationale |
|---|--------|-----------|
| 1 | F2P count ∈ [1, 20] | 0 = ungradable; >20 = mega-bugs with diffuse supervision |
| 2 | localized P2P ∈ [3, 500] | P2P (pass-to-pass) tests restricted to files adjacent to the F2P tests; a floor of 3 guarantees a regression constraint (P2P=0 instances — e.g. snapshot-test clusters — are exploitable), a cap of 500 bounds grading cost |
| 3 | patch touches 1–3 files | keeps the bug localized |
| 4 | problem statement ≥ 100 chars | 22.5% of upstream instances ship without issue text (issue generation is a separate LLM step upstream); those are unusable as tasks |
| 5 | exclude `jd__tenacity` | contamination with our held-out eval repo |

Result: **23,583 instances / 127 repos** (`pool_v1/pool.jsonl`).

## 4. Repo selection and runtime images

We rank repos by post-filter instance count and take the **top 53** (top-33
first batch + next-20 second batch), covering **19,227 instances**. Each repo's
Docker image is converted once to an Apptainer SIF (~1.3 GB each, 69 GB total;
mirrored at `hf.co/datasets/Racktic/swesmith_sifs` together with the instance
registry).

## 5. Grading protocol and test-patch extraction

Because bug branches delete the F2P test files, grading needs a `test_patch`
that restores them. `extract_test_patches.py` computes, per instance,

```
test_patch = git diff <bug-branch> main -- <F2P ∪ localized-P2P test files>
```

inside the repo's own container (batched per repo; one upstream path quirk —
`../dev/`-prefixed test ids in `mido` — is normalized so pytest ids resolve
from the standard workdir). Grading then: restore tests via `test_patch` →
run F2P (must flip to green) and P2P (must stay green), verdict by pytest
return codes. All 19,227 extracted patches are non-empty.
`convert_pool.py` assembles the final registry (`top53.jsonl`) in our generic
PR-runtime format.

## 6. Anti-reward-hacking: git history neutralization

The bug branch's git history contains the *inverse* bug patch in plain sight
(`git log -p` reveals the fix). In a 1,912-trial audit, ~10% of agent
trajectories displayed the bug patch in their observations (0% exploited it,
but the channel is open). Before RL we therefore neutralize the workspace
inside the agent's sandbox: squash to a single orphan commit, delete all other
branches and the origin remote. The grading container is separate and
unaffected. A post-fix audit over 768 training trials found **zero** bug-patch
sightings.

## 7. Empirical difficulty calibration

Heuristic difficulty tiers shipped with SWE-smith turned out to be unreliable,
so we measured difficulty directly: an **eval-only pass@k campaign** with the
base model (Qwen3.5-4B, official eval decoding limits, avg@2 over 2,391
trials on 3 pilot repos) yields per-strategy pass@1:

| bucket | strategies (pass@1) |
|--------|---------------------|
| easy (≥25%) | class_rm_base 75 · ctrl_shuffle 42 · func_basic 40 · op_swap 33 · op_change 32 · op_change_const 30 |
| mid (10–25%) | invert_if 23 · combine_file 16 · remove_assign 13 · remove_cond 11 · class_rm_funcs 11 |
| hard (<10%) | lm_rewrite 8 · remove_loop 4 · combine_module 1 · remove_wrapper 0 |

Strategy-level pass rate is used as the difficulty proxy for all repos
(instance-level rates exist for the pilot repos). Overall base performance:
pass@1 26.8%, pass@2 33.8%.

## 8. Episode (sequence) construction

A training **episode** is a 6+6 sequence: two repos, six instances each,
ordered easiest→hardest within each repo segment (by strategy pass rate). The
agent writes a memory after every instance; the memory is the only channel
across instances, which is what makes the sequence a continual-learning unit.

De-duplication constraints (SWE-smith mints 3–4 bug variants per function —
19,227 instances collapse onto only ~5,700 distinct functions):

- an instance appears in **at most one episode** across the whole dataset;
- **function cap = 2**: at most two instances of the same function overall,
  and they must use different corruption strategies;
- within a segment, no repeated function;
- repo pairing is **top-k random** (k=5 among repos with the most remaining
  segments): near-maximal yield with diverse repo pairings (162/110/117
  distinct pairs across the three tiers).

## 9. Three-tier curriculum (final dataset, v3)

Generated by `gen_episodes_v3.py` (fixed seed, fully reproducible):

| tier | segment recipe | episodes | instances | expected pass@1 |
|------|----------------|----------|-----------|-----------------|
| `pure_easy` | 6 × easy | 63 | 756 | ~35% |
| `easy_mid` | 3 easy + 3 mid | 138 | 1,656 | ~25% |
| `mixed` | 2 easy + 2 mid + 2 hard | 181 | 2,172 | ~18% |
| **total** | | **382** | **4,584** | |

Zero instance overlap across tiers; 53 repos all participate. Concatenated in
tier order and consumed **sequentially** (dataset shuffling disabled), this
yields a step-level curriculum: the run warms up on dense-reward easy episodes
(also the regime with the fewest zero-variance GRPO groups) before moving to
harder mixtures. With 2 episodes per rollout and 120 rollouts, a full run
draws 240 episodes — **every episode is seen at most once**, eliminating the
repetition that drove memorization in the previous pool.

The tier sizes reflect a measured trade-off frontier: the tiers compete for
the shared easy/mid instance budget under the function cap, and pure-easy is
structurally capped at ~63 by the requirement of six distinct easy strategies
from a single repo.

## 10. Artifacts

| artifact | location |
|----------|----------|
| pipeline + generators | `scripts/swesmith/` (this repo, branch `main`) |
| episodes (all tiers + concatenated) | `data/swe_smith/episodes_v3_*.jsonl` |
| instance registry (1.1 GB, embeds test patches; SIF paths must be rewritten per cluster) | `hf.co/datasets/Racktic/swesmith_sifs` → `registry/top53.jsonl` |
| container images (53 SIFs, 69 GB) | `hf.co/datasets/Racktic/swesmith_sifs` |
| pass@k difficulty tables | `/project/flame/qixinx/swe_smith/passk_v1/` (orchard) |
| task runtime (activation, grading, git neutralization) | `src/tasks/swe_smith/task.py` |

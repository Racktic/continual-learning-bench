# SWE-Bench-CL Dataset Viewer

Single-file, interactive viewer for showing the dataset (advisor / collaborators).

- **`dataset_viewer.html`** — self-contained page (inline CSS + JS, no external assets, ~550 KB).
  Open directly in a browser. Two tabs:
  - **Issues by repo** — pick a repo, browse all its issues, click any issue to read the full
    problem statement, modified files, FAIL→PASS / PASS→PASS test counts, difficulty, qwen baseline.
  - **CL sequences** — pick any of the 54 sequences; bar height = combined difficulty (easy→hard
    within each repo block), dashed line = repo switch; click a bar to open that issue.
- **`dataset_view.json`** — the embedded data (all 231 issues + 54 sequences).
- **`gen_viewer.py`** — reproducible generator (repo-relative paths).

Online (private) artifact: https://claude.ai/code/artifact/ac73925e-f8eb-4bea-aa98-55f88fd3a573

## Regenerate

```bash
python notes/dataset_viewer/gen_viewer.py
```

Reads `data/swe_bench_cl/train_pool_avg0.6.jsonl` (231-issue training pool) + the miles
`swecl_train_episodes.jsonl` (54 episodes), rewrites `dataset_view.json` and `dataset_viewer.html`.

- Episode path defaults to `/home/qixinx/miles/...`; override with
  `SWECL_EPISODES=/path/to/episodes.jsonl python notes/dataset_viewer/gen_viewer.py`.
- `STMT_CAP` caps each embedded problem statement (default 8000 chars).
- Difficulty formula lives in `combined()`.

## Curriculum design

See `notes/swe-bench-cl-sequence-design.md` for the full rationale (inter-group knowledge-family
pairing + intra-group difficulty curriculum).

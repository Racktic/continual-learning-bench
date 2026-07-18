#!/usr/bin/env python3
"""Regenerate the interactive SWE-Bench-CL dataset viewer (dataset_viewer.html).

Self-contained + repo-relative. Embeds every issue in the training pool (full
problem statements) and every synthesized CL sequence, then renders one
interactive, self-contained HTML page (inline CSS + JS, no external assets):

  * "Issues by repo" tab  — pick a repo, browse its issues, click an issue to
    read the full problem statement, modified files, tests, difficulty, baseline.
  * "CL sequences" tab    — pick any of the 9+10 sequences; bar height encodes
    combined difficulty (easy->hard within each repo block); click a bar to open
    the same issue detail.

Run from the clbench repo root:
    python notes/dataset_viewer/gen_viewer.py
"""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
POOL = os.path.join(REPO, "data/swe_bench_cl/train_pool_avg0.6.jsonl")
EPS = os.environ.get(
    "SWECL_EPISODES",
    "/home/qixinx/miles/examples/codebase_adaption/data/swecl_train_episodes.jsonl",
)
STMT_CAP = 8000

FAMILY = {"astropy/astropy": "A", "pydata/xarray": "A", "scikit-learn/scikit-learn": "A",
          "matplotlib/matplotlib": "A", "pytest-dev/pytest": "B", "sphinx-doc/sphinx": "B",
          "django/django": "B", "sympy/sympy": "bridge"}
SHORT = {"astropy/astropy": "astropy", "pydata/xarray": "xarray",
         "scikit-learn/scikit-learn": "scikit-learn", "matplotlib/matplotlib": "matplotlib",
         "pytest-dev/pytest": "pytest", "sphinx-doc/sphinx": "sphinx",
         "django/django": "django", "sympy/sympy": "sympy"}
ESSENCE = {"astropy": "numpy science + units/IO", "xarray": "labeled N-d arrays",
           "scikit-learn": "estimator API + numerics", "matplotlib": "rendering / figure state",
           "pytest": "test-framework core", "sphinx": "docs framework (extensions/builders)",
           "django": "ORM + web framework", "sympy": "symbolic expression trees + printing"}
FAM_ORDER = {"A": 0, "B": 1, "bridge": 2}


def combined(r):
    """Combined difficulty in [0,1]: 0.5 human (difficulty_score) + 0.5 qwen empirical."""
    s = r.get("difficulty_score")
    human = ((float(s) - 1) / 3) if isinstance(s, (int, float)) else 0.5
    q = 1 - (r.get("baseline_qwen35_9b") or {}).get("solved4", 0) / 4
    return round(0.5 * human + 0.5 * q, 3)


def extract():
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in open(POOL)}
    issues = {}
    counts = {}
    for iid, r in rows.items():
        short = SHORT[r["repo"]]
        b = r.get("baseline_qwen35_9b") or {}
        counts[short] = counts.get(short, 0) + 1
        issues[iid] = {
            "id": iid, "repo": short, "family": FAMILY[r["repo"]],
            "difficulty": r.get("difficulty"), "score": r.get("difficulty_score"),
            "avg4": b.get("avg4"), "solved4": b.get("solved4"), "combined": combined(r),
            "files": r.get("modified_files") or [],
            "f2p": len(r.get("FAIL_TO_PASS") or []), "p2p": len(r.get("PASS_TO_PASS") or []),
            "created": (r.get("created_at") or "")[:10], "commit": (r.get("base_commit") or "")[:10],
            "statement": (r.get("problem_statement") or "").strip()[:STMT_CAP],
        }

    repos = []
    seen = set()
    for full in sorted(SHORT, key=lambda f: (FAM_ORDER[FAMILY[f]], -counts.get(SHORT[f], 0))):
        short = SHORT[full]
        if short in seen:
            continue
        seen.add(short)
        repos.append({"short": short, "family": FAMILY[full],
                      "essence": ESSENCE[short], "count": counts.get(short, 0)})

    eps = [json.loads(l) for l in open(EPS)]
    sequences = []
    for e in eps:
        m = e["metadata"]
        sequences.append({"idx": m["episode_index"], "repoA": m["repoA"], "repoB": m["repoB"],
                          "ids": m["instance_ids"], "stages": m["stage_labels"]})

    return {"issues": issues, "repos": repos, "sequences": sequences,
            "totals": {"pool": len(rows), "repos": len(repos), "episodes": len(eps)}}


def main():
    D = extract()
    json.dump(D, open(os.path.join(HERE, "dataset_view.json"), "w"), ensure_ascii=False)
    data_js = json.dumps(D, ensure_ascii=False).replace("</", "<\\/")
    out = _TEMPLATE.replace("/*__DATA__*/", data_js)
    open(os.path.join(HERE, "dataset_viewer.html"), "w").write(out)
    print(f"wrote dataset_viewer.html  ({D['totals']['pool']} issues, "
          f"{D['totals']['episodes']} sequences)")


_TEMPLATE = r"""<title>SWE-Bench-CL Dataset Viewer</title>
<style>
  :root{
    --bg:#f3f4f7; --panel:#ffffff; --panel2:#f8f9fb; --ink:#1a2030; --muted:#606a7e; --line:#e3e6ec;
    --accent:#4f46e5; --famA:#0f9488; --famA-bg:#e4f3f0; --famB:#b56209; --famB-bg:#fbeedd;
    --famBr:#7c48d6; --famBr-bg:#efe9fb; --shadow:0 1px 2px rgba(20,26,40,.05),0 10px 30px rgba(20,26,40,.06);
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0c1016; --panel:#151b24; --panel2:#111721; --ink:#e6eaf1; --muted:#8a94a6; --line:#252d3a;
    --accent:#8b83f5; --famA:#2ed4c0; --famA-bg:#0c2b28; --famB:#f0a24d; --famB-bg:#2a2013;
    --famBr:#a98cf7; --famBr-bg:#211a38; --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px rgba(0,0,0,.35);
  }}
  :root[data-theme="light"]{
    --bg:#f3f4f7; --panel:#ffffff; --panel2:#f8f9fb; --ink:#1a2030; --muted:#606a7e; --line:#e3e6ec;
    --accent:#4f46e5; --famA:#0f9488; --famA-bg:#e4f3f0; --famB:#b56209; --famB-bg:#fbeedd;
    --famBr:#7c48d6; --famBr-bg:#efe9fb; --shadow:0 1px 2px rgba(20,26,40,.05),0 10px 30px rgba(20,26,40,.06);
  }
  :root[data-theme="dark"]{
    --bg:#0c1016; --panel:#151b24; --panel2:#111721; --ink:#e6eaf1; --muted:#8a94a6; --line:#252d3a;
    --accent:#8b83f5; --famA:#2ed4c0; --famA-bg:#0c2b28; --famB:#f0a24d; --famB-bg:#2a2013;
    --famBr:#a98cf7; --famBr-bg:#211a38; --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);line-height:1.5;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1120px;margin:0 auto;padding:40px 24px 80px}
  code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  .eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:700}
  h1{font-size:29px;line-height:1.2;margin:8px 0 8px;letter-spacing:-.015em;text-wrap:balance}
  .lede{color:var(--muted);font-size:15px;max-width:76ch;margin:0}
  .totals{display:flex;gap:26px;margin:22px 0 8px;flex-wrap:wrap}
  .stat{display:flex;flex-direction:column}
  .stat b{font-size:25px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
  .stat span{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
  .tabs{display:flex;gap:6px;margin:26px 0 20px;border-bottom:1px solid var(--line)}
  .tab{appearance:none;background:none;border:0;padding:10px 16px;font-size:14px;font-weight:600;color:var(--muted);
    cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
  .tab.active{color:var(--accent);border-bottom-color:var(--accent)}
  .tab:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:6px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}
  .chip{appearance:none;cursor:pointer;background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--line);
    border-radius:9px;padding:7px 12px;font-size:13px;color:var(--ink);display:flex;gap:8px;align-items:baseline;transition:.12s}
  .chip:hover{border-color:var(--accent)}
  .chip.fA{border-left-color:var(--famA)} .chip.fB{border-left-color:var(--famB)} .chip.fbridge{border-left-color:var(--famBr)}
  .chip.active{background:var(--panel2);box-shadow:inset 0 0 0 1px var(--accent)}
  .chip b{font-weight:700} .chip .c{font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
  .chip .ess{font-size:11px;color:var(--muted)}
  .list{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
  @media(max-width:720px){.list{grid-template-columns:1fr}}
  .icard{appearance:none;text-align:left;cursor:pointer;background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:13px 15px;box-shadow:var(--shadow);transition:.12s}
  .icard:hover{transform:translateY(-1px);border-color:var(--accent)}
  .icard:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .icard .iid{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;font-weight:600;margin-bottom:7px}
  .badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
  .badge{font-size:10.5px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--muted);white-space:nowrap}
  .badge.base{background:var(--panel2)}
  .badge.diffdot{border:0;color:#fff;font-weight:600}
  .snip{font-size:12.5px;color:var(--ink);opacity:.72;margin:0;
    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
  .seqbar-wrap{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:16px 18px 12px}
  .seqsel{display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap}
  select{font:inherit;padding:7px 10px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
  .seqhead{display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}
  .segtag{font-size:13px;font-weight:650;padding:4px 11px;border-radius:8px}
  .segtag em{font-style:normal;font-weight:500;opacity:.7;font-size:12px}
  .segtag.ta{background:var(--famA-bg);color:var(--famA)} .segtag.tb{background:var(--famB-bg);color:var(--famB)}
  .segtag.brdg{background:var(--famBr-bg);color:var(--famBr)}
  .arrow{color:var(--muted);font-weight:700}
  .track{display:flex;align-items:flex-end;gap:5px;height:140px;padding:6px 2px 0;border-bottom:2px solid var(--line);overflow-x:auto}
  .bar{flex:1 1 0;min-width:22px;border-radius:5px 5px 0 0;position:relative;cursor:pointer;
    display:flex;align-items:flex-start;justify-content:center;transition:opacity .12s;appearance:none;border:0}
  .bar:hover{opacity:.8} .bar:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .bar .bn{font-size:9.5px;color:#fff;opacity:.9;margin-top:3px;font-variant-numeric:tabular-nums}
  .bar.segA{background:linear-gradient(180deg,var(--famA),color-mix(in srgb,var(--famA) 60%, transparent))}
  .bar.segB{background:linear-gradient(180deg,var(--famB),color-mix(in srgb,var(--famB) 60%, transparent))}
  .bar.segBr{background:linear-gradient(180deg,var(--famBr),color-mix(in srgb,var(--famBr) 60%, transparent))}
  .divider{flex:0 0 2px;align-self:stretch;background:repeating-linear-gradient(180deg,var(--muted) 0 5px,transparent 5px 10px);
    position:relative;margin:0 8px}
  .divider span{position:absolute;top:-3px;left:50%;transform:translateX(-50%);white-space:nowrap;font-size:10px;
    color:var(--muted);background:var(--panel);padding:0 4px}
  .axis{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:7px;padding:0 2px}
  .note{margin-top:22px;background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:15px 18px;font-size:13.5px}
  .note b{color:var(--accent)}
  .hint{font-size:12px;color:var(--muted);margin:0 0 14px}
  /* modal */
  .backdrop{position:fixed;inset:0;background:rgba(10,14,20,.55);backdrop-filter:blur(2px);display:none;
    align-items:flex-start;justify-content:center;padding:40px 16px;z-index:50;overflow-y:auto}
  .backdrop.open{display:flex}
  .modal{background:var(--panel);border:1px solid var(--line);border-radius:16px;max-width:820px;width:100%;
    box-shadow:0 20px 60px rgba(0,0,0,.4);overflow:hidden}
  .modal header{padding:16px 20px;border-bottom:1px solid var(--line);background:var(--panel2);
    display:flex;justify-content:space-between;align-items:flex-start;gap:16px;position:sticky;top:0}
  .modal .mid{font-family:ui-monospace,Menlo,monospace;font-size:14px;font-weight:700;color:var(--accent)}
  .modal .mrepo{font-size:12px;color:var(--muted);margin-top:2px}
  .xbtn{appearance:none;border:1px solid var(--line);background:var(--panel);color:var(--muted);border-radius:8px;
    width:32px;height:32px;font-size:18px;cursor:pointer;flex:none;line-height:1}
  .xbtn:hover{color:var(--ink);border-color:var(--accent)}
  .mbody{padding:18px 20px 22px;max-height:70vh;overflow-y:auto}
  .mmeta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
  .mfiles{font-size:12.5px;color:var(--muted);margin-bottom:16px}
  .mfiles code{font-size:12px;color:var(--ink);background:var(--panel2);padding:1px 6px;border-radius:5px;border:1px solid var(--line);margin:2px 4px 2px 0;display:inline-block}
  .mstmt{font-size:13.5px;line-height:1.62;white-space:pre-wrap;word-break:break-word;
    background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto}
  .mstmt h4{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
  .foot{margin-top:30px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
</style>

<div class="wrap">
  <div class="eyebrow">Continual-Learning Benchmark · Dataset</div>
  <h1>SWE-Bench-CL: real GitHub issues arranged into a learning curriculum</h1>
  <p class="lede">Real issues from 8 open-source libraries organized into <b>9+10, repo-blocked</b>
  continual-learning sequences: an agent solves issues in order, carrying only <strong>memory</strong>
  (never code) across them, testing whether earlier knowledge transfers. Within each block issues ramp
  <strong>easy&rarr;hard</strong>; repo pairs are chosen by <strong>shared knowledge family</strong>.</p>

  <div class="totals">
    <div class="stat"><b id="tPool">–</b><span>training issues</span></div>
    <div class="stat"><b id="tRepos">–</b><span>repositories</span></div>
    <div class="stat"><b id="tEps">–</b><span>CL sequences</span></div>
    <div class="stat"><b>9+10</b><span>per sequence (learn+transfer)</span></div>
  </div>

  <div class="tabs">
    <button class="tab active" id="tabIssues">Issues by repo</button>
    <button class="tab" id="tabSeqs">CL sequences</button>
  </div>

  <section id="viewIssues">
    <div class="chips" id="repoChips"></div>
    <p class="hint">Click any issue to read its full problem statement, modified files and tests.</p>
    <div class="list" id="issueList"></div>
  </section>

  <section id="viewSeqs" hidden>
    <div class="seqbar-wrap">
      <div class="seqsel"><label for="seqPick">Sequence:</label><select id="seqPick"></select></div>
      <div class="seqhead" id="seqHead"></div>
      <div class="track" id="seqTrack"></div>
      <div class="axis"><span>easy</span><span>within-block difficulty, easy &rarr; hard (resets each block)</span><span>hard</span></div>
      <p class="hint" style="margin-top:12px">Bar height = combined difficulty (0.5 human + 0.5 qwen). Click a bar to open the issue.</p>
    </div>
    <div class="note"><b>Why this design?</b> Pairing (inter-group) follows whether two libraries share
    high-level knowledge — numeric family transfers numpy broadcasting / metadata preservation, framework
    family transfers hook &amp; node-tree pipelines — so transfer is meaningful. Ordering (intra-group) uses
    stratified-random sampling over combined difficulty + easy&rarr;hard sorting. The first governs transfer
    signal, the second the difficulty curriculum.</div>
  </section>

  <div class="foot mono">data/swe_bench_cl/train_pool_avg0.6.jsonl · swecl_train_episodes.jsonl ·
  difficulty = 0.5·(difficulty_score−1)/3 + 0.5·(1−qwen_solved/4) · baseline = qwen3.5-9B pass@4</div>
</div>

<div class="backdrop" id="backdrop"><div class="modal" id="modal"></div></div>

<script>
const DATA = /*__DATA__*/;
const FAMCLS = {A:"fA", B:"fB", bridge:"fbridge"};
const esc = s => String(s==null?"":s).replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function diffColor(c){ // green(easy)->red(hard)
  if(c==null) return "var(--muted)";
  const h = Math.round(140 - c*140); return `hsl(${h} 62% 42%)`;
}
function baseTxt(it){ return (it.solved4!=null&&it.avg4!=null) ? `qwen ${it.solved4}/4 · avg ${it.avg4.toFixed(2)}` : "—"; }

document.getElementById("tPool").textContent = DATA.totals.pool;
document.getElementById("tRepos").textContent = DATA.totals.repos;
document.getElementById("tEps").textContent = DATA.totals.episodes;

/* ---- issues tab ---- */
let activeRepo = DATA.repos[0].short;
const chipsEl = document.getElementById("repoChips");
chipsEl.innerHTML = DATA.repos.map(r =>
  `<button class="chip ${FAMCLS[r.family]}" data-repo="${esc(r.short)}">
     <b>${esc(r.short)}</b><span class="c">${r.count}</span><span class="ess">${esc(r.essence)}</span>
   </button>`).join("");
chipsEl.querySelectorAll(".chip").forEach(c => c.onclick = () => { activeRepo = c.dataset.repo; renderIssues(); });

function renderIssues(){
  chipsEl.querySelectorAll(".chip").forEach(c => c.classList.toggle("active", c.dataset.repo===activeRepo));
  const items = Object.values(DATA.issues).filter(i => i.repo===activeRepo).sort((a,b)=>a.combined-b.combined);
  document.getElementById("issueList").innerHTML = items.map(it => `
    <button class="icard" data-id="${esc(it.id)}">
      <div class="iid">${esc(it.id)}</div>
      <div class="badges">
        <span class="badge diffdot" style="background:${diffColor(it.combined)}">diff ${it.combined.toFixed(2)}</span>
        <span class="badge">${esc(it.difficulty)}</span>
        <span class="badge base">${esc(baseTxt(it))}</span>
      </div>
      <p class="snip">${esc(it.statement.slice(0,180))}…</p>
    </button>`).join("");
  document.querySelectorAll("#issueList .icard").forEach(c => c.onclick = () => openModal(c.dataset.id));
}

/* ---- sequences tab ---- */
const seqPick = document.getElementById("seqPick");
seqPick.innerHTML = DATA.sequences.map((s,i) =>
  `<option value="${i}">#${s.idx} — ${esc(s.repoA)} → ${esc(s.repoB)}</option>`).join("");
seqPick.onchange = () => renderSeq(+seqPick.value);

function segClass(pos){ return pos<9 ? "segA" : "segB"; }
function renderSeq(i){
  const s = DATA.sequences[i];
  document.getElementById("seqHead").innerHTML =
    `<span class="segtag ta">Block 1 · ${esc(s.repoA)} <em>×9 learn</em></span>
     <span class="arrow">→</span>
     <span class="segtag tb">Block 2 · ${esc(s.repoB)} <em>×10 transfer</em></span>`;
  let html = "";
  s.ids.forEach((id,pos) => {
    const it = DATA.issues[id] || {combined:null};
    const h = it.combined==null ? 22 : Math.round(22 + it.combined*78);
    const tip = `${id}  |  difficulty ${it.combined==null?"—":it.combined}  |  qwen ${it.solved4==null?"?":it.solved4}/4`;
    html += `<button class="bar ${segClass(pos)}" style="height:${h}%" data-id="${esc(id)}" title="${esc(tip)}"><span class="bn">${pos+1}</span></button>`;
    if(pos===8) html += `<div class="divider" title="notify_change: switch repo"><span>switch →</span></div>`;
  });
  const track = document.getElementById("seqTrack");
  track.innerHTML = html;
  track.querySelectorAll(".bar").forEach(b => b.onclick = () => openModal(b.dataset.id));
}

/* ---- modal ---- */
const backdrop = document.getElementById("backdrop");
function openModal(id){
  const it = DATA.issues[id];
  if(!it){ return; }
  const files = it.files.length ? it.files.map(f=>`<code>${esc(f)}</code>`).join("") : "<span>—</span>";
  document.getElementById("modal").innerHTML = `
    <header>
      <div><div class="mid">${esc(it.id)}</div><div class="mrepo">${esc(it.repo)} · created ${esc(it.created)} · base ${esc(it.commit)}</div></div>
      <button class="xbtn" id="xbtn" aria-label="Close">×</button>
    </header>
    <div class="mbody">
      <div class="mmeta">
        <span class="badge diffdot" style="background:${diffColor(it.combined)}">combined difficulty ${it.combined.toFixed(2)}</span>
        <span class="badge">${esc(it.difficulty)} · score ${esc(it.score)}</span>
        <span class="badge base">qwen3.5-9B: solved ${esc(it.solved4)}/4 · avg ${it.avg4==null?"—":it.avg4.toFixed(3)}</span>
        <span class="badge">FAIL→PASS ${it.f2p} · PASS→PASS ${it.p2p}</span>
      </div>
      <div class="mfiles"><b>Modified files:</b><br>${files}</div>
      <div class="mstmt"><h4>Problem statement</h4>${esc(it.statement)}${it.statement.length>=8000?"\n\n… (truncated)":""}</div>
    </div>`;
  document.getElementById("xbtn").onclick = closeModal;
  backdrop.classList.add("open");
}
function closeModal(){ backdrop.classList.remove("open"); }
backdrop.onclick = e => { if(e.target===backdrop) closeModal(); };
document.addEventListener("keydown", e => { if(e.key==="Escape") closeModal(); });

/* ---- tabs ---- */
const tI = document.getElementById("tabIssues"), tS = document.getElementById("tabSeqs");
tI.onclick = () => { tI.classList.add("active"); tS.classList.remove("active");
  document.getElementById("viewIssues").hidden=false; document.getElementById("viewSeqs").hidden=true; };
tS.onclick = () => { tS.classList.add("active"); tI.classList.remove("active");
  document.getElementById("viewIssues").hidden=true; document.getElementById("viewSeqs").hidden=false; };

renderIssues();
renderSeq(0);
</script>
"""


if __name__ == "__main__":
    main()

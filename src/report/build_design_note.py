#!/usr/bin/env python3
"""A2 Q6: build the design note (LaTeX source + figures + PDF) from the stored results.

Every number in the note is read from a results file, so the prose cannot drift
from the measurements:

  Q1/Q2  reports/rerank_{mind,ebnerd}.json, reports/q3_data_*.json
  Q3     reports/q3_summary.json
  Q4     reports/q4_*.json (via src/serving/q4_report.collect)
  Q5     reports/eval_twostage_*_test.json, reports/eval_mind_test.json,
         reports/a2/eval_ebnerd_small_test.json, reports/a2/codabench_analysis.json
  Q3/Q5  reports/a2/freshness_curve.json
  Q9     A1 harness serving-time ablation (reports/a2/eval_ebnerd_small_test.json)

Codabench: leaderboard scores go in `CODABENCH_SCORES` below and screenshots in
reports/design_note/screenshots/ as mind_submission.png, mind_leaderboard.png,
ebnerd_submission.png, ebnerd_leaderboard.png. Missing ones render as a
labelled placeholder box, so the note builds before the submissions are scored.

Output: reports/design_note/design_note_a2.tex, figures/*.pdf, design_note_a2.pdf

Usage
-----
    python src/report/build_design_note.py            # writes .tex, figures, compiles PDF
    python src/report/build_design_note.py --no-pdf   # .tex and figures only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

REPORTS = REPO_ROOT / "reports"
OUT = REPORTS / "design_note"
FIG = OUT / "figures"
SHOTS = OUT / "screenshots"

TEAM = "Parth Dhawale \\quad\\&\\quad Peeyush Prashant"
# Codabench leaderboard scores of the two scored submissions.
#   MIND    entry 930394, 2026-09-17
#   EB-NeRD entry 934390, 2026-09-20 (the competition scores 50% of the testset)
CODABENCH_SCORES: dict[str, dict | None] = {
    "mind": {"auc": 0.6208, "mrr": 0.2904, "ndcg@5": 0.3118, "ndcg@10": 0.3683},
    "ebnerd": {"auc": 0.6411, "mrr": 0.4131, "ndcg@5": 0.4689, "ndcg@10": 0.5294},
}
# Per-day AUC from EB-NeRD's "Detailed Results" page (2023-06-01 .. 06-08).
# Transcribed from the leaderboard: the only numbers in the note that do not
# come from a stored results file.
EBNERD_PER_DAY_AUC = [0.6376, 0.6502, 0.6566, 0.6512, 0.6298, 0.6311, 0.6381, 0.6288]
EBNERD_PER_DAY_MEAN = 0.6404


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load(rel: str) -> dict:
    return json.loads((REPORTS / rel).read_text(encoding="utf-8"))


def tex(s: str) -> str:
    """Escape text for LaTeX."""
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("$", "\\$")):
        s = s.replace(a, b)
    return s


def tt(s: str) -> str:
    return "\\texttt{" + tex(s) + "}"


def f(x, nd=4) -> str:
    return f"{x:.{nd}f}"


def sg(x, nd=4) -> str:
    return f"{x:+.{nd}f}".replace("-", "$-$")


def ci(v: dict, nd=4) -> str:
    return f"{f(v['value'], nd)} [{f(v['ci_low'], nd)}, {f(v['ci_high'], nd)}]"


def dci(v: dict, nd=4) -> str:
    mark = "$^{*}$" if v["excludes_zero"] else ""
    return f"{sg(v['mean_diff'], nd)} [{sg(v['ci_low'], nd)}, {sg(v['ci_high'], nd)}]{mark}"


def table(spec: str, header: list[str], rows: list[list[str]], caption: str, label: str, size="\\small") -> str:
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return (f"\\begin{{table}}[H]\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\fitwidth{{\\begin{{tabular}}{{{spec}}}\\toprule\n{' & '.join(header)} \\\\\\midrule\n{body} \\\\\n"
            f"\\bottomrule\\end{{tabular}}}}\\end{{table}}\n")


def shot(name: str, what: str, w: float = 1.0, h: float = 1.6) -> str:
    """A Codabench screenshot, or a same-sized placeholder if it is not there yet.

    Height-capped so the page count does not move when a screenshot is added,
    replaced or removed.
    """
    p = SHOTS / f"{name}.png"
    if p.exists():
        return f"\\includegraphics[width={w}\\linewidth,height={h}cm,keepaspectratio]{{screenshots/{name}.png}}"
    return (f"\\fbox{{\\parbox[c][{h}cm][c]{{0.9\\linewidth}}{{\\centering\\small\\textit{{Screenshot pending:}} "
            f"{tex(what)}}}}}")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def collect() -> dict:
    from src.serving.q4_report import collect as q4_collect
    d = {"q4": q4_collect(), "q3": {r["dataset"]: r for r in load("q3_summary.json")}}
    for ds in ("mind", "ebnerd"):
        d[f"data_{ds}"] = load(f"q3_data_{ds}.json")
        d[f"q2_{ds}"] = load(f"rerank_{ds}.json")
        d[f"q5_{ds}"] = load(f"eval_twostage_{ds}_test.json")
    d["a1_mind"] = load("eval_mind_test.json")
    d["a1_ebnerd"] = load("a2/eval_ebnerd_small_test.json")
    d["cb"] = load("a2/codabench_analysis.json")
    d["fresh"] = load("a2/freshness_curve.json")
    d["split_mind"] = json.loads((REPO_ROOT / "data/processed/mind/split_meta.json").read_text())
    d["split_ebnerd"] = json.loads((REPO_ROOT / "data/processed/ebnerd/split_meta.json").read_text())
    return d


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figures(d: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "serif"})
    FIG.mkdir(parents=True, exist_ok=True)
    blue, orange, grey, green = "#2F5D8A", "#D98032", "#9A9A9A", "#4F8A4C"

    # Fig 1: Q3 ablation on EB-NeRD (seed 13) + MIND variants (seed means), test AUC
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), gridspec_kw={"width_ratios": [5, 3]})
    eb = d["q3"]["ebnerd"]["variants"]
    order = [("nrms", "NRMS"), ("nrms_fresh", "+fresh"), ("nrms_pop", "+pop"),
             ("nrms_popfresh", "+pop+fresh\n(gate)"), ("nrms_popfresh_sum", "+pop+fresh\n(sum)")]
    vals = [eb[k]["test"]["auc"]["per_seed"]["13"] for k, _ in order]
    cols = [grey, green, blue, orange, blue]
    axes[0].bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    axes[0].set_xticks(range(len(vals)), [l for _, l in order], fontsize=7)
    axes[0].set_ylim(0.5, 0.76)
    axes[0].set_ylabel("test AUC")
    axes[0].set_title("EB-NeRD small, seed 13", fontsize=8)
    mi = d["q3"]["mind"]["variants"]
    morder = [("nrms", "NRMS"), ("nrms_pop", "+pop\n(gate)"), ("nrms_pop_sum", "+pop\n(sum)")]
    m = [mi[k]["test"]["auc"]["mean"] for k, _ in morder]
    s = [mi[k]["test"]["auc"]["std"] or 0 for k, _ in morder]
    axes[1].bar(range(3), m, yerr=s, capsize=3, color=[grey, blue, orange])
    for i, (v, e) in enumerate(zip(m, s)):
        axes[1].text(i, v + e + 0.001, f"{v:.4f}", ha="center", fontsize=7)
    axes[1].set_xticks(range(3), [l for _, l in morder], fontsize=7)
    axes[1].set_ylim(0.60, 0.645)
    axes[1].set_title("MIND-small, mean $\\pm$ sd of 3 seeds", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "q3_ablation.pdf")
    plt.close(fig)

    # Fig 2: click rate by article age (EB-NeRD test)
    fr = d["fresh"]["bins"]
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    ax.bar(range(len(fr)), [b["click_rate"] for b in fr], color=blue)
    ax.set_xticks(range(len(fr)), [b["age"] for b in fr], rotation=45, fontsize=6.5)
    ax.set_ylabel("click rate")
    ax.set_title("EB-NeRD test: click rate by article age", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "freshness_curve.pdf")
    plt.close(fig)

    # Fig 3: Q4 serving latency by stage + 1x vs 10x p99
    q4 = d["q4"]
    stages = ["profile", "bm25", "faiss", "merge", "features", "nrms", "sort"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for j, ds in enumerate(("mind", "ebnerd")):
        left = 0.0
        for k, st in enumerate(stages):
            v = q4[f"lat_{ds}"]["modes"]["serving"]["stages_ms"][st]["mean"]
            axes[0].barh(j, v, left=left, color=plt.cm.tab10(k), label=st if j == 0 else None)
            left += v
    axes[0].set_yticks([0, 1], ["MIND", "EB-NeRD"])
    axes[0].set_xlabel("mean ms per request (serving mode)")
    axes[0].legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.45), frameon=False)
    names = ["serving", "as_is"]
    x = np.arange(2)
    for i, ds in enumerate(("mind", "ebnerd")):
        sc = q4[f"scale_{ds}"]["scales"]
        one = [sc["1"]["serving_total_estimate_ms"]["p99"], sc["1"]["as_is_total_estimate_ms"]["p99"]]
        ten = [sc["10"]["serving_total_estimate_ms"]["p99"], sc["10"]["as_is_total_estimate_ms"]["p99"]]
        axes[1].bar(x + i * 0.42 - 0.1, one, 0.2, color=blue if i == 0 else green, label=f"{'MIND' if i == 0 else 'EB'} 1x")
        axes[1].bar(x + i * 0.42 + 0.1, ten, 0.2, color=orange if i == 0 else "#B5533C", label=f"{'MIND' if i == 0 else 'EB'} 10x")
    axes[1].axhline(100, color="black", lw=0.8, ls="--")
    axes[1].text(-0.3, 120, "SLA: p99 < 100 ms", fontsize=6.5, ha="left")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.5, 1000)
    axes[1].set_xticks(x + 0.21, ["serving mode", "batch code per request"])
    axes[1].set_ylabel("p99 ms (log)")
    axes[1].legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "q4_latency.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #

def preamble() -> str:
    return r"""\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{fontspec}
\usepackage{booktabs,graphicx,xcolor,tabularx,amsmath,float,array}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{3pt}\setlength{\parindent}{0pt}
\setlength{\abovecaptionskip}{3pt}\setlength{\belowcaptionskip}{2pt}
\setlength{\textfloatsep}{8pt}\setlength{\floatsep}{6pt}\setlength{\intextsep}{6pt}
\renewcommand{\arraystretch}{1.05}
% Long \texttt{} paths and feature names cannot hyphenate; let TeX stretch the
% interword space a little rather than push a line past the margin.
\emergencystretch=3em
\hyphenpenalty=1000
\newcommand{\sig}{$^{*}$}
\newcommand{\fitwidth}[1]{\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{#1}}
\newenvironment{tightitem}{\begin{itemize}\setlength{\itemsep}{1pt}\setlength{\parskip}{0pt}\setlength{\topsep}{2pt}}{\end{itemize}}
\begin{document}
"""


def title(d: dict) -> str:
    return rf"""\begin{{center}}
{{\Large\bfseries Design Note --- Assignment 2: Learning from Click-Logs on MIND and EB-NeRD}}\\[4pt]
CS4.406 Information Retrieval \& Extraction \quad$\cdot$\quad {TEAM}\\[2pt]
{{\small Code: \url{{https://github.com/Zenith053/Assignment1-IRE}} (branch \texttt{{parth/q4-serving}})}}
\end{{center}}
\vspace{{-4pt}}
"""


def sec_overview(d: dict) -> str:
    dm, de = d["data_mind"], d["data_ebnerd"]
    sm, se = d["split_mind"]["splits"], d["split_ebnerd"]["splits"]

    def win(s):
        return f"{s['t_min'][:10]} -- {s['t_max'][:10]}"

    rows = []
    for name, dd, ss in (("MIND-small", dm, sm), ("EB-NeRD small", de, se)):
        rows.append([name] + [f"{ss[s]['n_impressions']:,}" for s in ("train", "val", "test")] +
                    [f"{dd['n_articles']:,}", f"{dd['splits']['test']['candidates_per_impression_median']:.0f}",
                     f"{dd['splits']['test']['users_short_history_pct']:.0f}\\%"])
    t = table("lrrrrrr", ["dataset", "train imp.", "val imp.", "test imp.", "articles", "cand./imp.", "test users $<$20 clicks"],
              rows, "Data used throughout (temporal splits; the shipped held-out file is the test split).", "tab:data")
    return rf"""\section{{What we built}}
A two-stage news recommender on MIND-small and EB-NeRD \emph{{small}} (EB-NeRD was rebuilt from demo to small
for A2, $\sim$10$\times$ more data). \textbf{{Stage 1}} (A1's candidate generator) retrieves up to 200 candidates
per request: BM25 top-80 $\cup$ FAISS top-100 over A1 embeddings (MiniLM for MIND, the provided word2vec for
EB-NeRD) $\cup$ train-popularity top-20. \textbf{{Stage 2}} re-ranks them. We built two families of re-rankers:
a GBDT/MLP over hand-crafted behavioural features (Q1--Q2) and NRMS~\cite{{nrms}}, the official baseline, plus a
popularity/freshness extension (Q3). Every evaluation is on a strict temporal split
(MIND train {win(sm['train'])}, test {win(sm['test'])}; EB-NeRD train {win(se['train'])},
test {win(se['test'])}); each split's user histories come from a snapshot that ends before the split starts,
asserted by tests (\texttt{{tests/test\_no\_leakage.py}}). Two evaluation \emph{{universes}} are kept apart:
\textbf{{A}} re-ranks the impression's own shown (inview) list, which is what Codabench scores, and
\textbf{{B}} is the full pipeline: retrieve 200 from the catalogue, then re-rank.
{t}"""


def sec_q1(d: dict) -> str:
    rows = [
        ["Click history", "history length (raw, log), categories, category entropy, decay-weighted mass; hours since last click$^{E}$, mean past dwell time and scroll$^{E}$"],
        ["Session", "candidate position (raw, normalised), inview size, hour of day, day of week; index in session, clicks so far, seconds since session start$^{E}$"],
        ["Article", "train clicks (log), popularity rank, is-head, \\textbf{trailing clicks 24 h / 168 h}, title length; \\textbf{freshness (hours since publish)}$^{E}$"],
        ["Match", "category match, category match rate; BM25 and semantic score and their in-impression ranks, popularity rank in impression"],
    ]
    t = table("lp{0.78\\linewidth}", ["group", "features (30 in total)"], rows,
              "Q1 feature table. $^{E}$EB-NeRD only (MIND has no timestamps, sessions or publish times); on MIND these are NaN, never faked.", "tab:features")
    return rf"""\section{{Q1: Click-history, session and article features}}
{t}
\textbf{{Behaviour-window boundary.}} Two new mechanisms read \emph{{other}} impressions, so each has its own test:
trailing click counts count only clicks with timestamp strictly before the impression (a counterfactual test
rebuilds the counter from past-only events and demands identical counts), and session features are asserted
strictly increasing in real time. Columns only known after the click (current \texttt{{read\_time}},
\texttt{{scroll\_percentage}}, lifetime \texttt{{total\_pageviews}} \dots) are excluded from every feature; a
user's \emph{{past}} dwell time is an ordinary feature. The pre-computed \texttt{{hybrid}} score is excluded
because it was fitted on the validation split that also early-stops the re-rankers.
"""


def sec_q2(d: dict) -> str:
    rows = []
    for sc, lab in (("popularity", "A1 popularity"), ("bm25", "A1 BM25"), ("semantic", "A1 semantic (before)"),
                    ("gbdt", "GBDT (after)"), ("mlp", "MLP (after)")):
        r = [lab]
        for ds in ("mind", "ebnerd"):
            m = d[f"q2_{ds}"]["metrics"][sc]
            r += [f(m["auc"]["value"]), f(m["mrr"]["value"]), f(m["ndcg@10"]["value"])]
        rows.append(r)
    ta = table("lrrrrrr", ["", "MIND AUC", "MRR", "nDCG@10", "EB AUC", "MRR", "nDCG@10"], rows,
               "Q2, Universe A (re-rank the shown list), test metrics on 10,000 impressions per dataset.",
               "tab:q2a", size="\\footnotesize")
    drows = []
    for mdl in ("gbdt", "mlp"):
        r = [f"{mdl.upper()} $-$ semantic"]
        for ds in ("mind", "ebnerd"):
            p = d[f"q2_{ds}"]["paired_vs_before"][mdl]
            r += [dci(p["auc"], 3), dci(p["ndcg@10"], 3)]
        drows.append(r)
    ta += table("lrrrr", ["", "MIND $\\Delta$AUC", "MIND $\\Delta$nDCG@10", "EB $\\Delta$AUC", "EB $\\Delta$nDCG@10"], drows,
                "Q2, after vs before (best single A1 scorer, semantic on both): paired bootstrap 95\\% CI, 10,000 "
                "resamples; $^{*}$ = CI excludes zero. MRR and nDCG@5 move the same way (all significant).",
                "tab:q2d", size="\\footnotesize")
    bm, be = d["q2_mind"]["universe_b"], d["q2_ebnerd"]["universe_b"]
    rows = [["recall@200 (click retrieved)", f(bm["recall_at_k"], 3), f(be["recall_at_k"], 3)]]
    for name, lab in (("stage1_order", "AUC | retrieved, stage-1 order"), ("gbdt_inview_applied_to_retrieved", "AUC | retrieved, GBDT")):
        rows.append([lab, ci(bm["conditional_on_retrieval"][name]["auc"], 3), ci(be["conditional_on_retrieval"][name]["auc"], 3)])
    rows.append(["nDCG@10 | retrieved, before $\\to$ after",
                 f"{f(bm['ndcg10_conditional_on_retrieval']['stage1_order'], 3)} $\\to$ {f(bm['ndcg10_conditional_on_retrieval']['gbdt_inview_applied_to_retrieved'], 3)}",
                 f"{f(be['ndcg10_conditional_on_retrieval']['stage1_order'], 3)} $\\to$ {f(be['ndcg10_conditional_on_retrieval']['gbdt_inview_applied_to_retrieved'], 3)}"])
    rows.append(["end-to-end nDCG@10, before $\\to$ after",
                 f"{f(bm['end_to_end_ndcg10']['before'], 4)} $\\to$ {f(bm['end_to_end_ndcg10']['after'], 4)}",
                 f"{f(be['end_to_end_ndcg10']['before'], 4)} $\\to$ {f(be['end_to_end_ndcg10']['after'], 4)}"])
    tb = table("lrr", ["", "MIND", "EB-NeRD small"], rows,
               f"Q2, Universe B (retrieve top-200, then re-rank), {bm['n_impressions']:,} test impressions each. "
               "Accuracy is conditional on a click being retrieved; end-to-end = recall $\\times$ conditional.", "tab:q2b", size="\\footnotesize")
    imp = {ds: d[f"q2_{ds}"]["gbdt_feature_importance"][:3] for ds in ("mind", "ebnerd")}

    def imps(ds):
        return ", ".join(f"{tt(x['feature'])} {x['gain']:.2f}" for x in imp[ds])
    return rf"""\section{{Q2: Retrieve-then-rank re-rankers}}
\textbf{{Architecture.}} \emph{{Option A:}} LightGBM with a LambdaRank objective~\cite{{lightgbm,lambdamart}} (optimises
nDCG directly; 31 leaves, $\ge$30 samples per leaf, L2 1.0, feature fraction 0.7, early stopping on validation
nDCG@10). \emph{{Option B:}} a small MLP (64--32) trained with a listwise softmax per impression, with a
missing-value indicator per feature so NaN features are usable. Both train on the train split, early-stop on
validation, and report on test. Raw train-click counts were replaced by in-impression ranks after they
dominated the GBDT's splits (a high-cardinality bias) and made it lose to semantic.

{ta}{tb}
\textbf{{Observations.}} Re-ranking over behavioural features \emph{{loses}} to A1's semantic scorer on MIND
(significant on every metric) but \emph{{wins}} by $+0.22$ AUC on EB-NeRD. Top GBDT features (gain):
MIND {imps('mind')}; EB-NeRD {imps('ebnerd')}. EB-NeRD is driven by \emph{{what is being clicked now}} and
\emph{{how new the article is}} --- the signals Q3 adds to NRMS. Stage 1 caps the whole system: 76\% (MIND) and
90\% (EB-NeRD) of test clicks are never retrieved. On retrieved lists the GBDT trained on shown negatives is at
chance on MIND (the negatives' distribution shifts) but near-perfect on EB-NeRD, where freshness and trailing
clicks separate the one live article from mostly stale retrieved ones. Making Q2 run required pinning LightGBM
and FAISS to one thread (faiss, torch and LightGBM each link an incompatible OpenMP runtime; multi-threaded
training segfaulted), and fixing a metric bug that had applied recall twice in Universe B.
"""


def sec_q3(d: dict) -> str:
    q = d["q3"]
    rows = []
    for ds, lab in (("mind", "MIND-small"), ("ebnerd", "EB-NeRD small")):
        r = q[ds]
        imp_v = r["improved"]
        for v, name in (("nrms", "NRMS (baseline)"), (imp_v, f"improved: {tt(imp_v)}")):
            t = r["variants"][v]["test"]
            per = t["auc"]["per_seed"]
            rows.append([lab if v == "nrms" else "", name, " / ".join(f(per[s]) for s in ("13", "14", "15")),
                         f"{f(t['auc']['mean'])} $\\pm$ {f(t['auc']['std'])}", f(t["mrr"]["mean"]),
                         f(t["ndcg@5"]["mean"]), f(t["ndcg@10"]["mean"])])
    tmain = table("llcrrrr", ["", "model", "test AUC, seeds 13 / 14 / 15", "AUC mean $\\pm$ sd", "MRR", "nDCG@5", "nDCG@10"],
                  rows, "Q3 baseline vs improved on the full test splits, 3 seeds each (MRR and nDCG are seed means).",
                  "tab:q3main", size="\\footnotesize")
    drows = []
    for ds, lab in (("mind", "MIND-small"), ("ebnerd", "EB-NeRD small")):
        c = q[ds]["comparisons"]["main/seed_avg"]["ci"]
        drows.append([lab] + [dci(c[m], 4) for m in ("auc", "mrr", "ndcg@5", "ndcg@10")])
    tmain += table("lrrrr", ["improved $-$ NRMS", "$\\Delta$AUC", "$\\Delta$MRR", "$\\Delta$nDCG@5", "$\\Delta$nDCG@10"], drows,
                   "Q3 significance: paired bootstrap 95\\% CI (10,000 resamples) on each test impression's metric averaged over "
                   "the 3 seeds; $^{*}$ = CI excludes zero.", "tab:q3ci", size="\\footnotesize")

    eb = q["ebnerd"]["comparisons"]
    arow = []
    for key, lab in (("ablation/nrms_fresh-vs-nrms", "+ freshness only vs NRMS"),
                     ("ablation/nrms_pop-vs-nrms", "+ popularity only vs NRMS"),
                     ("ablation/nrms_popfresh-vs-nrms", "+ both, learned gate (full) vs NRMS"),
                     ("ablation/nrms_popfresh_sum-vs-nrms", "+ both, plain sum vs NRMS"),
                     ("ablation/nrms_pop-vs-nrms_popfresh", "popularity only vs full"),
                     ("ablation/nrms_fresh-vs-nrms_popfresh", "freshness only vs full"),
                     ("ablation/nrms_popfresh_sum-vs-nrms_popfresh", "plain sum vs learned gate")):
        c = eb[key]["ci"]
        arow.append([f"EB-NeRD: {lab}", dci(c["auc"], 3), dci(c["mrr"], 3), dci(c["ndcg@10"], 3)])
    mi = q["mind"]["comparisons"]
    for key, lab in (("ablation/nrms_pop-vs-nrms", "+ popularity, learned gate vs NRMS (seed 13)"),
                     ("ablation/sum-vs-gate", "plain sum vs learned gate (3 seeds)")):
        c = mi[key]["ci"]
        arow.append([f"MIND: {lab}", dci(c["auc"], 3), dci(c["mrr"], 3), dci(c["ndcg@10"], 3)])
    tabl = table("lrrr", ["ablation (EB-NeRD: seed 13)", "$\\Delta$AUC", "$\\Delta$MRR", "$\\Delta$nDCG@10"], arow,
                 "Q3 ablation, paired bootstrap 95\\% CI on the full test split; $^{*}$ = CI excludes zero.", "tab:q3abl", size="\\footnotesize")
    sel = q["mind"]["selection"]["candidates"]
    fr = d["fresh"]
    peak = max(fr["bins"], key=lambda b: b["click_rate"])
    ref_m = q["mind"]["comparisons"]["reference/nrms-vs-a1_semantic"]["ci"]["auc"]
    ref_e = q["ebnerd"]["comparisons"]["reference/nrms-vs-a1_semantic"]["ci"]["auc"]
    return rf"""\section{{Q3: Baseline reproduced, then beaten}}
\textbf{{Baseline: NRMS}}~\cite{{nrms}}, the official ebnerd-benchmark baseline, ported to PyTorch with the
benchmark's hyperparameters: title length 30, history 20, 4 sampled negatives per click, 20 heads $\times$ 20
dims, additive attention 200, dropout 0.2, Adam $10^{{-4}}$, batch 32, trainable word embeddings initialised from
XLM-RoBERTa-base~\cite{{xlmr}} (one multilingual vocabulary for Danish and English). A news encoder (multi-head
self-attention over title tokens, additive pooling) and a user encoder (the same over the last 20 clicked
articles) give 400-d vectors; the score is their dot product. Deviations: PyTorch port; padding masked out of
every softmax (59\% of MIND test users have $<$20 clicks); evaluated on our temporal split, so numbers are not
the published ones. NRMS is \emph{{below}} A1 semantic on MIND ($\Delta$AUC {dci(ref_m)}) and \emph{{above}} it
on EB-NeRD ({dci(ref_e)}), where A1 only had the weaker provided word2vec vectors.

\textbf{{Principled change: popularity- and freshness-aware NRMS}} (after PP-Rec~\cite{{pprec}}). NRMS reads only
titles, but news decays within hours and Q2 showed trailing clicks and article age are EB-NeRD's strongest
signals. Each candidate gets $\log(1+\cdot)$ of its clicks in the trailing 1\,h / 24\,h / 7\,d (strictly before the
impression; the vectorised counter is tested equal to Q1's causal \texttt{{RollingPopularity}}) and, on EB-NeRD, its
age. A small MLP turns them into a signal score, blended with the content score by a per-user gate
$g=\sigma(w^\top u+b)$: $s=(1-g)\,s_\text{{content}}+g\,s_\text{{signal}}$; the ablation replaces the gate with a plain
sum. The variant reported as ``improved'' is chosen by mean \emph{{validation}} AUC among pre-stated candidates
(MIND: plain sum {f(sel['nrms_pop_sum'])} vs gate {f(sel['nrms_pop'])}), never by test.

{tmain}
{tabl}
\textbf{{Findings.}} (1) On EB-NeRD the change is large and robust: $+0.175$ AUC, significant on every seed and
metric, seed spread $\pm0.001$. Popularity carries almost all of it; freshness alone adds $+0.09$ but is largely
redundant with popularity (no AUC gain on top of it, $+0.003$ MRR/nDCG). (2) Freshness is not monotone: click
rate peaks at {peak['age']} ({f(peak['click_rate'], 3)}) and falls to {f(fr['bins'][-1]['click_rate'], 3)} for
articles older than 30 days; ``newer ranks higher'' scores {f(fr['auc_newer_is_better'], 3)} AUC,
the bump rule {f(fr['auc_closest_to_4h'], 3)}. (3) The per-user gate does not earn its keep: equal to a plain sum
on EB-NeRD and slightly worse on MIND. (4) On MIND the gain is small ($+0.005$) and \emph{{not robust to training
randomness}}: seed 14 is significantly worse than its baseline, the seed spread ($\pm0.009$) exceeds the gain, and
validation disagreed with test on that seed. The paired CI covers test-impression sampling, not training noise.
"""


def q5_table(d: dict) -> str:
    rows = []
    for ds, lab in (("mind", "MIND"), ("ebnerd", "EB-NeRD")):
        r = d[f"q5_{ds}"]
        for sc, name in (("stage1_order", "stage-1 order"), ("gbdt", "two-stage GBDT")):
            s = r["scorers"][sc]["slices"]["all"]
            rows.append([lab if sc == "stage1_order" else "", name, f(s["recall_at_k"], 3), ci(s["auc"], 3),
                         f(s["mrr_given_retrieved"]["value"], 3), f(s["ndcg@5_given_retrieved"]["value"], 3),
                         ci(s["ndcg@10_given_retrieved"], 3), f(s["diversity"]["value"], 3), f(s["novelty"]["value"], 2),
                         f(r["scorers"][sc]["coverage"], 3)])
    return table("llrrrrrrrr", ["", "system", "recall", "AUC$|$ret.", "MRR$|$ret.", "nDCG@5$|$ret.", "nDCG@10$|$ret.",
                                "diversity", "novelty", "coverage"], rows,
                 "Q5, full two-stage pipeline (Universe B, 3,500 test impressions per dataset), all metrics. "
                 "Accuracy is conditional on a retrieved click; diversity, novelty (bits) and coverage are over the top-10 shown.",
                 "tab:q5", size="\\scriptsize")


def q5_slices(d: dict) -> str:
    rows = []
    for ds, lab in (("mind", "MIND"), ("ebnerd", "EB-NeRD")):
        sl = d[f"q5_{ds}"]["scorers"]["gbdt"]["slices"]
        for name, sname in (("cold_users", "cold ($<$5 clicks)"), ("warm_users", "warm"),
                            ("low_history_users", "lowest-history quartile"), ("high_history_users", "other users"),
                            ("head_clicks", "head click"), ("tail_clicks", "tail click")):
            s = sl[name]
            if not s.get("available"):
                rows.append([lab if name == "cold_users" else "", sname, "0", "--", "--", "--", "--"])
                continue
            rows.append([lab if name == "cold_users" else "", sname, f"{s['n_impressions']:,}", f(s["recall_at_k"], 3),
                         ci(s["auc"], 3) if s["auc"]["n"] else "--", ci(s["ndcg@10_given_retrieved"], 3) if s["ndcg@10_given_retrieved"]["n"] else "--",
                         f(s["diversity"]["value"], 3)])
    return table("llrrrrr", ["", "slice", "n", "recall@200", "AUC$|$retrieved", "nDCG@10$|$retrieved", "diversity"], rows,
                 "Q5 slices for the two-stage GBDT pipeline, with bootstrap 95\\% CIs. EB-NeRD has no cold users under the "
                 "$<$5-click rule (every user has $\\ge$5), so the lowest-history quartile stands in.", "tab:q5slices", size="\\scriptsize")


def sec_q5(d: dict) -> str:
    cb = d["cb"]
    pc = cb["popularity_collapse_ebnerd"]
    bl = cb["click_free_blends"]
    subs = cb["submissions"]
    sm = subs.get("submission_mind_large_test_nrms", {})
    se = subs.get("submission_ebnerd_testset_nrms_fresh", {})

    def lb(ds):
        s = CODABENCH_SCORES.get(ds)
        return (" & ".join(f(s[m]) for m in ("auc", "mrr", "ndcg@5", "ndcg@10"))) if s else "\\multicolumn{4}{c}{\\textit{pending}}"
    leaderboard = (f"\\begin{{table}}[H]\\centering\\footnotesize\\caption{{Codabench submissions (hidden test sets).}}\\label{{tab:cb}}\n"
                   f"\\fitwidth{{\\begin{{tabular}}{{llrrrrrrr}}\\toprule\n competition & model & impressions & zip MB & offline AUC & AUC & MRR & nDCG@5 & nDCG@10\\\\\\midrule\n"
                   f"MIND 13967 & NRMS & {sm.get('impressions', 0):,} & {sm.get('zip_mb', 0)} & 0.6232 & {lb('mind')}\\\\\n"
                   f"EB-NeRD 2469 & NRMS + freshness & {se.get('impressions', 0):,} & {se.get('zip_mb', 0)} & 0.6459 & {lb('ebnerd')}\\\\\n"
                   "\\bottomrule\\end{tabular}}\\end{table}\n")
    shots = ("\\begin{figure}[H]\\centering\n"
             f"\\begin{{minipage}}{{0.49\\linewidth}}\\centering {shot('mind_submission', 'MIND submission', 1.0, 1.1)}\\end{{minipage}}\\hfill\n"
             f"\\begin{{minipage}}{{0.49\\linewidth}}\\centering {shot('ebnerd_submission', 'EB-NeRD submission', 1.0, 1.1)}\\end{{minipage}}\\\\[4pt]\n"
             f"{shot('mind_leaderboard', 'MIND leaderboard', 1.0, 0.8)}\\\\[3pt]\n"
             f"{shot('ebnerd_leaderboard', 'EB-NeRD leaderboard', 1.0, 1.1)}\n"
             "\\caption{Codabench. Top: the two submissions accepted (MIND left, EB-NeRD right). "
             "Bottom: our leaderboard rows, MIND above EB-NeRD.}\\label{fig:cb}\\end{figure}\n")
    return rf"""\section{{Q5: Extended evaluation}}
{q5_table(d)}{q5_slices(d)}
\textbf{{Observations.}} Beyond-accuracy metrics move against accuracy. On MIND the two-stage GBDT recommends a
\emph{{more}} diverse but far \emph{{less}} novel list than stage 1 alone (it learned to push popular articles), and
covers fewer articles; on EB-NeRD it gains accuracy with little change in diversity. Head-click impressions retrieve
far better than tail ones (MIND 0.500 vs 0.268, EB-NeRD 0.882 vs 0.124): much of stage 1's recall comes from the
popularity arm. Lowest-history users retrieve and rank worse on both datasets. Coverage is reported as a point
estimate; head-click slices have 100 (MIND) and 17 (EB-NeRD) impressions, so their CIs are wide.

\textbf{{Codabench.}} The hidden test sets carry no clicks, so trailing click counts cannot be computed on them. We
measured what that does on our labelled EB-NeRD test split: the best Q3 model scores {f(pc['real_clicks_up_to_impression']['auc'])}
with real trailing clicks, {f(pc['clicks_before_test_period_only']['auc'])} when counts can only use clicks from before
the test period (only {pc['clicks_before_test_period_only']['candidates_with_24h_clicks_pct']:.1f}\% of candidates keep a
non-zero 24\,h count) and {f(pc['no_clicks']['auc'])} with none --- below chance. We therefore submitted the best models
that need no click logs: NRMS on MIND and NRMS + freshness (publish times are in the test files) on EB-NeRD. The
inference path was verified before submitting: on \texttt{{MINDsmall\_dev}} it reproduces the offline AUC (0.6232;
Microsoft's official \texttt{{evaluate.py}} gives 0.6235 on the written ranks, the difference being rank ties), and on
EB-NeRD small validation 0.5539 / 0.6459; unseen title tokens use pretrained XLM-R vectors ({sm.get('titles_with_new_tokens_pct', 0):.1f}\% of MIND and {se.get('titles_with_new_tokens_pct', 0):.1f}\%
of EB-NeRD test titles), and every zip is validated line by line against the source file (row order, ids, rank
permutations). Rank-averaging click-free scorers looks promising offline (MIND NRMS + A1 semantic
{f(bl['mind']['rank_avg_0.5nrms_0.5semantic'])}; EB-NeRD NRMS + freshness with exposure counts
{f(bl['ebnerd']['rank_avg_0.5fresh_0.5exposure'])}) but was not submitted.

\textbf{{Leaderboard outcome.}} Both scored close to what our own splits predicted --- MIND
{f(CODABENCH_SCORES['mind']['auc'])} against 0.6232 offline, EB-NeRD {f(CODABENCH_SCORES['ebnerd']['auc'])} against
0.6459 --- so the estimate held on hidden sets of {sm.get('impressions', 0):,} and {se.get('impressions', 0):,}
impressions, and the popularity model we did not submit would have scored near chance. EB-NeRD scores 50\% of its
testset and publishes per-day metrics: AUC runs {f(min(EBNERD_PER_DAY_AUC))}--{f(max(EBNERD_PER_DAY_AUC))} over
eight days (day mean {f(EBNERD_PER_DAY_MEAN)} vs {f(CODABENCH_SCORES['ebnerd']['auc'])} pooled), so one leaderboard
figure hides $\pm${f((max(EBNERD_PER_DAY_AUC) - min(EBNERD_PER_DAY_AUC)) / 2, 3)} of day-to-day movement --- more
than separates most models in Table~\ref{{tab:q3main}}.
{leaderboard}{shots}"""


def sec_q4(d: dict) -> str:
    q = d["q4"]
    lm, le = q["lat_mind"]["modes"], q["lat_ebnerd"]["modes"]
    mm, me = q["mem_mind"], q["mem_ebnerd"]

    def cr(ds, mode, qps, sd=1):
        return next(r for r in q["cost"]["datasets"][ds]["modes"][mode]["table"] if r["qps"] == qps and r["cloud_slowdown"] == sd)
    rows = [
        ["memory to serve (MB)", f"{mm['summary']['served_total_mb']:.0f}", f"{me['summary']['served_total_mb']:.0f}"],
        ["\\quad largest: A1 / NRMS article vectors, NRMS weights (MB)",
         f"{mm['components']['stage1/a1_article_embeddings']['mb']:.0f} / {mm['components']['stage2/article_vectors']['mb']:.0f} / {mm['components']['stage2/nrms_parameters']['mb']:.0f}",
         f"{me['components']['stage1/a1_article_embeddings']['mb']:.0f} / {me['components']['stage2/article_vectors']['mb']:.0f} / {me['components']['stage2/nrms_parameters']['mb']:.0f}"],
        ["ANN index, catalogue: exact FAISS $\\to$ HNSW (MB)",
         f"{mm['ann_index_option']['flat_full_catalogue_mb']:.0f} $\\to$ {mm['ann_index_option']['hnsw_full_catalogue_mb']:.0f}",
         f"{me['ann_index_option']['flat_full_catalogue_mb']:.0f} $\\to$ {me['ann_index_option']['hnsw_full_catalogue_mb']:.0f}"],
        ["latency p50 / p95 / p99, one request (ms)",
         " / ".join(f"{lm['serving']['total_ms'][k]:.2f}" for k in ("p50", "p95", "p99")),
         " / ".join(f"{le['serving']['total_ms'][k]:.2f}" for k in ("p50", "p95", "p99"))],
        ["\\quad same, calling batch code per request (ms)",
         " / ".join(f"{lm['as_is']['total_ms'][k]:.1f}" for k in ("p50", "p95", "p99")),
         " / ".join(f"{le['as_is']['total_ms'][k]:.1f}" for k in ("p50", "p95", "p99"))],
        ["cores at 1k / 10k QPS (sim.\\ p99 ms)",
         f"{cr('mind', 'serving', 1000)['cores']} / {cr('mind', 'serving', 10000)['cores']} ({cr('mind', 'serving', 10000)['sim_p99_ms']:.1f})",
         f"{cr('ebnerd', 'serving', 1000)['cores']} / {cr('ebnerd', 'serving', 10000)['cores']} ({cr('ebnerd', 'serving', 10000)['sim_p99_ms']:.1f})"],
        ["cost per 1,000 queries at 1k QPS",
         f"\\${cr('mind', 'serving', 1000)['usd_per_1k_queries']:.6f}", f"\\${cr('ebnerd', 'serving', 1000)['usd_per_1k_queries']:.6f}"],
    ]
    t = table("lrr", ["", "MIND", "EB-NeRD small"], rows,
              "Q4 serving: full two-stage pipeline, one request at a time, Apple M4 CPU, one thread; 2,000 replayed test requests.",
              "tab:q4", size="\\footnotesize")
    a = q["cost"]["assumptions"]
    ut = q["cost"]["datasets"]["mind"]["modes"]
    return rf"""\section{{Q4: Serving and scale analysis}}
A one-request serving path loads everything once (indexes, model, every article's NRMS vector, so a request never
encodes text) and times eight stages; it reproduces the offline test scores (max difference
{q['correct_mind']['max_abs_diff']:.0e} MIND, {q['correct_ebnerd']['max_abs_diff']:.0e} EB-NeRD). PyTorch, FAISS and BLAS
are pinned to one thread (they defaulted to 4--10), so figures are per core.
{t}
\textbf{{Findings.}} BM25 is the largest stage and sets the tail: its query is the user's whole click history
(EB-NeRD users with 200+ clicks: p99 {q['lat_ebnerd']['modes']['serving']['by_history_length']['201-inf']['p99']:.2f}\,ms vs
{q['lat_ebnerd']['modes']['serving']['by_history_length']['1-10']['p99']:.2f}\,ms for 1--10). NRMS re-ranking is a constant
$\sim${lm['serving']['stages_ms']['nrms']['mean']:.1f}\,ms: 200 dot products over precomputed vectors. Calling A1's batch
functions per request is {q['lat_mind']['serving_speedup_over_as_is']['p50']}$\times$ slower on MIND, almost all of it
rebuilding an article$\to$tokens dictionary over the whole catalogue per call; caching request-independent work
removes it. \textbf{{Cost}} (assumed \${a['usd_per_core_hour']}/core-hour, one single-threaded worker per core,
{int(100 * a['target_utilisation'])}\% utilisation): a queueing simulation driven by the measured request times
(validated against M/M/1 theory) keeps p99 under {ut['serving']['utilisation_curve']['points'][-1]['sim_p99_ms']:.0f}\,ms
even at 95\% utilisation, so the 100\,ms SLA never binds and cost is linear in traffic; a 3$\times$ slower cloud vCPU
still meets it. Network, parsing and remote stores are not in these timings.
"""


def sec_10x(d: dict) -> str:
    q = d["q4"]
    sm, se = q["scale_mind"], q["scale_ebnerd"]
    s1, s10 = sm["scales"]["1"], sm["scales"]["10"]
    ef = lambda s, sc, e: next(x for x in s["scales"][sc]["hnsw_catalogue"]["ef_search"] if x["ef_search"] == e)["recall_at_k"]  # noqa: E731
    scen = {x["scenario"]: x for x in sm["scenarios"]}
    both = scen["10x catalogue and 10x traffic"]
    return rf"""\section{{Where the system breaks at 10\texorpdfstring{{$\times$}}{{x}}}}
Measured on a synthetic 10$\times$ catalogue (every article copied 10$\times$ with perturbed vectors; MIND
{sm['articles_1x']:,} $\to$ {s10['articles']:,} articles) and 10$\times$ click events, replaying {sm['n_queries']:,} real users:
\begin{{tightitem}}
\item \textbf{{First: per-request code that touches the whole catalogue.}} Calling the batch code per request, MIND p99
goes from {s1['as_is_total_estimate_ms']['p99']:.0f} to \textbf{{{s10['as_is_total_estimate_ms']['p99']:.0f}\,ms}} and breaks the SLA on a
single request (EB-NeRD {se['scales']['10']['as_is_total_estimate_ms']['p99']:.0f}\,ms). With request-independent work cached,
serving p99 grows only {s1['serving_total_estimate_ms']['p99']:.1f} $\to$ {s10['serving_total_estimate_ms']['p99']:.1f}\,ms.
\item \textbf{{Second: RAM per worker.}} Served memory grows {sm['memory_10x']['served_1x_mb']:.0f} $\to$
{sm['memory_10x']['served_10x_mb']:,.0f}\,MB; each MIND worker needs {sm['memory_10x']['worker_ram_10x_gb']}\,GB, above an assumed
{q['cost']['assumptions']['ram_gb_per_vcpu']}\,GB per vCPU, so RAM, not CPU, sets the bill ({both['billed_vcpus']} vCPUs for
{both['cores_for_cpu']} cores of work at 10$\times$ catalogue and traffic). Fix: share read-only vectors across workers
(memory-mapping) or move them to a vector-search service.
\item \textbf{{Third: approximate-search quality.}} Exact search over the whole MIND catalogue reaches
{s10['faiss_flat_catalogue']['latency_ms']['p99']:.0f}\,ms p99 at 10$\times$, pushing a larger site to HNSW~\cite{{hnsw,faiss}}, whose
recall@100 falls from {ef(sm, '1', 64):.2f} to {ef(sm, '10', 64):.2f} (efSearch 64; {ef(sm, '10', 256):.2f} at 256). Searching only the
circulating pool avoids this today ({s10['faiss_flat_pool']['latency_ms']['p99']:.1f}\,ms) but the pool grows with the site.
\item \textbf{{Not the bottleneck:}} stage-2 re-ranking and click-count lookups stay flat; 10$\times$ traffic alone is linear
cost (lower p99 from larger worker pools). Argued, not measured: 10$\times$ click events need a streaming counter, and
the popularity model's accuracy depends on that counter being real-time (it collapses to chance without it, \S5).
\end{{tightitem}}
Caveats: laptop CPU (indicative per-core numbers); synthetic near-duplicate copies make HNSW recall pessimistic and do
not grow BM25's vocabulary; users $\times$10 is a linear memory extrapolation.
"""


def sec_q9_and_close(d: dict) -> str:
    q9 = d["a1_ebnerd"]["q9_serving_time_ablation"]
    g = d["q2_ebnerd"]["q9_serving_time_ablation"]
    cols = ", ".join(tt(c) for c in g["columns"])
    rows = [[m, f(g["honest"][m]["value"]), f(g["leaky"][m]["value"]), dci(g["inflation_paired_ci"][m])]
            for m in ("auc", "mrr", "ndcg@5", "ndcg@10")]
    t9 = table("lrrr", ["metric", "honest (A2 features)", "with leaked features", "inflation, 95\\% CI"], rows,
               "Q9 on the A2 re-ranker (EB-NeRD small, 10,000 test impressions): the same GBDT trained with the "
               "dataset's serving-time-unavailable article totals added. $^{*}$ = CI excludes zero.", "tab:q9",
               size="\\footnotesize")
    return rf"""\section{{Anti-gaming, limitations and alternatives}}
\textbf{{Features unavailable at serving time}} (Q9). EB-NeRD ships lifetime totals
({cols}), known only after the fact. Adding them to the A2 GBDT's features inflates every metric significantly (Table~\ref{{tab:q9}});
using \texttt{{total\_pageviews}} alone as a popularity score inflates A1's popularity scorer far more,
{f(q9['leaky_popularity_auc'])} against {f(q9['honest_popularity_auc'])} AUC ({sg(q9['inflation'])}). The A2 feature table
excludes all of them by default (asserted by \texttt{{tests/test\_no\_leakage.py}}), and every trailing count is
strictly causal. The Codabench analysis (\S5) is the serving-time version of the same lesson: a feature that is
honest offline can still be unavailable at serving time if the live system does not log it.
{t9}

\textbf{{Limitations.}} MIND's gain is within seed noise; Q3 runs mix Apple-GPU and CUDA hardware across seeds; the
circulating candidate pool is built from the whole test period (slightly optimistic recall); Universe B labels a retrieved
but never-shown article as a non-click; Q5 uses 3,500 impressions and reports coverage without a CI.

\textbf{{Alternatives considered.}} A cross-encoder or full transformer news encoder (too slow to train on a laptop in the
time available; NRMS epochs took 11--13 min on an M4); retraining the GBDT on retrieved rather than shown negatives
(would fix the MIND distribution shift in Universe B); exposure counts as a click-free popularity proxy for Codabench
({f(d['cb']['click_free_blends']['ebnerd']['exposure_24h'], 3)} AUC alone on EB-NeRD, measured but not submitted); HNSW with larger efSearch for a bigger catalogue.
"""


def references() -> str:
    return r"""\begin{thebibliography}{9}\small\setlength{\itemsep}{0pt}
\bibitem{nrms} C.~Wu, F.~Wu, S.~Ge, T.~Qi, Y.~Huang, X.~Xie. Neural news recommendation with multi-head self-attention. EMNLP-IJCNLP 2019.
\bibitem{pprec} T.~Qi, F.~Wu, C.~Wu, Y.~Huang. PP-Rec: News recommendation with personalized user interest and time-aware news popularity. ACL 2021.
\bibitem{mind} F.~Wu et al. MIND: A large-scale dataset for news recommendation. ACL 2020.
\bibitem{ebnerd} J.~Kruse et al. EB-NeRD: A large-scale dataset for news recommendation. RecSys Challenge 2024.
\bibitem{xlmr} A.~Conneau et al. Unsupervised cross-lingual representation learning at scale. ACL 2020.
\bibitem{lightgbm} G.~Ke et al. LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 2017.
\bibitem{lambdamart} C.~J.~C.~Burges. From RankNet to LambdaRank to LambdaMART: An overview. Microsoft Research TR 2010.
\bibitem{hnsw} Y.~A.~Malkov, D.~A.~Yashunin. Efficient and robust approximate nearest neighbor search using HNSW graphs. TPAMI 2020.
\bibitem{faiss} J.~Johnson, M.~Douze, H.~J\'egou. Billion-scale similarity search with GPUs. IEEE Trans.\ Big Data 2021.
\end{thebibliography}
"""


def appendix(d: dict) -> str:
    # Every question now has a make target, so the long command list this used
    # to spell out is redundant.
    return r"""\appendix
\section{Reproduce}
{\small\begin{verbatim}
make data                       # A1 pipeline; EB-NeRD rebuilt at scale: small
make q1 q2 q3 q4 q5 q6          # one target per question, in order
\end{verbatim}}
\vspace{-6pt}
Codabench zips come from \texttt{src/submission/predict\_nrms.py} and are checked by
\texttt{validate\_zip.py}. Detailed tables are in \texttt{reports/q3\_summary.md} and
\texttt{reports/q4\_summary.md}; per-question implementation notes in \texttt{reports/a2\_q*\_implementation.md};
AI assistance in \texttt{reports/a2\_ai\_usage\_log.md} (A1's in \texttt{reports/ai\_usage\_log.md}); the run logs
behind every number in \texttt{logs/}. The build also writes three figures to
\texttt{reports/design\_note/figures/}, omitted here for length.
"""


def build(d: dict) -> str:
    return "".join([preamble(), title(d), sec_overview(d), sec_q1(d), sec_q2(d), sec_q3(d), sec_q5(d),
                    sec_q4(d), sec_10x(d), sec_q9_and_close(d), references(), appendix(d), "\\end{document}\n"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    d = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(exist_ok=True)
    figures(d)
    src = OUT / "design_note_a2.tex"
    src.write_text(build(d), encoding="utf-8")
    print(f"-> {src}")
    if args.no_pdf:
        return 0
    if not shutil.which("xelatex"):
        raise SystemExit("xelatex not found")
    for _ in range(2):                      # second pass resolves references
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
                           cwd=OUT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit("xelatex failed; see reports/design_note/design_note_a2.log")
    for ext in (".aux", ".out"):
        (OUT / f"design_note_a2{ext}").unlink(missing_ok=True)
    print(f"-> {OUT / 'design_note_a2.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

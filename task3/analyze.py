"""Build every Task 3 table/figure from saved results (offline and/or live, whichever exist).

python analyze.py --out OUT
Writes OUT/report_t3/.
"""
import argparse, csv, pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import *

INK, MUTED = "#0b0b0b", "#52514e"
COL = {"first": "#8a8985", "sc": "#2a78d6", "maxlogp": "#eda100", "maxsc": "#1baf7a", "sc_rwc": "#eb6834",
       "shortest": "#52514e", "vision_max": "#6250d6", "vision_rwc": "#9085e9", "visconf_max": "#e87ba4",
       "visconf_rwc": "#e34948", "tokfuse_max": "#008300", "tokfuse_rwc": "#0ca30c", "oracle": INK}
LABEL = {"first": "FirstAns", "sc": "Self-consistency", "maxlogp": "Max mean log-prob", "maxsc": "Max self-certainty",
         "sc_rwc": "Self-certainty + RWC", "shortest": "Shortest (length control)", "vision_max": "Vision-only (max)",
         "vision_rwc": "Vision-only + RWC", "visconf_max": "VisConf (max)", "visconf_rwc": "VisConf + RWC",
         "tokfuse_max": "Tokenwise fusion (max)", "tokfuse_rwc": "Tokenwise fusion + RWC", "oracle": "Oracle coverage"}
DSN = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, rd, name):
    fig.savefig(rd / f"{name}.png"); fig.savefig(rd / f"{name}.pdf"); plt.close(fig)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else (f"{v:.3e}" if v != 0 and abs(v) < 1e-3 else f"{v:.3f}")
    return str(v)


def write_table(rows, rd, name):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    rows = [{k: r.get(k, "") for k in keys} for r in rows]
    with open(rd / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    (rd / f"{name}.md").write_text("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
    tex = ["\\begin{tabular}{l" + "r" * (len(keys) - 1) + "}", "\\toprule", " & ".join(esc(k) for k in keys) + " \\\\", "\\midrule"]
    tex += [" & ".join(esc(fmt(r[k])) for k in keys) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    (rd / f"{name}.tex").write_text("\n".join(tex) + "\n")


def pct(x):
    return 100 * float(x)


def ci_txt(t):
    return f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"


# ----------------------------------------------------------------------------- Steps 1-2
def selection(key, od, rd):
    S = pd.read_parquet(od / f"{key}_selection.parquet")
    E = S[S.split == "eval"]
    methods = list(dict.fromkeys(E.method))
    perq = E.groupby(["method", "N", "qid"])[["correct", "token_cost", "oracle", "n_groups"]].mean().reset_index()
    agg = perq.groupby(["method", "N"])[["correct", "token_cost", "oracle", "n_groups"]].mean().reset_index()
    rows = []
    for N in BUDGETS:
        a = agg[agg.N == N].set_index("method")
        r = dict(N=N, oracle_coverage_pct=pct(a.oracle.iloc[0]), distinct_groups=a.n_groups.iloc[0],
                 token_cost_all=a.loc["sc", "token_cost"], token_cost_FirstAns=a.loc["first", "token_cost"])
        for m in methods:
            r[f"{m}_acc_pct"] = pct(a.loc[m, "correct"])
        rows.append(r)
    write_table(rows, rd, f"{key}_accuracy_by_budget")
    # comparison at N=8 and 32 with paired bootstrap vs self-consistency and vs FirstAns
    comp = []
    for N in (8, 32):
        P = perq[perq.N == N].pivot(index="qid", columns="method", values="correct")
        for m in methods:
            row = dict(N=N, method=LABEL[m], accuracy_pct=pct(P[m].mean()))
            if m != "sc":
                row["minus_SC_pp"] = ci_txt(paired_bootstrap(P[m], P["sc"]))
            if m != "first":
                row["minus_FirstAns_pp"] = ci_txt(paired_bootstrap(P[m], P["first"]))
            comp.append(row)
        if key == "mathvista":
            for a_, b_ in (("tokfuse_max", "visconf_max"), ("tokfuse_rwc", "visconf_rwc"), ("visconf_rwc", "sc_rwc"),
                           ("visconf_rwc", "vision_rwc"), ("visconf_rwc", "visconf_max"), ("sc_rwc", "maxsc")):
                comp.append(dict(N=N, method=f"{LABEL[a_]} − {LABEL[b_]}", accuracy_pct=np.nan,
                                 minus_SC_pp=ci_txt(paired_bootstrap(P[a_], P[b_]))))
        else:
            comp.append(dict(N=N, method=f"{LABEL['sc_rwc']} − {LABEL['maxsc']}", accuracy_pct=np.nan,
                             minus_SC_pp=ci_txt(paired_bootstrap(P["sc_rwc"], P["maxsc"]))))
    write_table(comp, rd, f"{key}_method_comparison")
    # sampling vs selection failures
    dec = []
    for m in ("first", "sc", "sc_rwc", "maxsc") + (("visconf_rwc",) if key == "mathvista" else ()):
        X = E[E.method == m]
        X = X.assign(no_correct=~X.oracle, missed=X.oracle & ~X.correct)
        d = X.groupby(["N", "qid"])[["correct", "no_correct", "missed"]].mean().groupby("N").mean()
        for N, r in d.iterrows():
            dec.append(dict(method=LABEL[m], N=N, correct_pct=pct(r.correct), no_correct_candidate_pct=pct(r.no_correct),
                            correct_available_but_missed_pct=pct(r.missed)))
    write_table(dec, rd, f"{key}_failure_decomposition")
    # curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    shown = ["oracle", "first", "sc", "maxlogp", "maxsc", "sc_rwc", "shortest"] + (
        ["vision_rwc", "visconf_max", "visconf_rwc"] if key == "mathvista" else [])        # full set in the tables
    for m in shown:
        if m == "oracle":
            y = [pct(agg[(agg.method == "sc") & (agg.N == N)].oracle.iloc[0]) for N in BUDGETS]
            cst = [agg[(agg.method == "sc") & (agg.N == N)].token_cost.iloc[0] for N in BUDGETS]
            ls = "--"
        else:
            y = [pct(agg[(agg.method == m) & (agg.N == N)].correct.iloc[0]) for N in BUDGETS]
            cst = [agg[(agg.method == m) & (agg.N == N)].token_cost.iloc[0] for N in BUDGETS]
            ls = "-"
        ax[0].plot(BUDGETS, y, ls, marker="o", ms=3, lw=1.6, color=COL[m], label=LABEL[m])
        ax[1].plot(cst, y, ls, marker="o", ms=3, lw=1.6, color=COL[m])
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(BUDGETS); ax[0].set_xticklabels(BUDGETS)
    ax[0].set_xlabel("candidates N"); ax[0].set_ylabel("accuracy (%)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("mean generated tokens per question (offline proxy)")
    ax[0].set_title(f"{DSN[key]} — held-out (800 q, 10 orders)", fontsize=9)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    save(fig, rd, f"fig_{key}_accuracy_vs_budget")
    return S


# ----------------------------------------------------------------------------- Step 3
def signal_tables(key, od, rd):
    D = pd.read_parquet(od / f"{key}_signals_per_question.parquet")
    rows = []
    for s, g in D.groupby("score", sort=False):
        m = g[g.mixed]
        rows.append(dict(score=s, top_minus_bottom_quartile_pp_all=pct(g.top_minus_bottom.mean()),
                         top_minus_bottom_quartile_pp_mixed=pct(m.top_minus_bottom.mean()), n_mixed_questions=len(m),
                         mean_spearman_with_correct_mixed=m.spearman_correct.mean(),
                         mean_spearman_with_length=g.spearman_length.mean()))
    write_table(rows, rd, f"{key}_signal_associations")
    if (od / f"{key}_conf_vision_corr.parquet").exists():
        C = pd.read_parquet(od / f"{key}_conf_vision_corr.parquet")
        pq_ = C.groupby("qid")[["r_calconf_vision", "r_kl_vision", "cov_c_v"]].mean()
        rows = []
        for col in ("r_calconf_vision", "r_kl_vision", "cov_c_v"):
            x = pq_[col].dropna()
            rows.append(dict(statistic=col, mean=x.mean(), median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75),
                             frac_positive=(x > 0).mean(), correct_rollouts_mean=C[C.correct][col].mean(),
                             incorrect_rollouts_mean=C[~C.correct][col].mean()))
        write_table(rows, rd, f"{key}_confidence_vision")
        idn = jload(od / f"{key}_fusion_identity.json")
        jdump(idn, rd / f"{key}_fusion_identity.json")
        T = jload(od / f"{key}_trajectories.json")
        tr = T["trajectories"]
        fig, axes = plt.subplots(len(tr), 1, figsize=(9, 1.7 * len(tr)), sharex=False)
        for ax, t in zip(np.atleast_1d(axes), tr):
            c, v = np.array(t["c"]), np.array(t["v"])
            w = max(1, len(c) // 50)
            sm = lambda x: np.convolve(x, np.ones(w) / w, mode="same") if len(x) else x
            ax.plot(sm(c), color=COL["visconf_rwc"], lw=1.2, label="calibrated confidence c")
            ax.plot(sm(v) / max(1e-9, v.max()), color=COL["vision_max"], lw=1.2, label="visual engagement v (scaled)")
            ax.set_title(f"{t['qid']} rollout {t['rollout']} · answer {t['answer']!r} · gold {t['gold']!r} · "
                         f"{'correct' if t['correct'] else 'wrong'}", fontsize=7, loc="left")
            ax.set_ylim(-0.05, 1.05)
        np.atleast_1d(axes)[0].legend(fontsize=6, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("generated token")
        fig.suptitle(T["rule"], fontsize=7, y=1.0)
        fig.tight_layout(); save(fig, rd, f"fig_{key}_trajectories")


# ----------------------------------------------------------------------------- Step 5
def adaptive_tables(key, od, rd, S):
    sel = jload(od / f"{key}_adaptive_selected.json")
    jdump(sel, rd / f"{key}_adaptive_selected.json")
    G = pd.read_csv(od / f"{key}_adaptive_dev_grid.csv")
    write_table(G.to_dict("records"), rd, f"{key}_adaptive_dev_grid")
    A = pd.read_parquet(od / f"{key}_adaptive_eval.parquet")
    E = S[(S.split == "eval") & (S.method == "sc")]
    fixed = E.groupby(["N", "qid"])[["correct", "token_cost"]].mean().groupby("N").mean()
    rows = [dict(policy=f"fixed N={N} self-consistency", accuracy_pct=pct(r.correct), mean_token_cost=r.token_cost,
                 mean_rollouts=N) for N, r in fixed.iterrows()]
    f32 = E[E.N == 32].groupby("qid").correct.mean()
    stopdist = []
    for fam, g in A.groupby("family"):
        pq_ = g.groupby("qid")[["correct", "token_cost", "n_used"]].mean()
        ch = sel["chosen"][fam]
        rows.append(dict(policy=f"adaptive {fam} (tau={ch['tau']}, gamma={ch['gamma']})", accuracy_pct=pct(pq_.correct.mean()),
                         mean_token_cost=pq_.token_cost.mean(), mean_rollouts=pq_.n_used.mean(),
                         minus_fixed32_pp=ci_txt(paired_bootstrap(pq_.correct, f32.loc[pq_.index]))))
        dist = g.n_used.value_counts(normalize=True)
        early = g[g.n_used < 32]
        wrong_early = early[~early.correct]
        stopdist.append(dict(family=fam, **{f"stop_at_{n}_pct": pct(dist.get(n, 0.0)) for n in (4, 8, 16, 32)},
                             early_stops=len(early), incorrect_early_stops=len(wrong_early),
                             incorrect_early_recovered_by_fixed32=int(wrong_early.fixed32_correct.sum())))
        bins = pd.cut(g.pool_correct_frac, [-0.01, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0],
                      labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"])
        b = g.groupby(bins, observed=False).agg(n=("correct", "size"), stop_early_pct=("n_used", lambda x: pct((x < 32).mean())),
                                                mean_rollouts=("n_used", "mean"), accuracy_pct=("correct", lambda x: pct(x.mean())))
        write_table([dict(pool_correct_fraction=str(k), **r) for k, r in b.iterrows()], rd, f"{key}_adaptive_{fam}_by_pool_correctness")
    write_table(rows, rd, f"{key}_adaptive_vs_fixed")
    write_table(stopdist, rd, f"{key}_adaptive_stopping")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(fixed.token_cost, 100 * fixed.correct, "o-", color=COL["sc"], lw=1.6, ms=3, label="fixed-N self-consistency")
    for N, r in fixed.iterrows():
        ax.annotate(f"N={N}", (r.token_cost, 100 * r.correct), fontsize=6, color=MUTED, xytext=(3, -8), textcoords="offset points")
    for fam, mk, c in (("agreement", "s", COL["sc_rwc"]), ("quality", "D", COL["visconf_rwc"])):
        g = A[A.family == fam].groupby("qid")[["correct", "token_cost"]].mean()
        ax.scatter([g.token_cost.mean()], [100 * g.correct.mean()], marker=mk, s=40, color=c, zorder=3,
                   edgecolor="white", linewidth=1, label=f"adaptive · {fam}")
    ax.set_xscale("log"); ax.set_xlabel("mean generated tokens per question"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"{DSN[key]} — held-out accuracy vs cost", fontsize=9); ax.legend(fontsize=7)
    save(fig, rd, f"fig_{key}_adaptive_cost")
    return A


# ----------------------------------------------------------------------------- qualitative (offline)
def qualitative(key, out, rd, S, A, k=3):
    P = pickle.load(open(out / "prepared" / f"{key}.pkl", "rb"))
    Q, sp = P["Q"], P["splits"]
    qm = "visconf_rwc" if key == "mathvista" else "sc_rwc"
    X = S[(S.split == "eval") & (S.N == 32) & (S.order == 0)].set_index(["qid", "method"])
    md = [f"# {DSN[key]}: qualitative cases",
          "Selection rule (fixed in advance, no gold used to choose): held-out questions in seed-6304 permutation order; "
          f"the first {k} where, at N=32 and reveal order 0, self-consistency and {LABEL[qm]} pick different answers; "
          f"and the first {k} incorrect early stops (quality-aware policy) where fixed-32 self-consistency was correct.\n"]
    md.append(f"## Selection disagreements (SC vs {LABEL[qm]})")
    n = 0
    for q in sp["eval"]:
        a, b = X.loc[(q, "sc")], X.loc[(q, qm)]
        if a.answer != b.answer:
            e = Q[q]
            votes = pd.Series([e["reps"][g] for g in e["gid"]]).value_counts().head(5).to_dict()
            md.append(f"### `{q}`\n**Q:** {e['question'][:500]}\n\n- gold: `{e['gold']}`\n- SC: `{a.answer}` "
                      f"({'✓' if a.correct else '✗'})\n- {LABEL[qm]}: `{b.answer}` ({'✓' if b.correct else '✗'})\n"
                      f"- answer-group votes among 32: {votes}\n")
            n += 1
            if n == k:
                break
    md.append("## Premature (incorrect) early stops recovered by fixed-32")
    g = A[(A.family == "quality") & (A.n_used < 32) & (~A.correct) & (A.fixed32_correct)]
    order_ = {q: i for i, q in enumerate(sp["eval"])}
    g = g.assign(pos=g.qid.map(order_)).sort_values(["pos", "order"]).drop_duplicates("qid").head(k)
    for _, r in g.iterrows():
        e = Q[r.qid]
        perm = P["orders"][r.qid][r.order]
        seen = [e["reps"][e["gid"][i]] for i in perm[:r.n_used]]
        md.append(f"### `{r.qid}` (order {r.order}, stopped at n={r.n_used})\n**Q:** {e['question'][:400]}\n\n"
                  f"- gold: `{e['gold']}`\n- revealed answers: {pd.Series(seen).value_counts().to_dict()}\n"
                  f"- correct fraction in full pool of 32: {e['correct'].mean():.2f}\n")
    (rd / f"{key}_qualitative.md").write_text("\n".join(md))


# ----------------------------------------------------------------------------- live
def live_tables(out, rd):
    ld = out / "live"
    R = [json.loads(l) for l in open(ld / "calls.jsonl")]
    if not R:
        return
    by = {}
    for r in R:
        by.setdefault(r["qid"], {})[r["call"]] = r
    LQ = jload(ld / "live_questions.json")
    qids = [x["qid"] for x in LQ]
    gold = {x["qid"]: x["gold"] for x in LQ}
    sc = MathScorer()
    per, trans = [], []
    for q in qids:
        c = by.get(q, {})
        if len(c) < 5:
            continue
        g = gold.get(q)
        par = [c[f"parallel{k}"] for k in range(3)]
        gid, reps = build_groups([p["extracted"] for p in par], sc.eq)
        win, _ = vote_winner(gid, [1, 1, 1])
        ans = dict(single=c["single"]["extracted"], parallel=None if reps[win] == NO_ANSWER else reps[win],
                   refinement=c["revision"]["extracted"], initial=par[0]["extracted"])
        corr = {k: sc.grade(g, v) for k, v in ans.items()}
        calls = dict(single=[c["single"]], parallel=par, refinement=[par[0], c["revision"]])
        for s, cl in calls.items():
            per.append(dict(qid=q, strategy=s, correct=corr[s], extracted=ans[s] is not None,
                            input_tokens=sum(x["input_tokens"] for x in cl), output_tokens=sum(x["retained_tokens"] for x in cl),
                            seconds=sum(x["seconds"] for x in cl), truncated_any=any(x["truncated"] for x in cl),
                            truncated_responses=sum(x["truncated"] for x in cl), n_responses=len(cl)))
        trans.append(dict(qid=q, initial_correct=corr["initial"], revised_correct=corr["refinement"],
                          answer_changed=(ans["initial"] != ans["refinement"])))
    P = pd.DataFrame(per)
    P.to_csv(rd / "live_per_question.csv", index=False)
    rows = []
    for s, g in P.groupby("strategy", sort=False):
        rows.append(dict(strategy=s, n_questions=len(g), accuracy_pct=pct(g.correct.mean()),
                         extraction_rate_pct=pct(g.extracted.mean()),
                         truncation_rate_responses_pct=pct(g.truncated_responses.sum() / g.n_responses.sum()),
                         mean_input_tokens=g.input_tokens.mean(), mean_output_tokens=g.output_tokens.mean(),
                         mean_seconds=g.seconds.mean(), total_minutes=g.seconds.sum() / 60))
    info = jload(ld / "live_info.json")
    peak = max(r["peak_mem_GB"] for r in R)
    for r in rows:
        r["peak_gpu_mem_GB"] = peak
    write_table(rows, rd, "live_strategies")
    T = pd.DataFrame(trans)
    tt = dict(wrong_to_correct=int((~T.initial_correct & T.revised_correct).sum()),
              correct_to_wrong=int((T.initial_correct & ~T.revised_correct).sum()),
              correct_to_correct=int((T.initial_correct & T.revised_correct).sum()),
              wrong_to_wrong=int((~T.initial_correct & ~T.revised_correct).sum()),
              answer_changed=int(T.answer_changed.sum()), n=len(T))
    jdump(dict(transitions=tt, gpu=info.get("gpu"), model=info.get("model"), revision=info.get("revision"),
               paired_bootstrap_pp={f"{a}-{b}": ci_txt(paired_bootstrap(
                   P[P.strategy == a].set_index("qid").correct.astype(float),
                   P[P.strategy == b].set_index("qid").correct.astype(float))) for a, b in
                   (("parallel", "single"), ("refinement", "single"), ("refinement", "parallel"))}),
          rd / "live_transitions.json")
    md = ["# Live: revision examples", "Selection rule: first 2 questions (live order) of each transition type.\n"]
    for kind, mask in (("wrong→correct", ~T.initial_correct & T.revised_correct),
                       ("correct→wrong", T.initial_correct & ~T.revised_correct),
                       ("wrong→wrong (changed)", ~T.initial_correct & ~T.revised_correct & T.answer_changed)):
        md.append(f"## {kind}")
        for q in T[mask].qid.head(2):
            c = by[q]
            md.append(f"### `{q}` — gold `{gold.get(q)}`\n**Initial ({c['parallel0']['extracted']}):**\n\n> "
                      + c["parallel0"]["response"][-700:].replace("\n", "\n> ")
                      + f"\n\n**Revision ({c['revision']['extracted']}):**\n\n> " + c["revision"]["response"][-900:].replace("\n", "\n> ") + "\n")
    (rd / "live_qualitative.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=default_out()); a = ap.parse_args()
    out = Path(a.out); od = out / "offline"; rd = out / "report_t3"; rd.mkdir(parents=True, exist_ok=True)
    style()
    for key in ("math", "mathvista"):
        if (od / f"{key}_selection.parquet").exists():
            S = selection(key, od, rd)
            signal_tables(key, od, rd)
            A = adaptive_tables(key, od, rd, S)
            qualitative(key, out, rd, S, A)
    if (out / "live" / "calls.jsonl").exists():
        live_tables(out, rd)
    print("report artefacts in", rd)
    for p in sorted(rd.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()

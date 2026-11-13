
import argparse, csv, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import *

COLORS = {"full_ft": "#2a78d6", "l2sp": "#eb6834", "lora4": "#1baf7a", "lora16": "#eda100",
          "oplora16": "#e87ba4", "full_ft_mid": "#6250d6", "baseline": "#8a8985"}
LABEL = {"baseline": "Baseline", "full_ft_mid": "Full FT (mid)", "full_ft": "Full FT", "l2sp": "Full FT + L2-SP",
         "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
BNAME = {"gsm8k_cot": "GSM8K", "hellaswag": "HellaSwag", "boolq": "BoolQ", "arc_easy": "ARC-Easy"}
INK, MUTED = "#0b0b0b", "#52514e"
PAIRS = [("lora4", "lora16", "Rank (LoRA-4 → LoRA-16)"), ("full_ft", "l2sp", "Regularisation (Full FT → +L2-SP)"),
         ("lora16", "oplora16", "Orthogonal protection (LoRA-16 → OPLoRA-16)"),
         ("full_ft", "lora16", "Full FT → LoRA-16"), ("full_ft_mid", "full_ft", "Full FT progression (mid → final)")]


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, rd, name):
    fig.savefig(rd / f"{name}.png"); fig.savefig(rd / f"{name}.pdf"); plt.close(fig)


def write_table(rows, rd, name, latex_cols=None):
    if not rows:
        return
    with open(rd / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    keys = list(rows[0])
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3e}" if (v != 0 and abs(v) < 0.01) or abs(v) >= 1e6 else f"{v:.2f}"
        return str(v)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    (rd / f"{name}.md").write_text("\n".join(md) + "\n")
    cols = latex_cols or keys
    tex = ["\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(c.replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\", "\\midrule"]
    tex += [" & ".join(fmt(r[c]).replace("%", "\\%").replace("_", "\\_") for c in cols) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (rd / f"{name}.tex").write_text("\n".join(tex) + "\n")


def load_scores(out):
    sc = {}
    for c, _, _ in CONDITIONS:
        p = out / "eval" / c / "scores.json"
        if p.exists():
            sc[c] = jload(p)
    return sc


def metric(sc, task, key):
    return 100 * float(sc[task]["agg"][key])


def benchmark_tables(out, rd):
    sc = load_scores(out)
    if "baseline" not in sc:
        print("no baseline scores yet"); return sc
    base = sc["baseline"]; rows, extra, gsm = [], [], []
    for c, _, _ in CONDITIONS:
        if c not in sc:
            continue
        s = sc[c]; r = {"condition": LABEL[c]}
        for t, (_, prim, ex) in BENCH.items():
            r[f"{BNAME[t]} (%)"] = metric(s, t, prim)
            r[f"{BNAME[t]} Δpp"] = metric(s, t, prim) - metric(base, t, prim)
        rows.append(r)
        e = {"condition": LABEL[c], "GSM8K strict (%)": metric(s, "gsm8k_cot", "exact_match,strict-match"),
             "HellaSwag acc raw (%)": metric(s, "hellaswag", "acc,none"), "HellaSwag acc_norm (%)": metric(s, "hellaswag", "acc_norm,none"),
             "ARC-E acc raw (%)": metric(s, "arc_easy", "acc,none"), "ARC-E acc_norm (%)": metric(s, "arc_easy", "acc_norm,none"),
             "BoolQ acc (%)": metric(s, "boolq", "acc,none")}
        extra.append(e)
        g = s["gsm8k_cot"]
        gsm.append({"condition": LABEL[c], "flexible EM (%)": 100 * g["flexible"], "strict EM (%)": 100 * g["strict"],
                    "unextractable flex (%)": 100 * g["invalid_rate_flexible"], "unextractable strict (%)": 100 * g["invalid_rate_strict"],
                    "truncated @1024 (%)": 100 * g["truncation_rate"], "flex EM on non-truncated (%)": 100 * g["flexible_on_nontruncated"],
                    "mean gen tokens": g["mean_gen_tokens"]})
    write_table(rows, rd, "table_benchmarks")
    write_table(extra, rd, "table_benchmarks_raw_vs_norm")
    write_table(gsm, rd, "table_gsm8k_extraction")
    return sc


def per_example_correct(out, cond, task):
    p = out / "eval" / cond / f"{task}_samples.jsonl"
    rows = [json.loads(l) for l in open(p)]
    key = {"gsm8k_cot": "exact_match[flexible-extract]", "hellaswag": "acc_norm", "boolq": "acc", "arc_easy": "acc_norm"}[task]
    return {r["doc_id"]: float(r.get(key) or 0) for r in rows}


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def comparison_table(out, rd, sc):
    rows = []
    for A, B, label in PAIRS:
        if A not in sc or B not in sc:
            continue
        for t, (_, prim, _) in BENCH.items():
            ca, cb = per_example_correct(out, A, t), per_example_correct(out, B, t)
            ids = sorted(set(ca) & set(cb))
            b = sum(1 for i in ids if ca[i] == 1 and cb[i] == 0); c = sum(1 for i in ids if ca[i] == 0 and cb[i] == 1)
            rows.append({"comparison": label, "benchmark": BNAME[t], "A (%)": metric(sc[A], t, prim), "B (%)": metric(sc[B], t, prim),
                         "B − A (pp)": metric(sc[B], t, prim) - metric(sc[A], t, prim), "A✓B✗": b, "A✗B✓": c,
                         "McNemar p": mcnemar_p(b, c)})
    write_table(rows, rd, "table_controlled_comparisons")


def plot_benchmarks(rd, sc):
    if "baseline" not in sc:
        return
    conds = [c for c, _, _ in CONDITIONS if c in sc and c != "baseline"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=False)
    for ax, (t, (_, prim, _)) in zip(axes, BENCH.items()):
        d = [metric(sc[c], t, prim) - metric(sc["baseline"], t, prim) for c in conds]
        ax.bar(range(len(conds)), d, color=[COLORS[c] for c in conds], width=0.7, edgecolor="white", linewidth=1)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([LABEL[c] for c in conds], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{BNAME[t]} (base {metric(sc['baseline'], t, prim):.1f}%)", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Δ vs baseline (pp)")
    fig.suptitle("Change from the unchanged Qwen3-0.6B, primary metric", fontsize=10, y=1.03)
    save(fig, rd, "fig_benchmark_deltas")


def read_log(out, m):
    p = out / "runs" / m / "metrics.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def ema(x, a=0.05):
    o, s = [], None
    for v in x:
        s = v if s is None else a * v + (1 - a) * s; o.append(s)
    return o


def plot_learning(out, rd):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    for m in METHODS:
        L = read_log(out, m)
        if not L:
            continue
        u = [r["update"] for r in L]; y = [r["sft_loss"] for r in L]
        ax.plot(u, y, color=COLORS[m], alpha=0.18, lw=0.8)
        ax.plot(u, ema(y), color=COLORS[m], lw=2, label=LABEL[m])
        ax2.plot(u, [r["lr"] for r in L], color=COLORS[m], lw=2, label=LABEL[m])
    ax.set_xlabel("optimizer update"); ax.set_ylabel("SFT loss (response tokens, nats)")
    ax.set_title("SFT loss (faint: per update, bold: EMA α=0.05)", fontsize=9); ax.legend(fontsize=7)
    ax.axvline(MID_UPDATE, color=MUTED, ls=":", lw=0.8)
    ax2.set_xlabel("optimizer update"); ax2.set_ylabel("learning rate"); ax2.set_title("LR schedule", fontsize=9)
    save(fig, rd, "fig_sft_loss")

    L = read_log(out, "l2sp")
    if L:
        u = [r["update"] for r in L]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3))
        a1.plot(u, [r["sft_loss"] for r in L], color=COLORS["l2sp"], lw=1, alpha=0.4, label="SFT loss")
        a1.plot(u, ema([r["total_loss"] for r in L]), color=INK, lw=1.5, label="total = SFT + λΩ (EMA)")
        a1.plot(u, ema([r["sft_loss"] for r in L]), color=COLORS["l2sp"], lw=2, label="SFT (EMA)")
        a1.set_xlabel("optimizer update"); a1.set_ylabel("loss"); a1.legend(fontsize=7); a1.set_title("L2-SP objective", fontsize=9)
        a2.plot(u, [r["lambda_omega"] for r in L], color=COLORS["l2sp"], lw=2)
        a2.set_xlabel("optimizer update"); a2.set_ylabel("λΩ  (λ = 1e-3)"); a2.set_title("Penalty λΩ = (λ/2)‖θ−θ₀‖²", fontsize=9)
        save(fig, rd, "fig_l2sp_penalty")


def regularisation_table(out, rd):
    rows = []
    for m in ("full_ft", "l2sp"):
        s = out / "runs" / m / "train_summary.json"
        if not s.exists():
            continue
        s = jload(s); L = read_log(out, m)
        r = {"run": LABEL[m], "‖θ−θ0‖ mid": s["displacement_l2"].get(str(MID_UPDATE), float("nan")),
             "‖θ−θ0‖ final": s["displacement_l2"].get(str(FINAL_UPDATE), float("nan")),
             "final SFT (EMA)": ema([x["sft_loss"] for x in L])[-1] if L else float("nan")}
        if m == "l2sp" and L:
            r.update({"final Ω": L[-1]["omega"], "final λΩ": L[-1]["lambda_omega"], "mean λΩ": sum(x["lambda_omega"] for x in L) / len(L),
                      "final λΩ / SFT": L[-1]["lambda_omega"] / L[-1]["sft_loss"]})
        rows.append(r)
    keys = ["run", "‖θ−θ0‖ mid", "‖θ−θ0‖ final", "final SFT (EMA)", "final Ω", "final λΩ", "mean λΩ", "final λΩ / SFT"]
    rows = [{k: r.get(k, float("nan")) for k in keys} for r in rows]
    write_table(rows, rd, "table_regularisation")


def resources_table(out, rd):
    rows = []
    for m in METHODS:
        p = out / "runs" / m / "train_summary.json"
        if not p.exists():
            continue
        s = jload(p); o = s["overhead"]
        rec = s["checkpoint_sizes_bytes"]; exp = s["export_sizes_bytes"]
        extra = []
        if "svd_preprocess_s" in o:
            extra.append(f"SVD {o['svd_preprocess_s']:.1f}s, bases {o['basis_bytes_fp32']/2**20:.0f} MB, export {o.get(f'export_s_update_{FINAL_UPDATE}', 0):.1f}s")
        if "l2sp_ref_bytes" in o:
            extra.append(f"θ0 ref {o['l2sp_ref_bytes']/2**30:.2f} GB on {o['l2sp_ref_device']}, build {o['l2sp_ref_build_s']:.1f}s, penalty {o['l2sp_penalty_s']/60:.1f} min")
        rows.append({"method": LABEL[m], "trainable params": f"{s['trainable_params']:,}",
                     "% of backbone": 100 * s["trainable_params"] / (s["total_params_in_graph"] - (0 if m in ("full_ft", "l2sp") else s["trainable_params"])),
                     "train time (h)": s["train_time_h"], "peak GPU alloc (GB)": s["peak_gpu_mem_alloc_GB"],
                     "peak GPU reserved (GB)": s["peak_gpu_mem_reserved_GB"],
                     "recovery ckpt (GB)": (max(rec.values()) / 2**30) if rec else float("nan"),
                     "final export (MB)": exp.get(str(FINAL_UPDATE), float("nan")) / 2**20,
                     "skipped updates": len(s["skipped_updates"]), "method overhead": "; ".join(extra) or "—",
                     "GPU": s["gpu"]})
    write_table(rows, rd, "table_resources")


def diagnostics(out, rd):
    dd = out / "diagnostics"
    if not (dd / "weight_space.csv").exists():
        return
    ws = list(csv.DictReader(open(dd / "weight_space.csv")))
    act = list(csv.DictReader(open(dd / "activation_summary.csv")))
    conds = [c for c in ["full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16"] if any(r["condition"] == c for r in ws)]
    rows = []
    for c in conds:
        r = {"condition": LABEL[c]}
        for l in DIAG_LAYERS:
            w = next(x for x in ws if x["condition"] == c and int(x["layer"]) == l)
            mth = next(x for x in act if x["condition"] == c and int(x["layer"]) == l and x["domain"] == "math")
            nm = next(x for x in act if x["condition"] == c and int(x["layer"]) == l and x["domain"] == "nonmath")
            r[f"L{l} ρk"] = float(w["rho_k"]); r[f"L{l} η"] = float(w["eta"])
            r[f"L{l} R math"] = float(mth["R_mean"]); r[f"L{l} R non-math"] = float(nm["R_mean"])
            r[f"L{l} R math/non"] = float(mth["R_mean"]) / max(float(nm["R_mean"]), 1e-12)
        rows.append(r)
    # scientific formatting for small values
    def f(v):
        return f"{v:.3g}" if isinstance(v, float) else v
    write_table([{k: f(v) for k, v in r.items()} for r in rows], rd, "table_diagnostics")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    x = range(len(DIAG_LAYERS)); w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        rho = [float(next(z for z in ws if z["condition"] == c and int(z["layer"]) == l)["rho_k"]) for l in DIAG_LAYERS]
        eta = [float(next(z for z in ws if z["condition"] == c and int(z["layer"]) == l)["eta"]) for l in DIAG_LAYERS]
        xs = [j + (i - len(conds) / 2 + 0.5) * w for j in x]
        axes[0].bar(xs, rho, w * 0.9, color=COLORS[c], label=LABEL[c])
        for xv, rv in zip(xs, rho):
            if rv < 1e-3:
                axes[0].text(xv, 0.005, "≈0", ha="center", va="bottom", fontsize=6, color=MUTED)
        axes[1].bar(xs, eta, w * 0.9, color=COLORS[c])
        for d, mk in (("math", "o"), ("nonmath", "^")):
            v = [float(next(z for z in act if z["condition"] == c and int(z["layer"]) == l and z["domain"] == d)["R_mean"]) for l in DIAG_LAYERS]
            axes[2].scatter(xs, v, color=COLORS[c], marker=mk, s=28, edgecolor="white", linewidth=1,
                            label=(f"{LABEL[c]}" if d == "math" else None))
    axes[0].set_title("ρk: share of ‖ΔW‖² in top-128 output subspace", fontsize=8)
    axes[1].set_title("η = ‖ΔW‖F / ‖W0‖F", fontsize=8); axes[1].set_yscale("log")
    axes[2].set_title("mean R(z) = ‖ΔWz‖/‖W0z‖  (● math, ▲ HellaSwag)", fontsize=8); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xticks(list(x)); ax.set_xticklabels([f"layer {l}" for l in DIAG_LAYERS]); ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(conds), fontsize=7, bbox_to_anchor=(0.5, -0.08))
    save(fig, rd, "fig_diagnostics")


MATH_PAT = re.compile(r"(\\boxed|\\frac|\$|=\s*-?\d|\b\d+\s*[\+\-\*/×÷]\s*\d+|the answer is|step \d|calculate|equation)", re.I)


def json_ok(text):
    t = text.strip()
    try:
        json.loads(t); strict = True
    except Exception:
        strict = False
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    lenient = strict
    if not strict and m:
        try:
            json.loads(m.group(1)); lenient = True
        except Exception:
            pass
    return strict, lenient


def repetition(text, n=4):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def qualitative(out, rd):
    qp = jload(out / "splits" / "qual_prompts.json")
    allr = []
    for c, _, _ in CONDITIONS:
        p = out / "eval" / c / "qualitative.json"
        if p.exists():
            allr += jload(p)
    if not allr:
        return
    for r in allr:
        r["json_valid_strict"], r["json_valid_lenient"] = json_ok(r["response"]) if r["format"] == "json" else (None, None)
        r["repetition_4gram"] = repetition(r["response"])
        r["math_markers"] = len(MATH_PAT.findall(r["response"]))
    grouped = {}
    for b in qp["plain_order"]:
        grouped[b] = {f: {r["condition"]: r for r in allr if r["base"] == b and r["format"] == f} for f in ("plain", "json", "template")}
    jdump(grouped, rd / "qualitative_168_grouped.json")
    # auto summary per condition
    summ = []
    for c, _, _ in CONDITIONS:
        R = [r for r in allr if r["condition"] == c]
        if not R:
            continue
        J = [r for r in R if r["format"] == "json"]
        summ.append({"condition": LABEL[c], "n": len(R), "truncated": sum(r["truncated"] for r in R),
                     "JSON valid strict": f"{sum(bool(r['json_valid_strict']) for r in J)}/{len(J)}",
                     "JSON valid lenient": f"{sum(bool(r['json_valid_lenient']) for r in J)}/{len(J)}",
                     "mean 4-gram repetition": sum(r["repetition_4gram"] for r in R) / len(R),
                     "responses with math markers": sum(r["math_markers"] > 0 for r in R),
                     "mean gen tokens": sum(r["gen_tokens"] for r in R) / len(R)})
    write_table(summ, rd, "table_qualitative_auto")
    # annotation sheet: first & last plain-order questions × 3 formats × 7 conditions = 42
    rows = []
    for b in qp["annotation_questions"]:
        for f in ("plain", "json", "template"):
            for c, _, _ in CONDITIONS:
                r = grouped[b][f].get(c)
                if r is None:
                    continue
                rows.append({"question": b, "format": f, "condition": LABEL[c], "truncated": r["truncated"],
                             "gen_tokens": r["gen_tokens"], "json_valid_strict": r["json_valid_strict"],
                             "repetition_4gram": round(r["repetition_4gram"], 3), "math_markers": r["math_markers"],
                             "format_compliance(0-2)": "", "coherence_relevance(0-2)": "", "repetition(0-2)": "",
                             "unnecessary_math(Y/N)": "", "inappropriate(Y/N)": "", "excerpt_for_report": "", "notes": "",
                             "response": r["response"]})
    with open(rd / "annotation_sheet_42.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # aligned excerpts (same question & format across all conditions) for quick reading
    md = ["# Aligned responses for the two annotated questions\n"]
    for b in qp["annotation_questions"]:
        for f in ("plain", "json", "template"):
            first = next(iter(grouped[b][f].values()))
            md.append(f"## {b} — {f}\n\n**Prompt:** {first['prompt']}\n")
            if first.get("system"):
                md.append(f"**System:** {first['system']}\n")
            for c, _, _ in CONDITIONS:
                r = grouped[b][f].get(c)
                if r:
                    flag = " [TRUNCATED]" if r["truncated"] else ""
                    md.append(f"- **{LABEL[c]}**{flag} ({r['gen_tokens']} tok): " + r["response"][:400].replace("\n", " ⏎ ") + ("…" if len(r["response"]) > 400 else ""))
            md.append("")
    (rd / "qualitative_aligned_annotated_questions.md").write_text("\n".join(md))
    return len(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=default_out()); a = ap.parse_args()
    out = Path(a.out); rd = out / "report"; rd.mkdir(parents=True, exist_ok=True)
    style()
    sc = benchmark_tables(out, rd)
    if "baseline" in sc:
        comparison_table(out, rd, sc); plot_benchmarks(rd, sc)
    plot_learning(out, rd); regularisation_table(out, rd); resources_table(out, rd)
    diagnostics(out, rd); qualitative(out, rd)
    print("report artefacts in", rd)
    for p in sorted(rd.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
"""Build every Task 2 table/figure from saved result files.

python analyze.py --out OUT [--run main]
Writes OUT/report_t2/: learning curves, final tables (both suites x four image conditions, question types,
pair diagnostics, reference NLL, correct-image advantages, pair-bootstrap CIs), swap behaviour, resources,
qualitative complementary-pair figures, audit-robustness check.
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import *

INK, MUTED = "#0b0b0b", "#52514e"
C = {"correct": "#2a78d6", "swap": "#eb6834", "mismatched": "#1baf7a", "none": "#8a8985",
     "grounding_dev": "#2a78d6", "vqa_dev": "#eb6834"}
CONDS = ["correct", "swap", "mismatched", "none"]
SUITES = ["grounding_test", "vqa_test"]


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, rd, name):
    fig.savefig(rd / f"{name}.png"); fig.savefig(rd / f"{name}.pdf"); plt.close(fig)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3e}" if v != 0 and abs(v) < 1e-3 else f"{v:.3f}"
    return str(v)


def write_table(rows, rd, name):
    if not rows:
        return
    keys = []
    for r in rows:                                   # ordered union of columns (suites have different question types)
        keys += [k for k in r if k not in keys]
    rows = [{k: r.get(k, "") for k in keys} for r in rows]
    with open(rd / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + \
         ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    (rd / f"{name}.md").write_text("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("#", "\\#").replace("&", "\\&")
    tex = ["\\begin{tabular}{" + "l" * 2 + "r" * (len(keys) - 2) + "}", "\\toprule", " & ".join(esc(k) for k in keys) + " \\\\",
           "\\midrule"] + [" & ".join(esc(fmt(r[k])) for k in keys) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    (rd / f"{name}.tex").write_text("\n".join(tex) + "\n")


def ci(p):
    return f"{100 * p[0]:.1f} [{100 * p[1]:.1f}, {100 * p[2]:.1f}]"


def ci_raw(p, d=3):
    return f"{p[0]:.{d}f} [{p[1]:.{d}f}, {p[2]:.{d}f}]"


def ema(x, a=0.02):
    o, s = [], None
    for v in x:
        s = v if s is None else a * v + (1 - a) * s; o.append(s)
    return o


# ----------------------------------------------------------------------------- learning curves
def learning_curves(run_dir, rd):
    tl = [json.loads(l) for l in open(run_dir / "logs" / "train_log.jsonl")]
    dl = [json.loads(l) for l in open(run_dir / "logs" / "dev_log.jsonl")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    u = [r["update"] for r in tl]; y = [r["loss"] for r in tl]
    ax[0, 0].plot(u, y, color=C["correct"], alpha=0.15, lw=0.6)
    ax[0, 0].plot(u, ema(y), color=C["correct"], lw=1.8, label="training loss (EMA 0.02)")
    ax[0, 0].set_title("Training loss (example-weighted, includes replay)"); ax[0, 0].set_ylabel("nats / token")
    du = [r["update"] for r in dl]
    ax[0, 1].plot(du, [r["caption_loss"] for r in dl], "o-", color=INK, ms=3, lw=1.5)
    ax[0, 1].set_title("Development caption reference loss (96 images)"); ax[0, 1].set_ylabel("nats / token")
    for s in ("grounding_dev", "vqa_dev"):
        for cond, ls in (("correct", "-"), ("swap", "--"), ("none", ":")):
            ax[1, 0].plot(du, [r[s][f"nll_{cond}"] for r in dl], ls, color=C[s], lw=1.6, label=f"{s} · {cond}")
        ax[1, 1].plot(du, [r[s]["delta_nll_swap"] for r in dl], "o-", color=C[s], ms=3, lw=1.6, label=s)
    ax[1, 0].set_title("Reference-answer NLL (dev; termination excluded)"); ax[1, 0].set_ylabel("nats / token")
    ax[1, 0].legend(fontsize=6, ncol=2)
    ax[1, 1].axhline(0, color=MUTED, lw=0.8)
    ax[1, 1].set_title("Image-swap penalty ΔNLL = NLL(partner) − NLL(correct)"); ax[1, 1].legend(fontsize=7)
    for a_ in ax.ravel():
        a_.set_xlabel("optimizer update")
        for e in STAGE_ENDS[:-1]:
            a_.axvline(e, color=MUTED, ls=":", lw=0.9)
    for a_ in ax.ravel():
        for k in range(len(STAGES)):
            a_.text((stage_start(k) + STAGE_ENDS[k]) / 2, 0.98, f"S{k + 1}", ha="center", va="top", color=MUTED,
                    fontsize=8, transform=a_.get_xaxis_transform())
    fig.tight_layout(); save(fig, rd, "fig_learning_curves")
    rows = []
    for r in dl:
        row = dict(update=r["update"], stage=r["stage"], caption_loss=r["caption_loss"])
        for s in ("grounding_dev", "vqa_dev"):
            for k in ("nll_correct", "nll_swap", "nll_none", "delta_nll_swap", "frac_delta_positive"):
                row[f"{s}.{k}"] = r[s][k]
        rows.append(row)
    write_table(rows, rd, "table_dev_curves")
    mix = defaultdict(Counter)
    for r in tl:
        mix[r["stage"]].update(r["mix"])
    comps = ["caption", "presence", "counting", "vqa"]
    write_table([dict(stage=s, **{k: m.get(k, 0) for k in comps}, total=sum(m.values()),
                      **{f"{k}_frac": m.get(k, 0) / sum(m.values()) for k in comps}) for s, m in mix.items()],
                rd, "table_mixture_counts")


# ----------------------------------------------------------------------------- final results
def load_preds(ed):
    recs = [json.loads(l) for l in open(ed / "predictions.jsonl")]
    by = defaultdict(dict)
    for r in recs:
        by[(r["suite"], r["condition"])][(r["pair_id"], r["item"])] = r
    return recs, by


def arr(by, suite, cond, pairs, key, excl=None):
    return np.array([[float(by[(suite, cond)][(p["pair_id"], k)][key]) for k in range(2)]
                     for p in pairs if not (excl and p["pair_id"] in excl)])


def pair_level(by, suite, cond, pairs, excl=None):
    pa, ident = [], []
    for p in pairs:
        if excl and p["pair_id"] in excl:
            continue
        a, b = by[(suite, cond)][(p["pair_id"], 0)], by[(suite, cond)][(p["pair_id"], 1)]
        pa.append(float(a["match_canonical"] and b["match_canonical"]))
        ident.append(float(a["prediction_norm"] == b["prediction_norm"]))
    return np.array(pa), np.array(ident)


def final_tables(out, ed, rd, excl=None, tag=""):
    recs, by = load_preds(ed)
    sd = out / "splits"
    main_rows, adv_rows, swap_rows, kind_rows = [], [], [], []
    for suite in SUITES:
        pairs = jload(sd / f"{suite}.json")
        kinds = sorted({p["kind"] for p in pairs})
        for cond in CONDS:
            acc = arr(by, suite, cond, pairs, "accuracy", excl)
            nll = arr(by, suite, cond, pairs, "ref_nll", excl)
            emp = arr(by, suite, cond, pairs, "empty", excl)
            gt = arr(by, suite, cond, pairs, "gen_time_s", excl)
            pa, ident = pair_level(by, suite, cond, pairs, excl)
            row = dict(suite=suite, condition=cond, n_pairs=len(pa),
                       accuracy=ci(pair_bootstrap(acc)), pair_accuracy=ci(pair_bootstrap(pa)),
                       identical_answer_rate=ci(pair_bootstrap(ident)), empty_rate=f"{100 * emp.mean():.1f}",
                       ref_nll=ci_raw(pair_bootstrap(nll)), gen_ms_per_answer=f"{1000 * gt.mean():.1f}")
            main_rows.append(row)
            for kd in kinds:
                sub = [p for p in pairs if p["kind"] == kd]
                kpa, kid = pair_level(by, suite, cond, sub, excl)
                kind_rows.append(dict(suite=suite, condition=cond, question_type=kd, n_pairs=len(kpa),
                                      accuracy=ci(pair_bootstrap(arr(by, suite, cond, sub, "accuracy", excl))),
                                      pair_accuracy=ci(pair_bootstrap(kpa)), identical_answer_rate=f"{100 * kid.mean():.1f}",
                                      empty_rate=f"{100 * arr(by, suite, cond, sub, 'empty', excl).mean():.1f}",
                                      ref_nll=ci_raw(pair_bootstrap(arr(by, suite, cond, sub, "ref_nll", excl)))))
        acc_c = arr(by, suite, "correct", pairs, "accuracy", excl)
        nll_c = arr(by, suite, "correct", pairs, "ref_nll", excl)
        for other in ("swap", "mismatched", "none"):
            d_acc = acc_c - arr(by, suite, other, pairs, "accuracy", excl)
            d_nll = arr(by, suite, other, pairs, "ref_nll", excl) - nll_c
            pa_c, _ = pair_level(by, suite, "correct", pairs, excl)
            pa_o, _ = pair_level(by, suite, other, pairs, excl)
            adv_rows.append(dict(suite=suite, comparison=f"correct vs {other}",
                                 accuracy_advantage_pp=ci(pair_bootstrap(d_acc)),
                                 pair_accuracy_advantage_pp=ci(pair_bootstrap(pa_c - pa_o)),
                                 nll_advantage=ci_raw(pair_bootstrap(d_nll))))
            for kd in kinds:
                sub = [p for p in pairs if p["kind"] == kd]
                a_ = arr(by, suite, "correct", sub, "accuracy", excl) - arr(by, suite, other, sub, "accuracy", excl)
                n_ = arr(by, suite, other, sub, "ref_nll", excl) - arr(by, suite, "correct", sub, "ref_nll", excl)
                adv_rows.append(dict(suite=suite, comparison=f"  {kd}: correct vs {other}",
                                     accuracy_advantage_pp=ci(pair_bootstrap(a_)), pair_accuracy_advantage_pp="",
                                     nll_advantage=ci_raw(pair_bootstrap(n_))))
        # what happens under the complementary swap?
        for kd in ["all"] + kinds:
            sub = [p for p in pairs if (kd == "all" or p["kind"] == kd) and not (excl and p["pair_id"] in excl)]
            rs = [by[(suite, "swap")][(p["pair_id"], k)] for p in sub for k in range(2)]
            rc = [by[(suite, "correct")][(p["pair_id"], k)] for p in sub for k in range(2)]
            n = len(rs)
            swap_rows.append(dict(
                suite=suite, kind=kd, n_questions=n,
                follows_partner_answer=f"{100 * np.mean([r['follows_partner'] for r in rs]):.1f}",
                keeps_original_answer=f"{100 * np.mean([r['match_canonical'] for r in rs]):.1f}",
                other_answer=f"{100 * np.mean([not r['follows_partner'] and not r['match_canonical'] and not r['empty'] for r in rs]):.1f}",
                empty=f"{100 * np.mean([r['empty'] for r in rs]):.1f}",
                prediction_changed_vs_correct_image=f"{100 * np.mean([a['prediction_norm'] != b['prediction_norm'] for a, b in zip(rs, rc)]):.1f}",
                follows_partner_given_correct_image_right=f"{100 * np.mean([s['follows_partner'] for s, c in zip(rs, rc) if c['match_canonical']] or [np.nan]):.1f}"))
    write_table(main_rows, rd, f"table_final_main{tag}")
    write_table(kind_rows, rd, f"table_final_by_type{tag}")
    write_table(adv_rows, rd, f"table_correct_image_advantage{tag}")
    write_table(swap_rows, rd, f"table_swap_behaviour{tag}")
    return recs, by


def plot_final(out, ed, rd):
    recs, by = load_preds(ed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, suite in zip(axes, SUITES):
        pairs = jload(out / "splits" / f"{suite}.json")
        kinds = sorted({p["kind"] for p in pairs})
        w = 0.8 / len(CONDS)
        for i, cond in enumerate(CONDS):
            pts, los, his = [], [], []
            for kd in kinds:
                sub = [p for p in pairs if p["kind"] == kd]
                m, lo, hi = pair_bootstrap(arr(by, suite, cond, sub, "accuracy"))
                pts.append(100 * m); los.append(100 * (m - lo)); his.append(100 * (hi - m))
            xs = np.arange(len(kinds)) + (i - len(CONDS) / 2 + 0.5) * w
            ax.bar(xs, pts, w * 0.92, color=C[cond], label=cond, yerr=[los, his], capsize=2,
                   error_kw=dict(lw=0.8, ecolor=MUTED))
        ax.set_xticks(range(len(kinds))); ax.set_xticklabels(kinds); ax.set_ylim(0, 105)
        ax.set_title(suite + (" (exact match)" if suite.startswith("grounding") else " (VQA soft accuracy)"))
        ax.set_ylabel("accuracy (%)"); ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(1.05, -0.12))
    save(fig, rd, "fig_final_accuracy_by_condition")


# ----------------------------------------------------------------------------- qualitative
def qualitative(out, ed, rd, per_kind=2):
    """Selection rule (fixed before looking at predictions): for each suite and question type, the first
    `per_kind` pairs in the saved, seeded split order."""
    from PIL import Image
    recs, by = load_preds(ed)
    md = ["# Aligned complementary-pair examples",
          f"Selection rule: for each final-test suite and question type, the first {per_kind} pairs in the saved "
          "(seed 6304) split order. Selected before inspecting predictions.\n"]
    for suite in SUITES:
        pairs = jload(out / "splits" / f"{suite}.json")
        chosen = []
        for kd in sorted({p["kind"] for p in pairs}):
            chosen += [p for p in pairs if p["kind"] == kd][:per_kind]
        fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 3.6 * len(chosen)))
        axes = np.atleast_2d(axes)
        for r, p in enumerate(chosen):
            md.append(f"## {suite} · {p['kind']} · `{p['pair_id']}`\n**Q:** {p['question']}\n")
            for k in range(2):
                it = p["items"][k]
                split, iid = it["image"].split(":")
                ip = out / "images" / split / coco_filename(split, iid)
                ax = axes[r, k]
                if ip.exists():
                    ax.imshow(Image.open(ip).convert("RGB"))
                ax.set_xticks([]); ax.set_yticks([])
                preds = {c: by[(suite, c)][(p["pair_id"], k)]["prediction"] for c in CONDS}
                ref = it["target"] + (f"  (10 refs: {', '.join(it['answers'])})" if it.get("answers") else "")
                ax.set_xlabel(f"ref: {it['target']}\n" + "\n".join(f"{c}: {preds[c]!r}" for c in CONDS), fontsize=7, loc="left")
                md.append(f"- **image {'AB'[k]}** `{it['image']}` · reference: {ref}\n  - " +
                          "\n  - ".join(f"{c}: `{preds[c]}`" for c in CONDS))
            axes[r, 0].set_title(f"{p['kind']}: {p['question']}", fontsize=8, loc="left")
            md.append("")
        fig.tight_layout(); save(fig, rd, f"fig_qualitative_{suite}")
    (rd / "qualitative_pairs.md").write_text("\n".join(md))
    caps = jload(ed / "dev_captions_generated.json")
    (rd / "dev_captions_sample.md").write_text(
        "# Generated captions on development images (first 12 in saved order)\n\n" +
        "\n".join(f"- `{c['image']}`\n  - reference: {c['reference']}\n  - generated: {c['generated']}" for c in caps[:12]))


def resources(out, run_dir, ed, rd):
    s = jload(run_dir / "train_summary.json")
    ci_ = jload(out / "features" / "cache_info.json")
    ev = jload(ed / "eval_info.json")
    rows = [dict(phase="feature caching (SigLIP, separate)", trainable_params=0, time_min=ci_["seconds"] / 60,
                 peak_gpu_GB=ci_["peak_gpu_mem_GB"], note=f"{ci_['n_images']} images, {ci_['file_bytes'] / 2**30:.2f} GB FP16")]
    for st in STAGES:
        n = st["name"]
        rows.append(dict(phase=f"train {n}", trainable_params=s["trainable_by_stage"][n]["total"],
                         time_min=s["train_time_s_by_stage"][n] / 60, peak_gpu_GB=s["peak_gpu_mem_GB_by_stage"].get(n, 0.0),
                         note=f"connector {s['trainable_by_stage'][n]['connector']:,} + LoRA {s['trainable_by_stage'][n]['lora']:,}"))
    rows.append(dict(phase="development evaluations (all)", trainable_params=0, time_min=s["dev_eval_time_s"] / 60,
                     peak_gpu_GB=float("nan"), note=f"{len(DEV_EVAL_AT)} evaluation points"))
    tot = sum(v["gen_s"] for v in ev["timing"].values())
    rows.append(dict(phase="final evaluation generation", trainable_params=0, time_min=tot / 60, peak_gpu_GB=ev["peak_gpu_mem_GB"],
                     note="; ".join(f"{k}: {1000 * v['per_answer_s']:.1f} ms/answer" for k, v in ev["timing"].items())))
    write_table(rows, rd, "table_resources")


def audit_robustness(out, ed, rd):
    f = out / "audit" / "grounding_label_audit.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    filled = [r for r in rows if r["label_ok_YN"].strip()]
    if not filled:
        print("audit sheet not filled yet; skipping robustness table")
        return None
    bad = {r["pair_id"] for r in filled if r["label_ok_YN"].strip().upper().startswith("N")}
    summ = dict(n_checked_labels=len(filled), n_bad_labels=sum(r["label_ok_YN"].strip().upper().startswith("N") for r in filled),
                n_pairs_with_bad_label=len(bad),
                by_suite_kind={f"{s}/{k}": dict(checked=sum(1 for r in filled if r["suite"] == s and r["kind"] == k),
                                                bad=sum(1 for r in filled if r["suite"] == s and r["kind"] == k
                                                        and r["label_ok_YN"].strip().upper().startswith("N")))
                               for s in ("grounding_dev", "grounding_test") for k in ("presence", "counting")})
    jdump(summ, rd / "audit_summary.json")
    if bad:
        final_tables(out, ed, rd, excl=bad, tag="_excluding_flagged_labels")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--run", default="main")
    a = ap.parse_args()
    out = Path(a.out); run_dir = out / "runs" / a.run; ed = out / "eval" / a.run
    rd = out / ("report_t2" if a.run == "main" else f"report_t2_{a.run}"); rd.mkdir(parents=True, exist_ok=True)
    style()
    learning_curves(run_dir, rd)
    if (ed / "predictions.jsonl").exists():
        final_tables(out, ed, rd); plot_final(out, ed, rd); qualitative(out, ed, rd)
        resources(out, run_dir, ed, rd); audit_robustness(out, ed, rd)
    print("report artefacts in", rd)
    for p in sorted(rd.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()

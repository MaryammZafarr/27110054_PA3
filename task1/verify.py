
import argparse, csv
from pathlib import Path

import torch

from common import *

RES = []


def check(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    RES.append(dict(check=name, status=status, detail=str(detail)))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def lines(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def data_checks(out, tok):
    p = out / "data" / "train_tokenized.pt"
    if not check("training data prepared", p.exists(), p):
        return
    d = torch.load(p); ex, ids = d["examples"], d["source_ids"]
    check("exactly 40,000 training examples", len(ex) == N_TRAIN, len(ex))
    check("source order kept (indices strictly increasing, unique)", all(a < b for a, b in zip(ids, ids[1:])))
    check("no example longer than 2,048 tokens (no truncation needed)", max(len(a) + len(b) for a, b in ex) <= MAX_LEN,
          max(len(a) + len(b) for a, b in ex))
    eot = end_of_turn_id(tok)
    check("every response ends with <|im_end|>", all(r[-1] == eot for _, r in ex))
    check("no empty response", all(len(r) > 1 for _, r in ex))
    ptxt = tok.decode(ex[0][0])
    check("training prompt uses non-thinking chat template", "<think>\n\n</think>" in ptxt and "<|im_start|>assistant" in ptxt)
    st = jload(out / "data" / "train_stats.json") if (out / "data" / "train_stats.json").exists() else {}
    check("exclusion statistics saved", {"empty", "gsm8k_overlap", "too_long"} <= set(st), st)
    try:
        from datasets import load_dataset
        gsm = {normalize_text(q) for q in load_dataset("openai/gsm8k", "main", split="test")["question"]}
        mm = load_dataset("meta-math/MetaMathQA", split="train").select(ids)
        bad = sum(normalize_text(q) in gsm or normalize_text(o or "") in gsm for q, o in zip(mm["query"], mm["original_question"]))
        check("no selected example matches a GSM8K test question (re-checked)", bad == 0, bad)
    except Exception as e:
        check("decontamination re-check", False, f"could not re-load datasets: {e}", warn=True)
    pr = jload(out / "splits" / "probe_set.json")
    check("probe set: 32 GSM8K test + 32 HellaSwag val, seed 6304",
          len(pr["gsm8k_test_ids"]) == 32 and len(pr["hellaswag_validation_ids"]) == 32 and pr["seed"] == SEED)
    qp = jload(out / "splits" / "qual_prompts.json")
    check("24 qualitative prompts (8 plain / 8 json / 8 template)", len(qp["prompts"]) == 24, qp["annotation_questions"])


def train_checks(out):
    for m, cfg in METHODS.items():
        run = out / "runs" / m
        sp = run / "train_summary.json"
        if not check(f"{m}: training finished", sp.exists()):
            continue
        s = jload(sp)
        check(f"{m}: not a smoke run", not s["smoke"])
        L = lines(run / "metrics.jsonl")
        check(f"{m}: 1,250 consecutive optimizer updates logged", [r["update"] for r in L] == list(range(1, TOTAL_UPDATES + 1)), len(L))
        if L:
            check(f"{m}: 40,000 examples seen", L[-1]["examples_seen"] == N_TRAIN)
            lr = [r["lr"] for r in L]
            peak = max(range(len(lr)), key=lambda i: lr[i]) + 1
            check(f"{m}: cosine schedule peaks after 38 warmup updates at lr={cfg['lr']}",
                  abs(max(lr) - cfg["lr"]) / cfg["lr"] < 1e-6 and peak == WARMUP + 1, f"peak at update {peak}")
            check(f"{m}: SFT loss and LR logged", all("sft_loss" in r and "lr" in r for r in L))
            sk = [r["update"] for r in L if r["skipped"]]
            check(f"{m}: skipped (inf/NaN) updates logged", True, f"{len(sk)} skipped: {sk[:10]}")
        rl = jload(run / "recovery_log.json") if (run / "recovery_log.json").exists() else {}
        check(f"{m}: recovery checkpoints at updates 312/625/937/1250", set(CKPT_UPDATES) <= set(map(int, rl)), sorted(map(int, rl)))
        need = [MID_UPDATE, FINAL_UPDATE] if m == "full_ft" else [FINAL_UPDATE]
        for u in need:
            check(f"{m}: export at update {u}", (run / "exports" / f"update_{u:04d}" / "export_info.json").exists())
        for k in ("trainable_params", "train_time_s", "peak_gpu_mem_alloc_GB", "checkpoint_sizes_bytes", "export_sizes_bytes"):
            check(f"{m}: resource field {k}", k in s and s[k] not in (None, {}, 0), s.get(k) if k != "checkpoint_sizes_bytes" else "")
        if cfg["kind"] == "full":
            check(f"{m}: all parameters trainable", s["trainable_params"] == s["total_params_in_graph"])
            for u in need:
                check(f"{m}: ‖θ−θ0‖ recorded at {u}", str(u) in s["displacement_l2"], s["displacement_l2"].get(str(u)))
        if m == "l2sp":
            check("l2sp: Ω, λΩ, total loss logged every update", all({"omega", "lambda_omega", "total_loss"} <= set(r) for r in L))
            check("l2sp: λ = 1e-3", cfg["lam"] == 1e-3)
            o = s["overhead"]
            check("l2sp: reference storage / build / penalty time recorded",
                  {"l2sp_ref_bytes", "l2sp_ref_build_s", "l2sp_penalty_s"} <= set(o), {k: o.get(k) for k in ("l2sp_ref_bytes", "l2sp_ref_device")})
            if L:
                d_final = s["displacement_l2"].get(str(FINAL_UPDATE))
                check("l2sp: displacement consistent with Ω (‖θ−θ0‖² = 2Ω)", d_final is not None and
                      abs(d_final ** 2 - 2 * L[-1]["omega"]) / max(2 * L[-1]["omega"], 1e-12) < 0.05,
                      "Ω is logged before the last step, so a small difference is expected", warn=True)
        if cfg["kind"] in ("lora", "oplora"):
            exp_n = cfg["r"] * 630_784  # Qwen3-0.6B: 28 layers × r × (q 3072 + k 2048 + v 2048 + o 3072 + 3×4096)
            check(f"{m}: adapter parameter count r×630,784", s["trainable_params"] == exp_n, f"{s['trainable_params']:,}", warn=True)
        if m == "oplora16":
            o = s["overhead"]
            check("oplora16: SVD time, basis storage, export time recorded",
                  {"svd_preprocess_s", "basis_bytes_fp32", f"export_s_update_{FINAL_UPDATE}"} <= set(o))
            c = jload(run / "exports" / f"update_{FINAL_UPDATE:04d}" / "oplora_constraint_check.json")
            check("oplora16: Uk^T B~ ≈ 0 and A~ Vk ≈ 0 for every module (FP32)",
                  c["max_out_rel"] < 1e-4 and c["max_in_rel"] < 1e-4, {k: c[k] for k in c if k.startswith("max")})
    a, b = out / "runs/lora16/train_summary.json", out / "runs/oplora16/train_summary.json"
    if a.exists() and b.exists():
        check("LoRA-16 and OPLoRA-16 share identical initial unprojected factors",
              jload(a)["overhead"]["init_A_sha"] == jload(b)["overhead"]["init_A_sha"])
    a, b = out / "runs/lora4/train_summary.json", out / "runs/lora16/train_summary.json"
    if a.exists() and b.exists():
        check("LoRA-16 has 4× the adapter parameters of LoRA-4", jload(b)["trainable_params"] == 4 * jload(a)["trainable_params"])


def eval_checks(out, tok):
    total_q = 0
    for c, meth, _ in CONDITIONS:
        d = out / "eval" / c
        sp = d / "scores.json"
        if not check(f"{c}: benchmark scores", sp.exists()):
            continue
        s = jload(sp)
        check(f"{c}: full splits (no --limit)", s.get("limit") is None)
        for t, (n, prim, _) in BENCH.items():
            rows = lines(d / f"{t}_samples.jsonl")
            check(f"{c}: {t} per-example outputs, n={n}", len(rows) == n == s[t]["n"], len(rows))
            check(f"{c}: {t} primary metric {prim}", prim in s[t]["agg"])
        g = s["gsm8k_cot"]
        check(f"{c}: GSM8K prompts use non-thinking chat template", g["nothink_prompt_ok"])
        rows = lines(d / "gsm8k_cot_samples.jsonl")
        check(f"{c}: GSM8K extracted answers + truncation flags saved",
              rows and all("extracted[flexible-extract]" in r and "truncated" in r for r in rows))
        check(f"{c}: truncation measured from generate() (not re-tokenised)", g["truncation_exact_matches"] == g["n"],
              f"{g['truncation_exact_matches']}/{g['n']}", warn=True)
        if meth and METHODS[meth]["kind"] != "full":
            mc = d / "merge_check.json"
            check(f"{c}: merged vs unmerged logits check", mc.exists() and jload(mc)["pass"], jload(mc) if mc.exists() else "missing")
        q = jload(d / "qualitative.json") if (d / "qualitative.json").exists() else []
        check(f"{c}: 24 qualitative responses", len(q) == 24, len(q)); total_q += len(q)
    check("168 qualitative responses in total", total_q == 168, total_q)
    # probe prompts must match lm-eval's prompt format
    bl = out / "eval" / "baseline"
    if (bl / "gsm8k_cot_samples.jsonl").exists():
        pr = jload(out / "splits" / "probe_set.json")
        gs = {r["doc_id"]: r["prompt"] for r in lines(bl / "gsm8k_cot_samples.jsonl")}
        hs = {r["doc_id"]: r.get("context") for r in lines(bl / "hellaswag_samples.jsonl")}
        g_ok = all(chat_prompt(tok, gsm8k_cot_user_text(x["question"])) == gs.get(x["id"]) for x in pr["gsm8k"])
        h_ok = all(x["context"] == hs.get(x["id"]) for x in pr["hellaswag"])
        check("probe prompts identical to lm-eval benchmark prompts (GSM8K)", g_ok)
        check("probe prompts identical to lm-eval benchmark prompts (HellaSwag)", h_ok)


def diag_checks(out):
    dd = out / "diagnostics"
    ws = list(csv.DictReader(open(dd / "weight_space.csv"))) if (dd / "weight_space.csv").exists() else []
    check("weight-space diagnostics: 6 checkpoints × 3 layers", len(ws) == 18, len(ws))
    ex = list(csv.DictReader(open(dd / "activation_per_example.csv"))) if (dd / "activation_per_example.csv").exists() else []
    check("activation diagnostics: 6 × 3 layers × 64 probes", len(ex) == 6 * 3 * 64, len(ex))
    op = [r for r in ws if r["condition"] == "oplora16"]
    if op:
        check("OPLoRA ρk ≈ 0 at layers 0/13/27", all(float(r["rho_k"]) < 1e-6 for r in op), [r["rho_k"] for r in op])
    if (dd / "diagnostics_info.json").exists():
        check("diagnostics use the OPLoRA training bases", jload(dd / "diagnostics_info.json")["basis_source"] == "oplora_training_bases",
              warn=True)


def report_checks(out):
    rd = out / "report"
    for f in ["table_benchmarks.csv", "table_gsm8k_extraction.csv", "table_controlled_comparisons.csv", "fig_benchmark_deltas.png",
              "fig_sft_loss.png", "fig_l2sp_penalty.png", "table_regularisation.csv", "table_resources.csv",
              "table_diagnostics.csv", "fig_diagnostics.png", "qualitative_168_grouped.json", "annotation_sheet_42.csv",
              "qualitative_aligned_annotated_questions.md"]:
        check(f"report artefact {f}", (rd / f).exists())
    if (rd / "annotation_sheet_42.csv").exists():
        n = sum(1 for _ in csv.DictReader(open(rd / "annotation_sheet_42.csv")))
        check("annotation sheet has 42 rows", n == 42, n)
    if (rd / "table_benchmarks.csv").exists():
        n = sum(1 for _ in csv.DictReader(open(rd / "table_benchmarks.csv")))
        check("benchmark table has 7 conditions", n == 7, n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=default_out()); a = ap.parse_args()
    out = Path(a.out); tok = load_tokenizer()
    for f in (data_checks, train_checks, eval_checks, diag_checks, report_checks):
        try:
            f(out, tok) if f in (data_checks, eval_checks) else f(out)
        except Exception as e:
            check(f"{f.__name__} crashed", False, repr(e))
    jdump(RES, out / "verify_report.json")
    n = {s: sum(r["status"] == s for r in RES) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed")
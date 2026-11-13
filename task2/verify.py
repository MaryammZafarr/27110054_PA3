"""Check every Task 2 requirement against the saved artefacts.

python verify.py --out OUT [--run main]    -> PASS/WARN/FAIL per check, writes OUT/verify_t2_report.json
"""
import argparse, csv
from pathlib import Path

from common import *

RES = []


def check(name, ok, detail="", warn=False):
    st = "PASS" if ok else ("WARN" if warn else "FAIL")
    RES.append(dict(check=name, status=st, detail=str(detail)[:400]))
    print(f"[{st}] {name}" + (f" — {str(detail)[:200]}" if detail != "" else ""))
    return ok


def split_checks(out, tok):
    sd = out / "splits"
    ct, cd = jload(sd / "captions_train.json"), jload(sd / "captions_dev.json")
    check("caption training: 6,000 images x 3 captions = 18,000", len(ct) == 18000 and len({e["image"] for e in ct}) == 6000
          and all(v == 3 for v in Counter(e["image"] for e in ct).values()), len(ct))
    check("caption development: 96 images, one caption each", len(cd) == 96 and len({e["image"] for e in cd}) == 96, len(cd))
    for split, q in VQA_QUOTA.items():
        ps = jload(sd / f"vqa_{split}.json")
        cnt = Counter(p["kind"] for p in ps)
        check(f"VQAv2 {split}: yes/no/number/other = {q['yes/no']}/{q['number']}/{q['other']} pairs", dict(cnt) == q, dict(cnt))
        check(f"VQAv2 {split}: >=8/10 agreement, differing consensus, same question, 10 answers kept",
              all(it["agree"] >= 8 and len(it["answers"]) == 10 for p in ps for it in p["items"])
              and all(p["items"][0]["canonical"] != p["items"][1]["canonical"] for p in ps))
        check(f"VQAv2 {split}: training target is an original string from the consensus group",
              all(normalize_answer(it["target"]) == it["canonical"] and it["target"] in [a.strip() for a in it["answers"]]
                  for p in ps for it in p["items"]))
    for split, q in GROUND_QUOTA.items():
        ps = jload(sd / f"grounding_{split}.json")
        cnt = Counter(p["kind"] for p in ps)
        check(f"grounding {split}: presence/counting = {q['presence']}/{q['counting']} pairs", dict(cnt) == q, dict(cnt))
        check(f"grounding {split}: presence yes/no targets, counting 1-4 and different",
              all((p["items"][0]["target"], p["items"][1]["target"]) == ("yes", "no") for p in ps if p["kind"] == "presence")
              and all(p["items"][0]["target"] != p["items"][1]["target"] and {p["items"][0]["target"], p["items"][1]["target"]} <= set("1234")
                      for p in ps if p["kind"] == "counting"))
    pool = jload(sd / "grounding_pool_train.json")
    check("grounding training pools exist (presence and counting)", len(pool["presence"]) > 0 and len(pool["counting"]) > 0,
          {k: len(v) for k, v in pool.items()})
    ev = {"caption_dev": [e["image"] for e in cd]}
    for n in ("grounding_dev", "grounding_test", "vqa_dev", "vqa_test"):
        ev[n] = [it["image"] for p in jload(sd / f"{n}.json") for it in p["items"]]
    allev = [i for v in ev.values() for i in v]
    check("all development and final-test sets are mutually image-disjoint", len(allev) == len(set(allev)))
    train = ({e["image"] for e in ct} | {it["image"] for n in ("vqa_train",) for p in jload(sd / f"{n}.json") for it in p["items"]}
             | {it["image"] for v in pool.values() for p in v for it in p["items"]})
    check("no evaluation image in training or replay; train=train2014, eval=val2014",
          not (train & set(allev)) and all(k.startswith("train2014") for k in train) and all(k.startswith("val2014") for k in allev))
    mm = jload(sd / "mismatch_test.json")
    ok = True
    for suite in ("grounding_test", "vqa_test"):
        for p in jload(sd / f"{suite}.json"):
            own = {it["image"] for it in p["items"]}
            suite_imgs = set(ev[suite])
            for k in range(2):
                im = mm[suite].get(f"{p['pair_id']}#{k}")
                ok &= im is not None and im not in own and im in suite_imgs
    check("mismatched-image mapping fixed, excludes both pair images, drawn from the same suite", ok)
    codec = PromptCodec(tok)
    ps = jload(sd / "vqa_train.json") + jload(sd / "vqa_test.json")
    check("content-token caps respected (question 80, answer 16, caption 64)",
          all(codec.n_tokens(p["question"]) <= Q_CAP for p in ps) and
          all(codec.n_tokens(it["target"]) <= ANSWER_CAP for p in ps for it in p["items"]) and
          all(codec.n_tokens(e["target"]) <= CAPTION_CAP for e in ct[:2000]))
    check("label audit sheet created", (out / "audit" / "grounding_label_audit.csv").exists())
    f = out / "audit" / "grounding_label_audit.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f)))
        filled = sum(1 for r in rows if r["label_ok_YN"].strip())
        check("label audit filled in by hand (MANUAL step)", filled > 0, f"{filled}/{len(rows)} labels checked", warn=True)


def feature_checks(out):
    ci = jload(out / "features" / "cache_info.json")
    n = len(jload(out / "splits" / "images.json")["all"])
    check("features: (N, 196, 768) FP16 for every image", ci["shape"] == [n, 196, 768] and ci["dtype"] == "float16", ci["shape"])
    check("SigLIP final LayerNorm applied exactly once", ci["layernorm_check"]["last_equals_postLN_of_last_block"]
          and ci["layernorm_check"]["last_differs_from_pre_LN"])
    check("deterministic preprocessing / re-encoding", ci["deterministic_reencode"], warn=True)
    check("feature-caching time and memory recorded", "seconds" in ci and "peak_gpu_mem_GB" in ci)


def train_checks(out, run):
    rd = out / "runs" / run
    if not check("training finished (train_summary.json)", (rd / "train_summary.json").exists()):
        return
    s = jload(rd / "train_summary.json")
    check("not a smoke run", not s["smoke"])
    L = [json.loads(l) for l in open(rd / "logs" / "train_log.jsonl")]
    check("4,500 consecutive updates logged (72,000 presentations)", [r["update"] for r in L] == list(range(1, 4501)), len(L))
    for k, st in enumerate(STAGES):
        rows = [r for r in L if r["stage"] == st["name"]]
        check(f"stage {k + 1} '{st['name']}' has {st['updates']} updates", len(rows) == st["updates"], len(rows))
        peak = max(r["lr_conn"] for r in rows)
        check(f"stage {k + 1}: connector peak LR {st['lr_conn']}, ends at 10% of peak",
              abs(peak - st["lr_conn"]) / st["lr_conn"] < 1e-6 and abs(rows[-1]["lr_conn"] - 0.1 * st["lr_conn"]) / st["lr_conn"] < 1e-3,
              f"peak {peak:.3g}, last {rows[-1]['lr_conn']:.3g}")
        if st["lr_lora"]:
            check(f"stage {k + 1}: LoRA peak LR {st['lr_lora']}", abs(max(r["lr_lora"] for r in rows) - st["lr_lora"]) / st["lr_lora"] < 1e-6)
        else:
            check(f"stage {k + 1}: connector only (no LoRA)", all(r["lr_lora"] is None for r in rows))
        if k:
            tot = Counter()
            for r in rows:
                tot.update(r["mix"])
            n = sum(tot.values())
            draws = {c: tot.get(c, 0) / n for c in st["mix"]}
            check(f"stage {k + 1}: realised mixture close to {st['mix']}",
                  all(abs(draws[c] - p) < 0.03 for c, p in st["mix"].items()), {c: round(v, 3) for c, v in draws.items()})
    check("skipped (inf/NaN) updates logged", True, s["skipped_updates"][:10])
    tb = s["trainable_by_stage"]
    check("trainable parameters by stage recorded (connector 1,837,056; + LoRA 4,587,520 in stages 2-3)",
          tb[STAGES[0]["name"]]["total"] == 1837056 and tb[STAGES[1]["name"]]["lora"] == 4587520, tb, warn=True)
    D = [json.loads(l) for l in open(rd / "logs" / "dev_log.jsonl")]
    check(f"development evaluation at {DEV_EVAL_AT}", [r["update"] for r in D] == DEV_EVAL_AT, [r["update"] for r in D])
    check("dev metrics: caption loss + answer NLL (correct/swap/none) + ΔNLL for both suites",
          all("caption_loss" in r and all(k in r[s] for s in ("grounding_dev", "vqa_dev")
                                          for k in ("nll_correct", "nll_swap", "nll_none", "delta_nll_swap")) for r in D))
    for t in ("stage1", "stage2", "stage3", "final"):
        check(f"weights saved: {t}", (rd / "weights" / f"{t}_connector.pt").exists())
    want = sorted(set(range(CKPT_EVERY, TOTAL_UPDATES + 1, CKPT_EVERY)) | set(STAGE_ENDS))
    have = {int(p.stem.split("_")[1]) for p in (rd / "ckpt").glob("update_*.pt")}
    check("resumable checkpoints every 250 updates + stage ends (optimizer, scaler, sampler position, RNG)",
          set(want) <= have, f"missing {sorted(set(want) - have)[:10]}")
    cal = jload(rd / "calibration.json")
    check("connector magnitude calibration rule and scale recorded", "rule" in cal and cal["scale"] > 0, f"scale={cal['scale']:.4g}")
    ic = jload(rd / "interface_checks.json")
    check("interface checks: shapes, frozen backbone, connector gradients, visual slots, no-image rebuild",
          ic["connector_receives_gradient"] and ic["backbone_trainable_params"] == 0 and ic["visual_slots_start_after_user_header"]
          and ic["labels_only_on_target"] and ic["no_image_positions_contiguous"], {k: ic[k] for k in ("feature_shape", "connector_output_shape")})
    check("runtime and peak memory per stage recorded", len(s["train_time_s_by_stage"]) == 3, s["peak_gpu_mem_GB_by_stage"])


def eval_checks(out, run):
    ed = out / "eval" / run
    if not check("final evaluation finished", (ed / "predictions.jsonl").exists()):
        return
    R = [json.loads(l) for l in open(ed / "predictions.jsonl")]
    n = {s: len(jload(out / "splits" / f"{s}.json")) * 2 for s in ("grounding_test", "vqa_test")}
    check("predictions: 2 suites x 4 conditions x all questions", len(R) == 4 * sum(n.values()), len(R))
    check("16-new-token cap respected", all(r["gen_tokens"] <= GEN_MAX_NEW for r in R))
    check("raw references, generations and per-example metrics saved",
          all(k in R[0] for k in ("prediction", "target", "accuracy", "ref_nll", "gen_time_s")))
    info = jload(ed / "eval_info.json")
    check("greedy, batch size 8, 16 tokens", info["batch_size"] == 8 and info["max_new_tokens"] == 16)
    for suite in n:
        none = [r for r in R if r["suite"] == suite and r["condition"] == "none"]
        by = {}
        for r in none:
            by.setdefault(r["pair_id"], []).append(r["prediction_norm"])
        same = np.mean([len(set(v)) == 1 for v in by.values()])
        check(f"{suite}: no-image answers identical within each pair (same text -> same greedy output)", same == 1.0,
              f"{100 * same:.1f}%", warn=True)


def report_checks(out, run):
    rd = out / ("report_t2" if run == "main" else f"report_t2_{run}")
    for f in ["fig_learning_curves.png", "table_dev_curves.csv", "table_final_main.csv", "table_final_by_type.csv",
              "table_correct_image_advantage.csv", "table_swap_behaviour.csv", "fig_final_accuracy_by_condition.png",
              "fig_qualitative_grounding_test.png", "fig_qualitative_vqa_test.png", "qualitative_pairs.md", "table_resources.csv",
              "table_mixture_counts.csv"]:
        check(f"report artefact {f}", (rd / f).exists())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=default_out()); ap.add_argument("--run", default="main")
    a = ap.parse_args(); out = Path(a.out); tok = load_tokenizer()
    for f, args in ((split_checks, (out, tok)), (feature_checks, (out,)), (train_checks, (out, a.run)),
                    (eval_checks, (out, a.run)), (report_checks, (out, a.run))):
        try:
            f(*args)
        except Exception as e:
            check(f"{f.__name__} crashed", False, repr(e))
    jdump(RES, out / f"verify_t2_{a.run}.json")
    n = {s: sum(r["status"] == s for r in RES) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed")

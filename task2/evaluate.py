"""Final evaluation of the completed Stage-3 model on both final-test suites under four image conditions
(correct / complementary swap / mismatched / no image): greedy, thinking off, 16 new tokens, batch 8,
plus teacher-forced reference NLL. Also generates captions for the 96 development images.

python evaluate.py --out OUT [--run main] [--weights final]
"""
import argparse
from pathlib import Path

from safetensors.torch import load_file

from common import *
from train import flat_items, nll_list

CONDITIONS = ["correct", "swap", "mismatched", "none"]


def load_vlm(out, run, tag, dev, amp):
    rd = Path(out) / "runs" / run
    summ = jload(rd / "train_summary.json")
    store = FeatureStore(out, summ["tokens"])
    model = load_lm(torch.float16 if amp else torch.float32, dev)
    model.config.use_cache = True
    conn = Connector(store.dim, model.config.hidden_size).to(dev)
    conn.load_state_dict(torch.load(rd / "weights" / f"{tag}_connector.pt", map_location=dev))
    lf = rd / "weights" / f"{tag}_lora.safetensors"
    if lf.exists():
        load_lora_state(model, load_file(str(lf)))
    model.eval(); conn.eval()
    return model, conn, store, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--run", default="main")
    ap.add_argument("--weights", default="final")
    a = ap.parse_args()
    out = Path(a.out); ed = out / "eval" / a.run; ed.mkdir(parents=True, exist_ok=True)
    dev = get_device(); amp = dev.type == "cuda"; set_seed()
    tok = load_tokenizer(); codec = PromptCodec(tok)
    model, conn, store, summ = load_vlm(out, a.run, a.weights, dev, amp)
    sd = out / "splits"
    mism = jload(sd / "mismatch_test.json")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    recs, timing = [], {}
    for suite in ("grounding_test", "vqa_test"):
        pairs = jload(sd / f"{suite}.json")
        items = flat_items(pairs)
        exs = [make_example(codec, it["image"], it["kind"], it["question"], it["target"]) for it in items]
        for cond in CONDITIONS:
            imgs = {"correct": [it["image"] for it in items], "swap": [it["partner"] for it in items],
                    "mismatched": [mism[suite][f"{it['pair']['pair_id']}#{it['item']}"] for it in items],
                    "none": [None] * len(items)}[cond]
            withimg = cond != "none"
            preds, lens, caps, times = [], [], [], []
            for b in range(0, len(exs), GEN_BS):
                feats = store.get(imgs[b:b + GEN_BS], dev) if withimg else None
                sync(); t0 = time.time()
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    p, n, c = greedy_generate(model, conn, codec, exs[b:b + GEN_BS], feats, store.n_tokens, withimg,
                                              GEN_MAX_NEW, dev)
                sync(); dt = time.time() - t0
                preds += p; lens += n; caps += c; times += [dt / len(p)] * len(p)
            nll = nll_list(model, conn, codec, store, exs, imgs, withimg, True, dev, amp)
            timing[f"{suite}/{cond}"] = dict(gen_s=float(np.sum(times)), per_answer_s=float(np.mean(times)))
            for it, img, pr, n, c, t, l in zip(items, imgs, preds, lens, caps, times, nll):
                p = it["pair"]; me = p["items"][it["item"]]; partner = p["items"][1 - it["item"]]
                pn = normalize_answer(pr)
                score = (vqa_soft_accuracy(pn, me["answers_norm"]) if suite.startswith("vqa") else float(pn == me["canonical"]))
                recs.append(dict(suite=suite, condition=cond, pair_id=p["pair_id"], item=it["item"], kind=p["kind"],
                                 question=p["question"], image_used=img, own_image=me["image"], target=me["target"],
                                 canonical=me["canonical"], partner_canonical=partner["canonical"],
                                 answers=me["answers"], prediction=pr, prediction_norm=pn, accuracy=score,
                                 match_canonical=pn == me["canonical"], follows_partner=pn == partner["canonical"],
                                 empty=pn == "", gen_tokens=n, hit_token_cap=c, ref_nll=l, gen_time_s=t))
            acc = np.mean([r["accuracy"] for r in recs if r["suite"] == suite and r["condition"] == cond])
            print(f"{suite:15s} {cond:10s} acc={acc:.3f} nll={np.mean(nll):.3f}", flush=True)
    with open(ed / "predictions.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    # captions for the development images (qualitative, RQ1)
    cap_dev = jload(sd / "captions_dev.json")
    cx = [make_example(codec, e["image"], "caption", None, e["target"]) for e in cap_dev]
    caps_out = []
    for b in range(0, len(cx), GEN_BS):
        feats = store.get([e["image"] for e in cx[b:b + GEN_BS]], dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            p, n, c = greedy_generate(model, conn, codec, cx[b:b + GEN_BS], feats, store.n_tokens, True, CAPTION_GEN_MAX, dev)
        caps_out += [dict(image=e["image"], reference=e["target"], generated=g, tokens=k, hit_cap=h)
                     for e, g, k, h in zip(cap_dev[b:b + GEN_BS], p, n, c)]
    jdump(caps_out, ed / "dev_captions_generated.json")
    jdump(dict(run=a.run, weights=a.weights, tokens=store.n_tokens, conditions=CONDITIONS, max_new_tokens=GEN_MAX_NEW,
               batch_size=GEN_BS, decoding="greedy, thinking disabled", timing=timing,
               peak_gpu_mem_GB=(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
               n_records=len(recs), smoke=SMOKE), ed / "eval_info.json")


if __name__ == "__main__":
    main()

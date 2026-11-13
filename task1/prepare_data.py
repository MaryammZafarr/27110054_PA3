
import argparse, random, time
from pathlib import Path

import torch
import yaml

from common import *


def build_train(tok, out):
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_norm = {normalize_text(q) for q in gsm["question"]}
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    excluded = []                      # (source index, reason)
    kept, ids, lengths = [], [], []
    eot = end_of_turn_id(tok)
    B = 2000
    for start in range(0, len(mm), B):
        rows = mm[start:start + B]
        cand = []
        for j in range(len(rows["query"])):
            q, r = rows["query"][j], rows["response"][j]
            oq = rows.get("original_question", [None] * len(rows["query"]))[j]
            if not (q or "").strip() or not (r or "").strip():
                excluded.append((start + j, "empty")); continue
            if normalize_text(q) in gsm_norm or (oq and normalize_text(oq) in gsm_norm):
                excluded.append((start + j, "gsm8k_overlap")); continue
            cand.append((start + j, q, r))
        if cand:
            p_ids = tok([chat_prompt(tok, q) for _, q, _ in cand], add_special_tokens=False)["input_ids"]
            r_ids = tok([r for _, _, r in cand], add_special_tokens=False)["input_ids"]
            for (idx, _, _), p, r in zip(cand, p_ids, r_ids):
                r = r + [eot]
                if len(p) + len(r) > MAX_LEN:
                    excluded.append((idx, "too_long")); continue
                kept.append((p, r)); ids.append(idx); lengths.append(len(p) + len(r))
                if len(kept) == N_TRAIN:
                    break
        if len(kept) == N_TRAIN:
            break
    assert len(kept) == N_TRAIN, f"only {len(kept)} eligible examples"
    last = ids[-1]                     # statistics cover exactly the source prefix [0, last]
    ex = [(i, why) for i, why in excluded if i <= last]
    stats = dict(scanned=last + 1, kept=len(kept), empty=sum(w == "empty" for _, w in ex),
                 gsm8k_overlap=sum(w == "gsm8k_overlap" for _, w in ex), too_long=sum(w == "too_long" for _, w in ex))
    assert stats["scanned"] == stats["kept"] + stats["empty"] + stats["gsm8k_overlap"] + stats["too_long"]
    jdump({"excluded": ex}, out / "data" / "excluded_ids.json")
    stats["last_source_index"] = ids[-1]
    stats["tokens_total"] = int(sum(lengths))
    stats["response_tokens_total"] = int(sum(len(r) for _, r in kept))
    stats["len_mean"], stats["len_max"] = float(sum(lengths) / len(lengths)), int(max(lengths))
    torch.save({"examples": kept, "source_ids": ids}, out / "data" / "train_tokenized.pt")
    jdump({"dataset": "meta-math/MetaMathQA", "split": "train", "source_indices": ids}, out / "splits" / "train_ids.json")
    jdump(stats, out / "data" / "train_stats.json")
    # sanity: re-check that the very first example decodes to prompt/response as expected
    p, r = kept[0]
    jdump({"prompt": tok.decode(p), "response": tok.decode(r)}, out / "data" / "example0_decoded.json")
    print("train stats:", stats)


def build_probe(out):
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    assert len(gsm) == 1319 and len(hs) == 10042
    rng = random.Random(SEED)
    g_ids = sorted(rng.sample(range(len(gsm)), 32))
    h_ids = sorted(rng.sample(range(len(hs)), 32))
    probe = {
        "seed": SEED,
        "gsm8k_test_ids": g_ids,
        "hellaswag_validation_ids": h_ids,
        "gsm8k": [{"id": i, "question": gsm[i]["question"]} for i in g_ids],
        "hellaswag": [{"id": i, "context": hellaswag_context(hs[i])} for i in h_ids],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    jdump(probe, out / "splits" / "probe_set.json")
    print("probe set saved")


def build_qual(yaml_path, out):
    data = yaml.safe_load(open(yaml_path))
    prompts, warnings = [], []
    for e in data:
        qid = e["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = re.sub(r"_(json|template)$", "", qid)
        paras = e.get("paraphrases") or [e.get("question")]
        if len(paras) != 1:
            warnings.append(f"{qid}: {len(paras)} paraphrases, using the first")
        prompts.append({"id": qid, "base": base, "format": fmt, "system": e.get("system"), "user": paras[0]})
    plain_order = [p["base"] for p in prompts if p["format"] == "plain"]
    counts = {f: sum(p["format"] == f for p in prompts) for f in ("plain", "json", "template")}
    assert len(prompts) == 24 and counts == {"plain": 8, "json": 8, "template": 8}, counts
    for b in plain_order:
        assert {p["format"] for p in prompts if p["base"] == b} == {"plain", "json", "template"}, b
    annot = [plain_order[0], plain_order[-1]]
    jdump({"prompts": prompts, "plain_order": plain_order, "annotation_questions": annot, "warnings": warnings},
          out / "splits" / "qual_prompts.json")
    print("qualitative prompts:", counts, "annotate:", annot, warnings)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--qual_yaml", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    set_seed()
    tok = load_tokenizer()
    build_probe(out)
    build_qual(a.qual_yaml, out)
    build_train(tok, out)
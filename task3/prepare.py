"""Step 0: load + join the supplied rollouts, verify token alignment, compute per-rollout scores, form answer
groups (no gold), grade groups with the fixed scorer, and save splits and reveal orders.

python prepare.py --data DATA_ROOT --out OUT [--mathvista_meta meta.json]
"""
import argparse, pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import *

KEYS = ["dataset", "sample_id", "rollout_id"]
README_TOTALS = {"math": dict(tokens=20742094, capped=523, null=521),
                 "mathvista": dict(tokens=8961703, capped=487, null=3032)}


def sorted_table(path, cols):
    t = pq.read_table(path, columns=cols)
    idx = pc.sort_indices(t, sort_keys=[("sample_id", "ascending"), ("rollout_id", "ascending")])
    return t.take(idx)


def flat(col):
    """ListArray column -> (values float64 ndarray, offsets ndarray)."""
    arr = col.combine_chunks()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    offs = np.asarray(arr.offsets)
    vals = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    return vals, offs


def seg_mean(vals, offs):
    n = np.diff(offs)
    s = np.add.reduceat(np.append(vals, 0.0), offs[:-1]) if len(n) else np.zeros(0)   # padded: safe for trailing empties
    s = np.where(n > 0, s, 0.0)                     # reduceat quirk on empty segments
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def mathvista_meta(path):
    if path:
        return jload(path)
    from datasets import load_dataset
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    return {str(r["pid"]): dict(question_type=r["question_type"], answer_type=r["answer_type"],
                                choices=r["choices"], precision=r["precision"], unit=r.get("unit"),
                                answer=r["answer"]) for r in ds}


def load_dataset_dir(root, key, checks):
    d = Path(root) / DATASETS[key]
    gen = sorted_table(d / "generations.parquet", KEYS + ["model", "question_text", "n_generated_text_tokens",
                                                          "hit_max_new_tokens", "gold_answer", "extracted_answer",
                                                          "generated_token_ids"])
    prob = sorted_table(d / "probability_metrics.parquet", KEYS + ["logp", "kl_u_p"])
    keys_gen = [gen.column(k).to_pylist() for k in KEYS]
    keys_prob = [prob.column(k).to_pylist() for k in KEYS]
    assert keys_gen == keys_prob, f"{key}: generations/probability keys differ"
    n_tok = np.asarray(gen.column("n_generated_text_tokens").to_numpy(), dtype=np.int64)
    ids_len = np.asarray(pc.list_value_length(gen.column("generated_token_ids")).to_numpy(zero_copy_only=False), dtype=np.int64)
    lp, lo = flat(prob.column("logp"))
    kl, ko = flat(prob.column("kl_u_p"))
    L, nl = seg_mean(lp, lo); K, nk = seg_mean(kl, ko)
    c = dict(rows=gen.num_rows, questions=len(set(keys_gen[1])),
             ids_len_eq_n=int((ids_len == n_tok).sum()), logp_len_eq_n=int((nl == n_tok).sum()),
             kl_len_eq_n=int((nk == n_tok).sum()), keys_joined=True,
             rollouts_per_question_32=bool(all(v == N_ROLL for v in
                                               __import__("collections").Counter(keys_gen[1]).values())),
             total_tokens=int(n_tok.sum()), capped=int(np.sum(gen.column("hit_max_new_tokens").to_numpy(zero_copy_only=False))),
             null_extractions=int(sum(clean_extraction(x) is None for x in gen.column("extracted_answer").to_pylist())),
             readme=README_TOTALS[key])
    out = dict(sample_id=keys_gen[1], rollout_id=keys_gen[2], T=n_tok, L=L, K=K,
               capped=np.asarray(gen.column("hit_max_new_tokens").to_numpy(zero_copy_only=False), dtype=bool),
               gold=gen.column("gold_answer").to_pylist(), extracted=gen.column("extracted_answer").to_pylist(),
               question_text=gen.column("question_text").to_pylist(), model=gen.column("model")[0].as_py())
    if key == "mathvista":
        att = sorted_table(d / "attention_metrics.parquet", KEYS + ["img_attn_total", "prompt_text_attn_total",
                                                                   "generated_text_attn_total", "n_generated_text_tokens"])
        assert [att.column(k).to_pylist() for k in KEYS] == keys_gen, "attention keys differ"
        im, io_ = flat(att.column("img_attn_total"))
        pr, _ = flat(att.column("prompt_text_attn_total"))
        ge, _ = flat(att.column("generated_text_attn_total"))
        cnt, _ = flat(att.column("n_generated_text_tokens"))
        den = im + pr + ge
        v = np.where(den > 0, im / np.where(den > 0, den, 1), 0.0)
        V, nv = seg_mean(v, io_)
        exp_cnt = np.concatenate([np.arange(n) for n in np.diff(io_)]) if len(cnt) else cnt
        c.update(att_len_eq_n=int((nv == n_tok).sum()), att_counts_are_0_to_N_minus_1=bool(np.array_equal(cnt, exp_cnt)),
                 first_token_generated_attention_zero=bool(np.all(ge[io_[:-1][np.diff(io_) > 0]] == 0)))
        out.update(V=V, k_vals=kl, k_offs=ko, v_vals=v)
    checks[key] = c
    return out


def per_question(key, D, meta, scorer):
    qids = sorted(set(D["sample_id"]))
    rows = {}
    for i, q in enumerate(D["sample_id"]):
        rows.setdefault(q, []).append(i)
    Q = {}
    for q in qids:
        r = rows[q]
        assert [D["rollout_id"][i] for i in r] == list(range(N_ROLL))
        raw = [clean_extraction(D["extracted"][i]) for i in r]
        gold = D["gold"][r[0]]
        if key == "math":
            gid, reps = build_groups(raw, scorer.eq)
            gcorr = [False if rep == NO_ANSWER else scorer.grade(gold, rep) for rep in reps]
            norm = raw
        else:
            m = meta[str(q)]
            norm = []
            for a in raw:
                if a is None:
                    norm.append(None)
                else:
                    z = mathvista_normalize(a, m)
                    norm.append(z if z is not None else "UNPARSEABLE:" + a)
            gid, reps = build_groups(norm, lambda a, b: a == b)
            gnorm = mathvista_gold(gold, m)
            gcorr = [rep != NO_ANSWER and not rep.startswith("UNPARSEABLE:") and rep == gnorm for rep in reps]
        e = dict(qid=q, rows=r, T=D["T"][r].astype(np.int64), L=D["L"][r], K=D["K"][r], capped=D["capped"][r],
                 gid=np.array(gid), reps=reps, gcorr=np.array(gcorr, dtype=bool),
                 correct=np.array([gcorr[g] for g in gid], dtype=bool), noans=np.array([reps[g] == NO_ANSWER for g in gid]),
                 gold=gold, answers_raw=raw, answers_norm=norm, question=D["question_text"][r[0]])
        if key == "mathvista":
            e["V"] = D["V"][r]
            e["k"] = [D["k_vals"][D["k_offs"][i]:D["k_offs"][i + 1]].astype(np.float32) for i in r]
            e["v"] = [D["v_vals"][D["k_offs"][i]:D["k_offs"][i + 1]].astype(np.float32) for i in r]
        Q[q] = e
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--mathvista_meta", default=None)
    a = ap.parse_args()
    out = Path(a.out); (out / "prepared").mkdir(parents=True, exist_ok=True)
    root = find_data_root(a.data)
    assert root is not None, "could not find qwen25-math/generations.parquet"
    t0 = time.time()
    scorer = MathScorer()
    meta = mathvista_meta(a.mathvista_meta)
    jdump(meta, out / "prepared" / "mathvista_metadata.json")
    checks, splits, info = {}, {}, {}
    for key in ("math", "mathvista"):
        D = load_dataset_dir(root, key, checks)
        Q = per_question(key, D, meta, scorer)
        qids = sorted(Q)
        sp = make_splits(qids)
        splits[key] = sp
        orders = make_orders(qids)
        with open(out / "prepared" / f"{key}.pkl", "wb") as f:
            pickle.dump(dict(Q=Q, orders=orders, splits=sp, model=D["model"]), f, protocol=4)
        n_groups = [len(e["reps"]) for e in Q.values()]
        info[key] = dict(model=D["model"], questions=len(Q), mean_groups=float(np.mean(n_groups)),
                         rollout_accuracy=float(np.mean([e["correct"].mean() for e in Q.values()])),
                         oracle_at_32=float(np.mean([e["correct"].any() for e in Q.values()])),
                         no_answer_rate=float(np.mean([e["noans"].mean() for e in Q.values()])))
        print(key, checks[key], info[key], flush=True)
        jdump(dict(dev=sp["dev"], eval=sp["eval"], permuted=sp["permuted"]), out / "splits" / f"{key}_splits.json")
        jdump(orders, out / "splits" / f"{key}_orders.json")
        del D, Q
    live = splits["math"]["eval"][:N_LIVE]
    jdump(dict(live_math_questions=live, rule="first 50 MATH evaluation questions in the seed-6304 permutation"),
          out / "splits" / "live_questions.json")
    jdump(dict(scorer=SCORER_VERSION, math_verify=MathScorer.version(), mathvista_rule="evaluation/calculate_score.py "
               "normalize_extracted_answer (lupantech/MathVista), MC mapped to option letters; unparseable non-MC "
               "extractions form their own (incorrect) group; missing -> NO_ANSWER",
               grouping="first-representative equivalence in rollout-id order, gold not consulted",
               frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")), out / "prepared" / "scorer_info.json")
    jdump(dict(checks=checks, info=info, seconds=time.time() - t0), out / "prepared" / "prepare_info.json")


if __name__ == "__main__":
    main()

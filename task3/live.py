"""Step 4: live generation on the fixed 50-question MATH subset with Qwen2.5-3B-Instruct (pinned revision, FP16,
batch size 1): single longer response vs parallel sampling vs sequential refinement, same 3,072-token allowance.
Resumable: every call is appended to calls.jsonl and skipped on rerun.

python live.py --data DATA_ROOT --out OUT [--time_budget_h 11]
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq

from common import *

CAP = dict(single=3072, parallel=1024, revision=2048)


def load_questions(root):
    t = pq.read_table(Path(root) / DATASETS["math"] / "generations.parquet",
                      columns=["sample_id", "rollout_id", "question_text", "gold_answer"]).to_pandas()
    t = t[t.rollout_id == 0].set_index("sample_id")
    sp = make_splits(t.index.tolist())
    qids = sp["eval"][:N_LIVE]
    return [dict(qid=q, question=t.loc[q, "question_text"], gold=t.loc[q, "gold_answer"]) for q in qids]


def seed_for(qi, call):
    return SEED * 1000 + qi * 10 + {"single": 0, "parallel0": 1, "parallel1": 2, "parallel2": 3, "revision": 4}[call]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--time_budget_h", type=float, default=1e9)
    ap.add_argument("--n_questions", type=int, default=N_LIVE, help="fewer only for smoke tests")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    T0 = time.time()
    out = Path(a.out); ld = out / "live"; ld.mkdir(parents=True, exist_ok=True)
    root = find_data_root(a.data)
    qs = load_questions(root)[:a.n_questions]
    jdump([dict(qid=q["qid"], gold=q["gold"]) for q in qs], ld / "live_questions.json")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rev = None if LIVE_REVISION in ("", "none") else LIVE_REVISION
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(LIVE_MODEL, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(LIVE_MODEL, revision=rev,
                                                 torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    load_s = time.time() - t_load
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    done = set()
    calls_f = ld / "calls.jsonl"
    if calls_f.exists():
        for l in open(calls_f):
            r = json.loads(l); done.add((r["qid"], r["call"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    f = open(calls_f, "a")

    def gen(qi, q, call, messages, cap):
        if (q["qid"], call) in done:
            return None
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(dev)
        seed = seed_for(qi, call)
        set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            o = model.generate(**ids, do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
                               max_new_tokens=cap, pad_token_id=tok.pad_token_id or list(eos)[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        new = o[0, ids["input_ids"].shape[1]:].tolist()
        stop_hit = any(t in eos for t in new)
        retained = new[:next((i for i, t in enumerate(new) if t in eos), len(new))]
        text = tok.decode(retained, skip_special_tokens=True)
        rec = dict(qid=q["qid"], call=call, seed=seed, cap=cap, messages=messages, prompt=prompt,
                   input_tokens=int(ids["input_ids"].shape[1]), output_tokens=len(new), retained_tokens=len(retained),
                   truncated=bool(len(new) >= cap and not stop_hit), response=text, extracted=last_boxed(text),
                   seconds=dt, peak_mem_GB=(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0)
        f.write(json.dumps(rec) + "\n"); f.flush()
        done.add((q["qid"], call))
        return rec

    def prev(qid, call):
        for l in open(calls_f):
            r = json.loads(l)
            if r["qid"] == qid and r["call"] == call:
                return r

    base = lambda q: [{"role": "system", "content": SYSTEM_MSG},
                      {"role": "user", "content": q["question"] + "\n\n" + INSTRUCTION}]
    paused = False
    for qi, q in enumerate(qs):
        if (time.time() - T0) / 3600 > a.time_budget_h:
            paused = True; print("time budget reached; rerun to resume", flush=True); break
        gen(qi, q, "single", base(q), CAP["single"])
        for k in range(3):
            gen(qi, q, f"parallel{k}", base(q), CAP["parallel"])
        init = prev(q["qid"], "parallel0")
        msgs = base(q) + [{"role": "assistant", "content": init["response"]}, {"role": "user", "content": REVIEW}]
        gen(qi, q, "revision", msgs, CAP["revision"])
        print(f"live {qi + 1}/{len(qs)} done ({(time.time() - T0) / 60:.1f} min)", flush=True)
    f.close()
    jdump(dict(model=LIVE_MODEL, revision=rev, dtype="float16" if dev.type == "cuda" else "float32", batch_size=1,
               sampling=dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0), caps=CAP,
               model_load_s=load_s, complete=not paused, n_questions=len(qs),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               torch=torch.__version__), ld / "live_info.json")


if __name__ == "__main__":
    main()

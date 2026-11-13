
import argparse, gc
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import *


def run_dir(out, method):
    return Path(out) / "runs" / method


def build_eval_model(out, cond, dev, dtype=torch.float16, merge=True):
    name, method, upd = COND[cond]
    if method is None:
        return load_model(dtype, MODEL_ID, dev, use_cache=True)
    cfg = METHODS[method]
    exp = run_dir(out, method) / "exports" / f"update_{upd:04d}"
    if cfg["kind"] == "full":
        return load_model(dtype, str(exp / "model_fp16"), dev, use_cache=True)
    sd = load_file(str(exp / "adapter.safetensors"))
    m = load_model(dtype, MODEL_ID, dev, use_cache=True)         # fresh backbone copy
    if merge:
        return merge_adapter_into(m, sd, cfg["alpha"] / cfg["r"])
    return load_unmerged_adapter(m, sd, cfg["r"], cfg["alpha"])


def probe_prompts(out, tok, n=4):
    probe = jload(Path(out) / "splits" / "probe_set.json")
    texts = [chat_prompt(tok, gsm8k_cot_user_text(x["question"])) for x in probe["gsm8k"][:n // 2]]
    texts += [x["context"] for x in probe["hellaswag"][:n // 2]]
    return texts


@torch.no_grad()
def merge_check(out, cond, tok, dev):
    res = {}
    texts = probe_prompts(out, tok)
    for prec, dt in (("fp32", torch.float32), ("fp16", torch.float16 if dev.type == "cuda" else torch.float32)):
        a = build_eval_model(out, cond, dev, dt, merge=True).eval()
        b = build_eval_model(out, cond, dev, dt, merge=False).eval()
        diffs, agree, tot, scale = [], 0, 0, []
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
            la, lb = a(ids).logits.float(), b(ids).logits.float()
            diffs.append((la - lb).abs()); scale.append(lb.abs().mean())
            agree += int((la.argmax(-1) == lb.argmax(-1)).sum()); tot += la.shape[1]
        d = torch.cat([x.flatten() for x in diffs])
        res[prec] = dict(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                         mean_abs_logit=float(torch.stack(scale).mean()), top1_agreement=agree / tot, n_positions=tot)
        del a, b; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    res["pass"] = bool(res["fp32"]["max_abs_diff"] < 1e-2 and res["fp32"]["top1_agreement"] == 1.0
                       and res["fp16"]["top1_agreement"] >= 0.98)
    return res


def make_lm(model, tok, batch_size):
    from lm_eval.models.huggingface import HFLM

    class NoThinkHFLM(HFLM):
        """HFLM whose chat template always uses enable_thinking=False and which records, per generated
        sequence, whether it hit the max-new-token cap without emitting a stop string / EOS."""
        gen_log = {}

        def apply_chat_template(self, chat_history, add_generation_prompt=True, **kw):
            return self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt, enable_thinking=False)

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            cont = super()._model_generate(context, max_length, stop, **generation_kwargs)
            att = generation_kwargs.get("attention_mask")
            n_ctx = context.shape[1]
            gen = cont[:, n_ctx:]
            stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, end_of_turn_id(self.tokenizer)}
            for i in range(cont.shape[0]):
                row_ctx = context[i][att[i].bool()] if att is not None else context[i]
                g = gen[i].tolist()
                hit_stop_tok = any(t in stop_ids for t in g)
                text = self.tokenizer.decode(g, skip_special_tokens=False)
                hit_stop_str = any(s in text for s in (stop or []))
                n_real = next((j for j, t in enumerate(g) if t in stop_ids), len(g))
                NoThinkHFLM.gen_log[tuple(row_ctx.tolist())] = dict(
                    gen_len_batch=len(g), gen_tokens_until_stop=n_real,
                    truncated=bool(len(g) >= GSM_MAX_NEW and not hit_stop_tok and not hit_stop_str))
            return cont

    lm = NoThinkHFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=4096)
    NoThinkHFLM.gen_log = {}
    return lm, NoThinkHFLM


def run_lmeval(model, tok, cond_dir, gen_bs, ll_bs, limit=None, gsm_task="gsm8k_cot",
               mc_tasks=("hellaswag", "boolq", "arc_easy"), task_manager=None):
    """task_manager / alternative task names are used only by the offline self-test."""
    import lm_eval
    from lm_eval import simple_evaluate
    from lm_eval.utils import handle_non_serializable
    seeds = dict(random_seed=SEED, numpy_random_seed=SEED, torch_random_seed=SEED, fewshot_random_seed=SEED)
    model.eval()
    out = {"lm_eval_version": lm_eval.__version__}

    # --- GSM8K: zero-shot CoT, non-thinking chat template, greedy, 1024 new tokens, stock stops/filters
    lm, cls = make_lm(model, tok, gen_bs)
    r = simple_evaluate(model=lm, tasks=[gsm_task], num_fewshot=0, apply_chat_template=True, task_manager=task_manager,
                        gen_kwargs=f"max_gen_toks={GSM_MAX_NEW},do_sample=False,temperature=0.0",
                        log_samples=True, limit=limit, bootstrap_iters=1000, **seeds)
    json.dump({k: r.get(k) for k in ("results", "configs", "versions", "n-shot", "higher_is_better", "n-samples", "config")},
              open(cond_dir / "lmeval_gsm8k.json", "w"), indent=2, default=handle_non_serializable)
    out["gsm8k_cot"] = save_gsm_samples(r["samples"][gsm_task], tok, cls.gen_log, cond_dir, lm)
    out["gsm8k_cot"]["agg"] = {k: v for k, v in r["results"][gsm_task].items() if isinstance(v, (int, float))}
    del lm; gc.collect()

    # --- likelihood benchmarks: stock completion prompts, no chat template
    lm, _ = make_lm(model, tok, ll_bs)
    r = simple_evaluate(model=lm, tasks=list(mc_tasks), num_fewshot=0, apply_chat_template=False, task_manager=task_manager,
                        log_samples=True, limit=limit, bootstrap_iters=1000, **seeds)
    json.dump({k: r.get(k) for k in ("results", "configs", "versions", "n-shot", "higher_is_better", "n-samples", "config")},
              open(cond_dir / "lmeval_mc.json", "w"), indent=2, default=handle_non_serializable)
    for name, t in zip(("hellaswag", "boolq", "arc_easy"), mc_tasks):
        out[name] = save_mc_samples(name, r["samples"][t], cond_dir)
        out[name]["agg"] = {k: v for k, v in r["results"][t].items() if isinstance(v, (int, float))}
    del lm; gc.collect()
    return out


def save_gsm_samples(samples, tok, gen_log, cond_dir, lm):
    by_doc = {}
    for s in samples:
        d = by_doc.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "question": s["doc"]["question"],
                                             "target": s["target"], "response": s["resps"][0][0]})
        f = s.get("filter", "none")
        d[f"extracted[{f}]"] = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
        d[f"exact_match[{f}]"] = s.get("exact_match")
        if "prompt" not in d:
            ctx = s["arguments"][0][0] if isinstance(s["arguments"][0], (list, tuple)) else s["arguments"]["gen_args_0"]["arg_0"]
            d["prompt"] = ctx
    exact = 0
    rows = []
    for k in sorted(by_doc):
        d = by_doc[k]
        key = tuple(lm.tokenizer(d["prompt"])["input_ids"])     # same call as lm-eval's tok_batch_encode
        g = gen_log.get(key)
        if g is not None:
            d.update(gen_tokens=g["gen_tokens_until_stop"], truncated=g["truncated"], truncation_source="generate"); exact += 1
        else:   # fallback: re-tokenise the returned text
            n = len(tok(d["response"], add_special_tokens=False)["input_ids"])
            d.update(gen_tokens=n, truncated=bool(n >= GSM_MAX_NEW - 4), truncation_source="retokenized")
        flex = d.get("extracted[flexible-extract]")
        d["extractable[flexible-extract]"] = flex not in (None, "", "[invalid]")
        d["extractable[strict-match]"] = d.get("extracted[strict-match]") not in (None, "", "[invalid]")
        d["thinking_block_in_prompt"] = "<think>\n\n</think>" in d["prompt"]
        rows.append(d)
    with open(cond_dir / "gsm8k_cot_samples.jsonl", "w") as f:
        for d in rows:
            f.write(json.dumps(d, default=str) + "\n")
    n = len(rows)
    acc = lambda key: sum(float(d.get(key) or 0) for d in rows) / max(n, 1)
    ntr = [d for d in rows if not d["truncated"]]
    return dict(n=n, flexible=acc("exact_match[flexible-extract]"), strict=acc("exact_match[strict-match]"),
                invalid_rate_flexible=1 - sum(d["extractable[flexible-extract]"] for d in rows) / max(n, 1),
                invalid_rate_strict=1 - sum(d["extractable[strict-match]"] for d in rows) / max(n, 1),
                truncation_rate=sum(d["truncated"] for d in rows) / max(n, 1),
                flexible_on_nontruncated=sum(float(d.get("exact_match[flexible-extract]") or 0) for d in ntr) / max(len(ntr), 1),
                mean_gen_tokens=sum(d["gen_tokens"] for d in rows) / max(n, 1),
                truncation_exact_matches=exact, nothink_prompt_ok=all(d["thinking_block_in_prompt"] for d in rows))


def save_mc_samples(task, samples, cond_dir):
    rows = []
    for s in sorted(samples, key=lambda x: x["doc_id"]):
        ll = [x[0][0] if isinstance(x[0], (list, tuple)) else x[0] for x in s["resps"]]
        arg0 = s["arguments"][0]
        ctx = arg0[0] if isinstance(arg0, (list, tuple)) else arg0.get("arg_0")
        row = {"doc_id": s["doc_id"], "target": s["target"], "loglik": [float(v) for v in ll],
               "pred_raw": int(np.argmax(ll)), "context": ctx}
        for m in ("acc", "acc_norm"):
            if m in s:
                row[m] = float(s[m])
        rows.append(row)
    with open(cond_dir / f"{task}_samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {"n": len(rows), "acc": sum(r.get("acc", 0) for r in rows) / max(len(rows), 1)}
    if all("acc_norm" in r for r in rows):
        res["acc_norm"] = sum(r["acc_norm"] for r in rows) / max(len(rows), 1)
    return res


@torch.no_grad()
def qualitative(model, tok, out, cond, dev):
    qp = jload(Path(out) / "splits" / "qual_prompts.json")
    stop = [end_of_turn_id(tok), tok.eos_token_id]
    res = []
    model.eval()
    for p in qp["prompts"]:
        text = chat_prompt(tok, p["user"], p.get("system"))
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
        g = model.generate(**ids, max_new_tokens=QUAL_MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                           eos_token_id=stop, pad_token_id=tok.pad_token_id)
        new = g[0, ids["input_ids"].shape[1]:].tolist()
        ended = any(t in stop for t in new)
        n_real = next((j for j, t in enumerate(new) if t in stop), len(new))
        res.append(dict(condition=cond, id=p["id"], base=p["base"], format=p["format"], system=p.get("system"),
                        prompt=p["user"], response=tok.decode(new[:n_real], skip_special_tokens=True),
                        gen_tokens=n_real, truncated=bool(len(new) >= QUAL_MAX_NEW and not ended)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--conditions", nargs="+", default=[c[0] for c in CONDITIONS])
    ap.add_argument("--parts", nargs="+", default=["mergecheck", "lmeval", "qual"])
    ap.add_argument("--gen_batch_size", type=int, default=32)
    ap.add_argument("--ll_batch_size", default="16")
    ap.add_argument("--limit", type=int, default=None, help="SMOKE TEST ONLY")
    a = ap.parse_args()
    dev = get_device(); set_seed()
    tok = load_tokenizer()
    tok.padding_side = "left"
    out = Path(a.out)
    for cond in a.conditions:
        cdir = out / "eval" / cond; cdir.mkdir(parents=True, exist_ok=True)
        meth = COND[cond][1]
        info = jload(cdir / "eval_info.json") if (cdir / "eval_info.json").exists() else {}
        if "mergecheck" in a.parts and meth and METHODS[meth]["kind"] != "full":
            mc = merge_check(out, cond, tok, dev); jdump(mc, cdir / "merge_check.json")
            print(cond, "merge check:", mc)
        if "lmeval" in a.parts or "qual" in a.parts:
            t0 = time.time()
            model = build_eval_model(out, cond, dev).eval()
            if dev.type == "cuda":
                assert next(model.parameters()).dtype == torch.float16, "evaluation must run in FP16"
            info["build_s"] = time.time() - t0
            if "lmeval" in a.parts:
                t0 = time.time()
                sc = run_lmeval(model, tok, cdir, a.gen_batch_size,
                                int(a.ll_batch_size) if a.ll_batch_size.isdigit() else a.ll_batch_size, a.limit)
                sc["limit"] = a.limit
                jdump(sc, cdir / "scores.json"); info["lmeval_s"] = time.time() - t0
                print(cond, {k: (v.get("flexible") or v.get("acc_norm") or v.get("acc")) for k, v in sc.items() if isinstance(v, dict)})
            if "qual" in a.parts:
                t0 = time.time()
                q = qualitative(model, tok, out, cond, dev); jdump(q, cdir / "qualitative.json")
                info["qual_s"] = time.time() - t0
            del model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        info["dtype"] = "float16"; info["condition"] = COND[cond]
        jdump(info, cdir / "eval_info.json")


if __name__ == "__main__":
    main()
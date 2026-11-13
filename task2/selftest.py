"""Offline unit tests on tiny random models (no downloads).  python selftest.py
Checks: VQA normalisation + soft accuracy, consensus target, example-weighted loss, visual-slot insertion,
connector gradients through a frozen LM, no-image rebuild, KV-cache greedy == full recompute, LoRA zero init,
2x2 pooling, per-stage LR schedule, pair bootstrap."""
import copy

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import common
from common import *

OK = []


def ok(name, cond, detail=""):
    OK.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail != "" else ""))


class Tok:
    """Minimal tokenizer stand-in: chars -> ids."""
    pad_token_id, eos_token_id, unk_token_id = 0, 2, 1

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 90) for c in s.replace("<|im_end|>", "\x01")]}

    def convert_tokens_to_ids(self, t):
        return 3 + 1 if t == "<|im_end|>" else 1

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<|im_start|>user\n" + msgs[-1]["content"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def main():
    torch.manual_seed(0)
    # ---- normalisation / scoring (official VQA rules)
    ok("normalise: articles, number words, punctuation", normalize_answer("Two dogs!") == "2 dogs" and
       normalize_answer("the Yes.") == "yes" and normalize_answer("a red-car") == "red car", normalize_answer("Two dogs!"))
    ok("normalise: contractions, commas between digits", normalize_answer("dont") == "don't" and normalize_answer("1,000") == "1000")
    ok("soft accuracy (10-way average): 4 of 10 -> 1.0, 3 -> 0.9, 2 -> 0.6, 0 -> 0",
       abs(vqa_soft_accuracy("x", ["x"] * 4 + ["y"] * 6) - 1.0) < 1e-9 and abs(vqa_soft_accuracy("x", ["x"] * 3 + ["y"] * 7) - 0.9) < 1e-9 and abs(vqa_soft_accuracy("x", ["x"] * 2 + ["y"] * 8) - 0.6) < 1e-9
       and vqa_soft_accuracy("z", ["x"] * 10) == 0.0)
    canon, k, tgt, _ = consensus(["Two", "2", "two", "2", "2", "two", "2", "3", "2", "two"])
    ok("consensus: normalised majority, agreement, original target string", canon == "2" and k == 9 and tgt == "2", (canon, k, tgt))

    # ---- tiny LM + codec
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
    lm = Qwen3ForCausalLM(cfg).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    codec = PromptCodec(Tok())
    dev = torch.device("cpu")
    conn = Connector(24, 64)
    exs = [make_example(codec, "a", "caption", None, "a dog"), make_example(codec, "b", "presence", "Is it?", "yes and more")]
    feats = torch.randn(2, 196, 24)
    b = build_batch(codec, exs, feats, 196, device=dev)
    n0 = len(codec.pre_ids)
    ok("visual slots right after the user header", bool(b["vmask"][0, n0:n0 + 196].all()) and not bool(b["vmask"][0, :n0].any())
       and int(b["vmask"].sum()) == 392)
    ok("labels only on answer tokens (+ end of turn)", int((b["labels"] != -100).sum()) == sum(len(e["target"]) for e in exs))

    # ---- example-weighted loss == manual mean of per-example means
    per, n = per_example_nll(lm, conn, b)
    logits = lm(inputs_embeds=embed(lm, conn, b), attention_mask=b["att"], position_ids=b["pos"]).logits[:, :-1]
    tgt = b["labels"][:, 1:]
    man = []
    for i in range(2):
        m = tgt[i] != -100
        man.append(F.cross_entropy(logits[i][m].float(), tgt[i][m]))
    ok("per-example mean NLL matches reference", torch.allclose(per, torch.stack(man), atol=1e-4), per.tolist())
    ok("loss = mean of per-example means (each example equal weight)", True)

    # ---- gradients reach the connector through the frozen LM (no torch.no_grad around the LM)
    conn.zero_grad()
    per_example_nll(lm, conn, b)[0].mean().backward()
    ok("connector receives gradients; LM params get none", conn.fc1.weight.grad is not None and float(conn.fc1.weight.grad.abs().sum()) > 0
       and all(p.grad is None for p in lm.parameters()))

    # ---- no-image: visual tokens removed, positions contiguous, text identical
    bn = build_batch(codec, exs, None, 196, with_image=False, device=dev)
    L0, L1 = int(b["att"][0].sum()), int(bn["att"][0].sum())
    ids_img = b["ids"][0, :L0][~b["vmask"][0, :L0]]
    ok("no-image rebuild: 196 fewer tokens, same text ids, contiguous positions",
       L0 - L1 == 196 and torch.equal(ids_img, bn["ids"][0, :L1]) and torch.equal(bn["pos"][0, :L1], torch.arange(L1)))

    # ---- KV-cache greedy == argmax with full recomputation; left padding invariance
    exs_q = [make_example(codec, "a", "vqa", "What?", None), make_example(codec, "b", "vqa", "What colour is it here?", None)]
    lm.config.use_cache = True
    outs, lens, caps = greedy_generate(lm, conn, codec, exs_q, feats, 196, True, 6, dev)
    ref = []
    for i, ex in enumerate(exs_q):
        bb = build_batch(codec, [ex], feats[i:i + 1], 196, with_target=False, device=dev)
        emb = embed(lm, conn, bb); ids = []
        cur_emb, att, pos = emb, bb["att"], bb["pos"]
        for _ in range(6):
            lg = lm(inputs_embeds=cur_emb, attention_mask=att, position_ids=pos).logits[0, -1]
            t = int(lg.argmax())
            if t in (codec.eot, 2):
                break
            ids.append(t)
            cur_emb = torch.cat([cur_emb, lm.get_input_embeddings()(torch.tensor([[t]]))], 1)
            att = torch.cat([att, torch.ones(1, 1, dtype=att.dtype)], 1)
            pos = torch.arange(att.shape[1])[None]
        ref.append(codec.tok.decode(ids).strip())
    ok("batched KV-cache greedy (left padded) == unbatched full recompute", outs == ref, (outs, ref))

    # ---- LoRA zero init leaves the function unchanged; only q/k/v/o wrapped
    before = lm(inputs_embeds=embed(lm, conn, b), attention_mask=b["att"], position_ids=b["pos"]).logits
    lm2 = copy.deepcopy(lm); attach_lora(lm2)
    after = lm2(inputs_embeds=embed(lm2, conn, b), attention_mask=b["att"], position_ids=b["pos"]).logits
    ok("LoRA (B=0) leaves outputs unchanged", torch.allclose(before, after, atol=1e-5))
    names = {n.split(".")[-1] for n, _ in lora_modules(lm2)}
    ok("LoRA targets exactly q/k/v/o_proj in every layer", names == set(LORA_CFG["targets"]) and len(lora_modules(lm2)) == 4 * 2)
    exp = 2 * 16 * ((64 + 64) + (64 + 32) * 2 + (64 + 64))
    ok("LoRA parameter count r*(din+dout)", sum(p.numel() for _, m in lora_modules(lm2) for p in (m.lora_A, m.lora_B)) == exp)

    # ---- 2x2 pooling for the 49-token variant
    x = torch.arange(196 * 2, dtype=torch.float32).view(1, 196, 2)
    pooled = x.view(1, 7, 2, 7, 2, 2).mean((2, 4)).reshape(1, 49, 2)
    ok("2x2 block average of the 14x14 grid", torch.allclose(pooled[0, 0], (x[0, 0] + x[0, 1] + x[0, 14] + x[0, 15]) / 4))

    # ---- LR schedule: 5% warmup, cosine to 10% of peak
    n = 2250
    m = [lr_mult(s, n) for s in range(n)]
    ok("LR: warmup reaches peak at 5%, ends at 10% of peak", abs(max(m) - 1) < 1e-9 and abs(m[-1] - 0.1) < 1e-9
       and m.index(max(m)) == round(0.05 * n) - 1, (m.index(max(m)), m[-1]))

    # ---- bootstrap
    v = np.array([[1, 0], [1, 1], [0, 0], [1, 1]] * 10)
    p, lo, hi = pair_bootstrap(v, n_boot=2000)
    ok("pair bootstrap: point = mean, CI brackets it", abs(p - v.mean()) < 1e-12 and lo <= p <= hi, (p, lo, hi))
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    raise SystemExit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()

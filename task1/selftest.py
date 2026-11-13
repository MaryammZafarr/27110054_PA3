
import copy
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import common
from common import *

torch.manual_seed(0)
OK = []


def ok(name, cond, detail=""):
    OK.append(cond); print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))


def tiny():
    cfg = Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=192, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=32, max_position_embeddings=256, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).float().eval(); m.config.use_cache = False
    return m


def main():
    m = tiny()
    # ---- chunked masked loss == reference CE over response tokens
    ex = [([1, 2, 3, 4], [5, 6, 7, 299]), ([8, 9], [10, 11, 12, 13, 14, 299])]
    ids, lab, att = collate(ex, 0, torch.device("cpu"))
    s, n = sft_loss_sum(m, ids, lab, att, chunk=3)
    logits = m(input_ids=ids, attention_mask=att).logits[:, :-1]
    ref = F.cross_entropy(logits.reshape(-1, 300), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    ok("chunked masked CE == full-logit CE", torch.allclose(s, ref, atol=1e-4), f"{s.item():.5f} vs {ref.item():.5f}")
    ok("only response + end-of-turn tokens counted", n == sum(len(r) for _, r in ex), n)
    ok("prompt and padding masked", int((lab != -100).sum()) == n)

    # ---- effective-batch normalisation: grads from 2 microbatches/total == one big batch mean
    m2 = copy.deepcopy(m); m2.zero_grad()
    for e in ex:
        i, l, a = collate([e], 0, torch.device("cpu")); s1, _ = sft_loss_sum(m2, i, l, a); (s1 / n).backward()
    g_acc = m2.model.layers[0].mlp.down_proj.weight.grad.clone()
    m3 = copy.deepcopy(m); m3.zero_grad(); s2, _ = sft_loss_sum(m3, ids, lab, att); (s2 / n).backward()
    ok("token-normalised accumulation == full-batch token mean", torch.allclose(g_acc, m3.model.layers[0].mlp.down_proj.weight.grad, atol=1e-6))

    # ---- LoRA zero-init leaves outputs unchanged; parameter count
    base_logits = m(ids, attention_mask=att).logits
    ml = copy.deepcopy(m)
    for p in ml.parameters():
        p.requires_grad_(False)
    attach_adapters(ml, 4, 8, 0.0)
    ok("LoRA with B=0 reproduces base logits", torch.allclose(ml(ids, attention_mask=att).logits, base_logits, atol=1e-5))
    exp = 3 * 4 * ((64 + 128) + (64 + 64) * 2 + (128 + 64) + (64 + 192) * 2 + (192 + 64))
    ntr = sum(p.numel() for p in ml.parameters() if p.requires_grad)
    ok("LoRA trainable params = r × Σ(din+dout) over 7 targets × layers", ntr == exp, f"{ntr} vs {exp}")
    ok("7 target modules per layer wrapped", len(adapter_modules(ml)) == 21)

    # ---- LoRA-16 and OPLoRA-16 identical initial factors
    mo = copy.deepcopy(m)
    for p in mo.parameters():
        p.requires_grad_(False)
    bases = compute_bases(mo, 8)
    i_op = attach_adapters(mo, 16, 32, 0.0, bases)
    m16 = copy.deepcopy(m)
    for p in m16.parameters():
        p.requires_grad_(False)
    i_16 = attach_adapters(m16, 16, 32, 0.0)
    ok("LoRA-16 and OPLoRA-16 share unprojected init", all(torch.equal(i_op[k], i_16[k]) for k in i_16))

    # ---- OPLoRA constraints after random training-like updates
    for _, mod in adapter_modules(mo):
        with torch.no_grad():
            mod.lora_B.normal_(); mod.lora_A.normal_()
    worst = 0.0
    for n_, mod in adapter_modules(mo):
        A, B = mod.effective_factors(); dW = mod.delta_weight()
        worst = max(worst, float((mod.Uk.T @ dW).norm() / dW.norm()), float((dW @ mod.Vk).norm() / dW.norm()),
                    float((mod.Uk.T @ B).norm() / B.norm()), float((A @ mod.Vk).norm() / A.norm()))
        # projector equivalence with dense formulation
        W0 = mod.base.weight.float(); U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        Pu = torch.eye(W0.shape[0]) - U[:, :8] @ U[:, :8].T; Pv = torch.eye(W0.shape[1]) - Vh[:8].T @ Vh[:8]
        dense = mod.scale * Pu @ mod.lora_B @ mod.lora_A @ Pv
        assert torch.allclose(dense, dW, atol=1e-4)
    ok("OPLoRA: ΔW, B~, A~ orthogonal to protected subspaces", worst < 1e-5, f"max rel {worst:.2e}")
    ok("OPLoRA effective factors == dense (I-UkUk^T) B A (I-VkVk^T)", True)
    ok("OPLoRA gradient flows to raw A and B", True if mo(ids).logits.sum().backward() is None and
       all(m_.lora_A.grad is not None and m_.lora_B.grad is not None for _, m_ in adapter_modules(mo)) else False)

    # ---- merge == unmerged
    sd = export_adapter_state(mo)
    merged = merge_adapter_into(copy.deepcopy(m), sd, 2.0)
    unm = load_unmerged_adapter(copy.deepcopy(m), sd, 16, 32)
    with torch.no_grad():
        a_, b_ = merged(ids).logits, unm(ids).logits
    ok("merged (projected factors) == unmerged logits (FP32)", torch.allclose(a_, b_, atol=1e-4), f"max {float((a_-b_).abs().max()):.1e}")
    with torch.no_grad():
        live = mo.eval()(ids).logits
    ok("exported effective factors reproduce the live OPLoRA model", torch.allclose(live, b_, atol=1e-4))

    # ---- L2-SP: added gradient equals d/dθ [λ/2 ||θ-θ0||²]; tied params counted once
    mf = copy.deepcopy(m); ref = {k: v.detach().clone() for k, v in mf.named_parameters()}
    with torch.no_grad():
        for p in mf.parameters():
            p.add_(0.01 * torch.randn_like(p))
    names = [k for k, _ in mf.named_parameters()]
    ok("tied embedding/lm_head counted once in named_parameters", not any("lm_head" in k for k in names), len(names))
    lam = 1e-3
    omega = 0.5 * sum(((p - ref[k]) ** 2).sum() for k, p in mf.named_parameters())
    (lam * omega).backward()
    g = {k: p.grad.clone() for k, p in mf.named_parameters()}
    ok("λ(θ−θ0) equals autograd of λΩ", all(torch.allclose(g[k], lam * (p.detach() - ref[k]), atol=1e-9) for k, p in mf.named_parameters()))
    ok("Ω includes biases/norm (1-D) params", any(p.ndim == 1 for _, p in mf.named_parameters()))

    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    raise SystemExit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
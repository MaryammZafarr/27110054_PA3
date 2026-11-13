
import argparse, csv
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import *

DIAG_CONDS = ["full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]


def probe_texts(out, tok):
    probe = jload(Path(out) / "splits" / "probe_set.json")
    items = [("math", x["id"], chat_prompt(tok, gsm8k_cot_user_text(x["question"]))) for x in probe["gsm8k"]]
    items += [("nonmath", x["id"], x["context"]) for x in probe["hellaswag"]]
    return items


@torch.no_grad()
def cache_activations(out, tok, dev):
    path = Path(out) / "diagnostics" / "probe_activations.pt"
    if path.exists():
        return torch.load(path)
    model = load_model(torch.float32, MODEL_ID, dev).eval()
    caught = {}
    hooks = [model.get_submodule(down_proj_name(l)).register_forward_hook(
        lambda mod, inp, outp, l=l: caught.__setitem__(l, inp[0].detach())) for l in DIAG_LAYERS]
    items = probe_texts(out, tok)
    Z = {l: [] for l in DIAG_LAYERS}
    for _, _, text in items:
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
        att = enc["attention_mask"][0]
        last = int(att.nonzero().max())                     
        model(**enc)
        for l in DIAG_LAYERS:
            Z[l].append(caught[l][0, last].float().cpu())
    for h in hooks:
        h.remove()
    W0 = {l: model.get_submodule(down_proj_name(l)).weight.detach().float().cpu() for l in DIAG_LAYERS}
    obj = dict(Z={l: torch.stack(v) for l, v in Z.items()}, W0=W0,
               domain=[d for d, _, _ in items], ids=[i for _, i, _ in items], texts=[t for _, _, t in items])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    del model
    return obj


def bases_for(out, W0):
    f = Path(out) / "runs" / "oplora16" / "bases_k128.pt"
    src = "oplora_training_bases" if f.exists() else "recomputed_svd_fp32"
    saved = torch.load(f) if f.exists() else None
    B = {}
    for l in DIAG_LAYERS:
        n = down_proj_name(l)
        if saved is not None:
            U, V = saved[n]
        else:
            U, S, Vh = torch.linalg.svd(W0[l], full_matrices=False); U, V = U[:, :PROTECT_K], Vh[:PROTECT_K].T
        B[l] = (U.float(), V.float())
    return B, src


def delta_w(out, cond, W0):
    _, method, upd = COND[cond]
    cfg = METHODS[method]
    exp = Path(out) / "runs" / method / "exports" / f"update_{upd:04d}"
    if cfg["kind"] == "full":
        W = load_file(str(exp / "diag_down_proj_fp32.safetensors"))
        return {l: W[down_proj_name(l) + ".weight"].float() - W0[l] for l in DIAG_LAYERS}
    sd = load_file(str(exp / "adapter.safetensors"))
    s = cfg["alpha"] / cfg["r"]
    return {l: s * sd[down_proj_name(l) + ".eff_B"].float() @ sd[down_proj_name(l) + ".eff_A"].float() for l in DIAG_LAYERS}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=default_out()); a = ap.parse_args()
    out = Path(a.out); dd = out / "diagnostics"; dd.mkdir(parents=True, exist_ok=True)
    dev = get_device(); tok = load_tokenizer()
    act = cache_activations(out, tok, dev)
    W0, Z, dom = act["W0"], act["Z"], act["domain"]
    bases, src = bases_for(out, W0)
    ws_rows, ex_rows, sum_rows = [], [], []
    for cond in DIAG_CONDS:
        try:
            dW = delta_w(out, cond, W0)
        except FileNotFoundError as e:
            print("skip", cond, e); continue
        for l in DIAG_LAYERS:
            U, V = bases[l]; D = dW[l]
            nD = float(D.norm())
            rho = float((U.T @ D).pow(2).sum() / (D.pow(2).sum() + 1e-30)) if nD > 0 else float("nan")
            rho_in = float((D @ V).pow(2).sum() / (D.pow(2).sum() + 1e-30)) if nD > 0 else float("nan")
            eta = nD / float(W0[l].norm())
            ws_rows.append(dict(condition=cond, layer=l, rho_k=rho, rho_k_input_side=rho_in, eta=eta,
                                dW_fro=nD, W0_fro=float(W0[l].norm()), k=PROTECT_K, basis_source=src))
            z = Z[l]                                                       # (64, 3072)
            R = (z @ D.T).norm(dim=1) / ((z @ W0[l].T).norm(dim=1) + 1e-8)
            for i, r in enumerate(R.tolist()):
                ex_rows.append(dict(condition=cond, layer=l, domain=dom[i], example_id=act["ids"][i], R=r))
            for d in ("math", "nonmath"):
                v = torch.tensor([r for i, r in enumerate(R.tolist()) if dom[i] == d])
                sum_rows.append(dict(condition=cond, layer=l, domain=d, n=len(v), R_mean=float(v.mean()),
                                     R_median=float(v.median()), R_std=float(v.std())))
    for name, rows in (("weight_space.csv", ws_rows), ("activation_per_example.csv", ex_rows),
                       ("activation_summary.csv", sum_rows)):
        if rows:
            with open(dd / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    jdump(dict(basis_source=src, layers=DIAG_LAYERS, k=PROTECT_K, n_math=dom.count("math"), n_nonmath=dom.count("nonmath"),
               conditions=sorted({r["condition"] for r in ws_rows})), dd / "diagnostics_info.json")
    for r in ws_rows:
        print(f"{r['condition']:12s} L{r['layer']:2d} rho_k={r['rho_k']:.4f} eta={r['eta']:.5f}")


if __name__ == "__main__":
    main()
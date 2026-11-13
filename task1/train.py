
import argparse, glob, sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import get_cosine_schedule_with_warmup

from common import *


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--resume", action="store_true",
                    help="no-op, kept for compatibility: resuming from the latest recovery checkpoint is automatic")
    ap.add_argument("--fresh", action="store_true", help="ignore existing recovery checkpoints and start from update 0")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="also write a recovery checkpoint every N updates (0 = only the standard schedule)")
    ap.add_argument("--keep_all_recovery", action="store_true",
                    help="keep all four recovery checkpoints (default keeps only the latest to save disk)")
    ap.add_argument("--l2sp_ref_device", default="cuda", choices=["cuda", "cpu"],
                    help="where theta_0 is kept for L2-SP (cpu = less GPU memory, more transfer)")
    ap.add_argument("--prune_recovery_at_end", action="store_true",
                    help="delete the final recovery checkpoint after a finished run (its size is already logged)")
    ap.add_argument("--smoke_updates", type=int, default=0, help="SMOKE TEST ONLY: train on this many updates")
    ap.add_argument("--model_id", default=MODEL_ID)
    return ap.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    a = parse()
    cfg = METHODS[a.method]
    out = Path(a.out); run = out / "runs" / a.method
    (run / "recovery").mkdir(parents=True, exist_ok=True); (run / "exports").mkdir(exist_ok=True)
    dev = get_device(); amp = dev.type == "cuda"
    set_seed(SEED)

    data = torch.load(out / "data" / "train_tokenized.pt")["examples"]
    total = TOTAL_UPDATES
    ckpts, mid = list(CKPT_UPDATES), MID_UPDATE
    if a.smoke_updates:
        total = a.smoke_updates
        ckpts = sorted({max(1, round(total * q)) for q in (0.25, 0.5, 0.75, 1.0)}); mid = ckpts[1] if len(ckpts) > 1 else ckpts[0]
    assert len(data) >= total * EFF_BS
    tok = load_tokenizer(a.model_id)

    overhead = {}
    if cfg["kind"] == "full":
        model = load_model(torch.float32, a.model_id, dev)           # FP32 master params
    else:
        model = load_model(torch.float16 if amp else torch.float32, a.model_id, dev)   # frozen FP16 backbone
        for p in model.parameters():
            p.requires_grad_(False)
        bases = None
        if cfg["kind"] == "oplora":
            sync(); t0 = time.time()
            bases = compute_bases(model, cfg["k"])
            sync(); overhead["svd_preprocess_s"] = time.time() - t0
            overhead["basis_bytes_fp32"] = int(sum(U.numel() * 4 + V.numel() * 4 for U, V in bases.values()))
            t0 = time.time()
            torch.save({n: (U.cpu(), V.cpu()) for n, (U, V) in bases.items()}, run / "bases_k128.pt")
            overhead["basis_save_s"] = time.time() - t0
            overhead["basis_file_bytes"] = dir_size(run / "bases_k128.pt")
        inits = attach_adapters(model, cfg["r"], cfg["alpha"], cfg["dropout"], bases)
        overhead["init_A_sha"] = sha(torch.cat([inits[n].flatten() for n in sorted(inits)]))
        if bases is not None:
            del bases
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for _, p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in trainable if p.ndim >= 2]
    no_decay = [p for n, p in trainable if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], betas=BETAS, eps=EPS,
                            fused=amp)   # fused CUDA kernel: no extra full-size temporaries (FP32 states either way)
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, total)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    assert all(p.dtype == torch.float32 for _, p in trainable), "trainable params must be FP32"

    lam = cfg.get("lam", 0.0)
    ref = None
    if lam > 0:
        sync(); t0 = time.time()
        ref_dev = torch.device(a.l2sp_ref_device if amp else "cpu")
        ref = {n: p.detach().clone().float().to(ref_dev) for n, p in trainable}
        if ref_dev.type == "cpu" and amp:
            ref = {n: t.pin_memory() for n, t in ref.items()}
        sync(); overhead["l2sp_ref_build_s"] = time.time() - t0
        overhead["l2sp_ref_bytes"] = int(sum(t.numel() * 4 for t in ref.values()))
        overhead["l2sp_ref_device"] = ref_dev.type
        overhead["l2sp_n_unique_params"] = len(ref)
        overhead["l2sp_penalty_s"] = 0.0

    state = dict(method=a.method, update=0, train_time_s=0.0, peak_alloc=0, peak_reserved=0,
                 skipped=[], ckpt_sizes={}, export_sizes={}, displacement={}, smoke=bool(a.smoke_updates))
    log_path = run / "metrics.jsonl"
    if not a.fresh:
        cands = sorted(glob.glob(str(run / "recovery" / "update_*.pt")))
        if cands:
            ck = torch.load(cands[-1], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"], strict=False)
            opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
            random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
            torch.set_rng_state(ck["rng"]["torch"])
            if torch.cuda.is_available() and ck["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
            state = ck["state"]; overhead.update(ck.get("overhead", {}))
            assert ck["data_position"] == state["update"] * EFF_BS
            # drop log lines written after this checkpoint
            if log_path.exists():
                lines = [l for l in open(log_path) if json.loads(l)["update"] <= state["update"]]
                open(log_path, "w").writelines(lines)
            print(f"resumed from {cands[-1]} at update {state['update']}")
        else:
            print("no recovery checkpoint found; starting fresh")
    if state["update"] == 0:
        for f in (log_path, run / "recovery_log.json"):
            if f.exists():
                f.unlink()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pad = tok.pad_token_id
    logf = open(log_path, "a")
    print(f"{a.method}: trainable={n_trainable:,} total={n_total:,} lr={cfg['lr']} lam={lam} start={state['update']}/{total}")

    for upd in range(state["update"], total):
        sync(); t0 = time.time()
        batch = data[upd * EFF_BS:(upd + 1) * EFF_BS]
        n_tok = sum(len(r) for _, r in batch)             
        sft = 0.0
        for mb in range(GRAD_ACCUM):
            ids, lab, att = collate(batch[mb * MICRO_BS:(mb + 1) * MICRO_BS], pad, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                loss_sum, _ = sft_loss_sum(model, ids, lab, att)
            loss = loss_sum / n_tok
            scaler.scale(loss).backward()
            sft += loss.item()
        scaler.unscale_(opt)                                
        rec = {}
        if ref is not None:                                 
            tp = time.time(); omega = 0.0
            with torch.no_grad():
                for n, p in trainable:
                    d = p.detach() - ref[n].to(p.device, non_blocking=True)
                    omega += 0.5 * float((d * d).sum())
                    p.grad.add_(d, alpha=lam)
            sync(); overhead["l2sp_penalty_s"] += time.time() - tp
            rec.update(omega=omega, lambda_omega=lam * omega, total_loss=sft + lam * omega)
        gnorm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], CLIP))
        lr = sched.get_last_lr()[0]
        s_before = scaler.get_scale()
        scaler.step(opt); scaler.update()
        skipped = amp and scaler.get_scale() < s_before
        opt.zero_grad(set_to_none=True); sched.step()
        sync(); dt = time.time() - t0
        state["train_time_s"] += dt; state["update"] = upd + 1
        if skipped:
            state["skipped"].append(upd + 1)
        rec = dict(update=upd + 1, examples_seen=(upd + 1) * EFF_BS, sft_loss=sft, lr=lr, grad_norm=gnorm,
                   loss_scale=s_before, skipped=bool(skipped), n_resp_tokens=n_tok, step_time_s=dt, **rec)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if (upd + 1) % 25 == 0 or upd == 0:
            print(f"[{a.method}] upd {upd+1}/{total} sft={sft:.4f} lr={lr:.2e} gn={gnorm:.2f}"
                  + (f" omega={rec['omega']:.4f}" if 'omega' in rec else "") + (" SKIPPED" if skipped else ""), flush=True)
        if torch.cuda.is_available():
            state["peak_alloc"] = max(state["peak_alloc"], torch.cuda.max_memory_allocated())
            state["peak_reserved"] = max(state["peak_reserved"], torch.cuda.max_memory_reserved())

        u = upd + 1
        if (u == mid and a.method == "full_ft") or u == total:
            export(model, cfg, a.method, run, u, tok, state, overhead, ref, a.model_id)
            model.train()
        if u in ckpts:
            save_recovery(model, cfg, opt, sched, scaler, state, overhead, run, u, a.keep_all_recovery)
        elif a.ckpt_every and u % a.ckpt_every == 0 and u < total:
            save_recovery(model, cfg, opt, sched, scaler, state, overhead, run, u, a.keep_all_recovery, milestone=False)

    logf.close()
    side = run / "recovery_log.json"
    if side.exists():
        state["ckpt_sizes"] = {k: v["bytes"] for k, v in jload(side).items()}
    summary = dict(method=a.method, config={**cfg, "seed": SEED, "micro_bs": MICRO_BS, "grad_accum": GRAD_ACCUM,
                   "eff_bs": EFF_BS, "updates": total, "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "betas": BETAS,
                   "eps": EPS, "clip": CLIP, "max_len": MAX_LEN, "fp16_amp": amp, "grad_ckpt": True,
                   "targets": TARGETS if cfg["kind"] != "full" else "all"},
                   trainable_params=n_trainable, total_params_in_graph=n_total,
                   train_time_s=state["train_time_s"], train_time_h=state["train_time_s"] / 3600,
                   peak_gpu_mem_alloc_GB=state["peak_alloc"] / 2**30, peak_gpu_mem_reserved_GB=state["peak_reserved"] / 2**30,
                   skipped_updates=state["skipped"], checkpoint_sizes_bytes=state["ckpt_sizes"],
                   export_sizes_bytes=state["export_sizes"], displacement_l2=state["displacement"],
                   overhead=overhead, smoke=state["smoke"],
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   torch=torch.__version__)
    jdump(summary, run / "train_summary.json")
    if a.prune_recovery_at_end and state["update"] == total:
        for f in (run / "recovery").glob("update_*.pt"):
            f.unlink()
        print("pruned recovery checkpoints (sizes kept in recovery_log.json)")
    print(json.dumps(summary, indent=1, default=str))


@torch.no_grad()
def displacement(model, cfg, trainable_ref, model_id):
    """||theta - theta_0||_2 over all unique parameters (FP32)."""
    if cfg["kind"] != "full":
        return math.sqrt(sum(float(m.delta_weight().pow(2).sum()) for _, m in adapter_modules(model)))
    if trainable_ref is not None:
        return math.sqrt(sum(float((p.detach() - trainable_ref[n].to(p.device)).pow(2).sum())
                             for n, p in model.named_parameters()))
    base = load_model(torch.float32, model_id, torch.device("cpu"))
    bp = dict(base.named_parameters())
    tot = sum(float((p.detach().cpu() - bp[n]).pow(2).sum()) for n, p in model.named_parameters())
    del base
    return math.sqrt(tot)


@torch.no_grad()
def export(model, cfg, method, run, u, tok, state, overhead, ref, model_id):
    d = run / "exports" / f"update_{u:04d}"; d.mkdir(parents=True, exist_ok=True)
    model.eval()
    sync(); t0 = time.time()
    if cfg["kind"] == "full":
        sd = {k: v.detach().to(torch.float16).cpu() for k, v in model.state_dict().items()}
        if getattr(model.config, "tie_word_embeddings", False):
            sd.pop("lm_head.weight", None)          # tied to embed_tokens; stored once
        model.save_pretrained(d / "model_fp16", state_dict=sd, safe_serialization=True)
        tok.save_pretrained(d / "model_fp16")
        diag = {down_proj_name(l) + ".weight": model.get_submodule(down_proj_name(l)).weight.detach().float().cpu().contiguous()
                for l in DIAG_LAYERS}
        save_file(diag, str(d / "diag_down_proj_fp32.safetensors"))
    else:
        sd = export_adapter_state(model)
        save_file(sd, str(d / "adapter.safetensors"))
        jdump(dict(r=cfg["r"], alpha=cfg["alpha"], scale=cfg["alpha"] / cfg["r"], targets=TARGETS,
                   kind=cfg["kind"], k=cfg.get("k")), d / "adapter_config.json")
        if cfg["kind"] == "oplora":  
            resid = []
            for n, m in adapter_modules(model):
                if m.Uk.shape[1] >= m.Uk.shape[0] or m.Vk.shape[1] >= m.Vk.shape[0]:
                    continue       
                A, B = m.effective_factors(); dW = m.scale * B @ A
                resid.append(dict(module=n,
                                  out_rel=float((m.Uk.T @ B).norm() / (B.norm() + 1e-12)),
                                  in_rel=float((A @ m.Vk).norm() / (A.norm() + 1e-12)),
                                  dW_out_rel=float((m.Uk.T @ dW).norm() / (dW.norm() + 1e-12)),
                                  dW_in_rel=float((dW @ m.Vk).norm() / (dW.norm() + 1e-12))))
            jdump(dict(max_out_rel=max(r["out_rel"] for r in resid), max_in_rel=max(r["in_rel"] for r in resid),
                       max_dW_out_rel=max(r["dW_out_rel"] for r in resid), max_dW_in_rel=max(r["dW_in_rel"] for r in resid),
                       per_module=resid), d / "oplora_constraint_check.json")
    sync(); dt = time.time() - t0
    overhead[f"export_s_update_{u}"] = dt
    state["export_sizes"][str(u)] = dir_size(d)
    t0 = time.time()
    state["displacement"][str(u)] = displacement(model, cfg, ref, model_id)
    overhead[f"displacement_s_update_{u}"] = time.time() - t0
    jdump({"update": u, "examples_seen": u * EFF_BS, "displacement_l2": state["displacement"][str(u)]}, d / "export_info.json")
    print(f"exported {d} ({state['export_sizes'][str(u)]/2**20:.1f} MB), ||theta-theta0||={state['displacement'][str(u)]:.4f}")


def tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def make_room(ck, run, p):
    """Atomic save needs the old and the new checkpoint on disk at once. If that would exceed the disk budget
    (T1_DISK_BUDGET_GB, counted over the whole /kaggle/working tree; unset = no check), drop the older rolling
    checkpoints first. That briefly gives up atomicity but keeps the run from dying with 'No space left'."""
    budget = float(os.environ.get("T1_DISK_BUDGET_GB", 0)) * 2**30
    if not budget:
        return
    used = dir_size(run.parents[2])                            # <work>/<out>/runs/<method> -> <work>
    need = tensor_bytes(ck)
    olds = [f for f in p.parent.glob("update_*.pt") if f != p]
    if used + need > budget and olds:
        print(f"WARNING: disk {used/2**30:.1f} GB used + {need/2**30:.1f} GB checkpoint > budget "
              f"{budget/2**30:.0f} GB; removing {len(olds)} older checkpoint(s) before saving", flush=True)
        for f in olds:
            f.unlink()


def save_recovery(model, cfg, opt, sched, scaler, state, overhead, run, u, keep_all, milestone=True):
    if cfg["kind"] == "full":
        msd = model.state_dict()
    else:
        msd = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    ck = dict(model=msd, optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(),
              rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                       cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
              data_position=u * EFF_BS, state=state, overhead=overhead)
    p = run / "recovery" / f"update_{u:04d}.pt"
    for stale in p.parent.glob("*.tmp"):        # left behind by a crash mid-save
        stale.unlink()
    make_room(ck, run, p)
    t0 = time.time()
    torch.save(ck, str(p) + ".tmp"); os.replace(str(p) + ".tmp", p)
    size = dir_size(p)
    if milestone:   # periodic (--ckpt_every) saves stay out of the logs so verify.py still sees the four milestones
        side = run / "recovery_log.json"
        log = jload(side) if side.exists() else {}
        log[str(u)] = dict(bytes=size, save_s=time.time() - t0, examples_seen=u * EFF_BS)
        jdump(log, side)
        state["ckpt_sizes"][str(u)] = size
    milestones = {int(k) for k in jload(run / "recovery_log.json")} if (run / "recovery_log.json").exists() else set()
    for old in (run / "recovery").glob("update_*.pt"):
        if old != p and (not keep_all or int(old.stem.split("_")[1]) not in milestones):
            old.unlink()
    print(f"recovery checkpoint {p} ({size/2**30:.2f} GB)")


if __name__ == "__main__":
    main()
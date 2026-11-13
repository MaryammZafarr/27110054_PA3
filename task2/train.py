"""Train the SigLIP-Qwen VLM through the 3-stage curriculum (4,500 updates), with development evaluation,
resumable checkpoints every 250 updates and connector/adapter weights after each stage.

python train.py --out OUT [--run main|tok49] [--tokens 196|49] [--resume] [--time_budget_h 11]
"""
import argparse, glob, sys
from pathlib import Path

from safetensors.torch import save_file, load_file

from common import *


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, out, codec):
        sd = Path(out) / "splits"
        self.cap_train = [make_example(codec, e["image"], "caption", None, e["target"], dict(ex_id=e["ex_id"]))
                          for e in jload(sd / "captions_train.json")]
        pools = jload(sd / "grounding_pool_train.json")
        self.pairs = {"presence": pools["presence"], "counting": pools["counting"], "vqa": jload(sd / "vqa_train.json")}
        self.pair_ex = {k: [pair_examples(codec, p) for p in v] for k, v in self.pairs.items()}
        self.cap_dev = [make_example(codec, e["image"], "caption", None, e["target"], dict(ex_id=e["ex_id"]))
                        for e in jload(sd / "captions_dev.json")]
        self.dev = {"grounding_dev": jload(sd / "grounding_dev.json"), "vqa_dev": jload(sd / "vqa_dev.json")}
        self.codec = codec


COMPONENTS = ["caption", "presence", "counting", "vqa"]


def make_plan(data, rng):
    """Whole-run sampling plan (deterministic from seed 6304) so resuming restores the sampler position exactly."""
    n_cap = len(data.cap_train)
    plan = {}
    s1 = STAGES[0]["updates"] * EFF_BS
    order = np.concatenate([rng.permutation(n_cap) for _ in range(math.ceil(s1 / n_cap))])[:s1]
    plan["stage0"] = order.astype(np.int64)
    for k in (1, 2):
        mix = STAGES[k]["mix"]
        names = list(mix); probs = np.array([mix[n] for n in names])
        n_draw = STAGES[k]["updates"] * (EFF_BS // 2)                  # each draw = 2 examples
        comp = rng.choice(len(names), size=n_draw, p=probs)
        rows = np.zeros((n_draw, 3), dtype=np.int64)
        for i, c in enumerate(comp):
            name = names[c]
            rows[i, 0] = COMPONENTS.index(name)
            if name == "caption":
                rows[i, 1], rows[i, 2] = rng.integers(n_cap), rng.integers(n_cap)
            else:
                rows[i, 1], rows[i, 2] = rng.integers(len(data.pairs[name])), -1
        plan[f"stage{k}"] = rows
    return plan


def update_examples(data, plan, u):
    k = stage_of(u); s = u - stage_start(k)
    if k == 0:
        idx = plan["stage0"][s * EFF_BS:(s + 1) * EFF_BS]
        return [data.cap_train[i] for i in idx], {"caption": EFF_BS}
    rows = plan[f"stage{k}"][s * (EFF_BS // 2):(s + 1) * (EFF_BS // 2)]
    exs, mix = [], Counter()
    for c, i, j in rows:
        name = COMPONENTS[c]
        if name == "caption":
            exs += [data.cap_train[i], data.cap_train[j]]
        else:
            exs += data.pair_ex[name][i]
        mix[name] += 2
    return exs, dict(mix)


# ----------------------------------------------------------------------------- evaluation helpers
def flat_items(pairs):
    out = []
    for p in pairs:
        for k, it in enumerate(p["items"]):
            out.append(dict(pair=p, item=k, image=it["image"], partner=p["items"][1 - k]["image"],
                            kind=p["kind"], question=p["question"], target=it["target"]))
    return out


@torch.no_grad()
def nll_list(model, conn, codec, store, examples, images, with_image, exclude_eot, dev, amp, bs=16):
    vals = []
    for b in range(0, len(examples), bs):
        exs = examples[b:b + bs]
        feats = store.get(images[b:b + bs], dev) if with_image else None
        batch = build_batch(codec, exs, feats, store.n_tokens, with_image=with_image, exclude_eot=exclude_eot, device=dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            v, _ = per_example_nll(model, conn, batch)
        vals += v.tolist()
    return vals


def dev_eval(model, conn, data, store, dev, amp, u):
    model.eval(); conn.eval()
    codec = data.codec
    t0 = time.time()
    res = dict(update=u, stage=STAGES[stage_of(max(u - 1, 0))]["name"] if u else "before_training")
    cap = nll_list(model, conn, codec, store, data.cap_dev, [e["image"] for e in data.cap_dev], True, False, dev, amp)
    res["caption_loss"] = float(np.mean(cap))
    for suite, pairs in data.dev.items():
        items = flat_items(pairs)
        exs = [make_example(codec, it["image"], it["kind"], it["question"], it["target"]) for it in items]
        r = {}
        r["correct"] = nll_list(model, conn, codec, store, exs, [it["image"] for it in items], True, True, dev, amp)
        r["swap"] = nll_list(model, conn, codec, store, exs, [it["partner"] for it in items], True, True, dev, amp)
        r["none"] = nll_list(model, conn, codec, store, exs, [None] * len(exs), False, True, dev, amp)
        d = np.array(r["swap"]) - np.array(r["correct"])
        res[suite] = dict(nll_correct=float(np.mean(r["correct"])), nll_swap=float(np.mean(r["swap"])),
                          nll_none=float(np.mean(r["none"])), delta_nll_swap=float(d.mean()),
                          frac_delta_positive=float((d > 0).mean()), by_kind={})
        for kind in sorted({it["kind"] for it in items}):
            m = np.array([it["kind"] == kind for it in items])
            res[suite]["by_kind"][kind] = dict(nll_correct=float(np.mean(np.array(r["correct"])[m])),
                                               nll_swap=float(np.mean(np.array(r["swap"])[m])),
                                               nll_none=float(np.mean(np.array(r["none"])[m])),
                                               delta_nll_swap=float(d[m].mean()))
    res["seconds"] = time.time() - t0
    model.train(); conn.train()
    return res


# ----------------------------------------------------------------------------- training
def make_optimizer(conn, model, k):
    st = STAGES[k]
    groups = []
    cw = [p for n, p in conn.named_parameters() if p.ndim >= 2]
    cb = [p for n, p in conn.named_parameters() if p.ndim < 2]
    groups += [dict(params=cw, weight_decay=WEIGHT_DECAY, lr=st["lr_conn"], peak=st["lr_conn"], name="connector"),
               dict(params=cb, weight_decay=0.0, lr=st["lr_conn"], peak=st["lr_conn"], name="connector_bias")]
    if st["lr_lora"] is not None:
        lp = [p for _, m in lora_modules(model) for p in (m.lora_A, m.lora_B)]
        groups.append(dict(params=lp, weight_decay=WEIGHT_DECAY, lr=st["lr_lora"], peak=st["lr_lora"], name="lora"))
    return torch.optim.AdamW(groups, betas=BETAS, eps=EPS)


def trainable_params(conn, model):
    return [p for p in conn.parameters() if p.requires_grad] + [p for _, m in lora_modules(model) for p in (m.lora_A, m.lora_B)]


@torch.no_grad()
def calibrate(model, conn, data, store, dev, n_img=256):
    """Rule: scale the connector output so that, at initialisation, the mean L2 norm of visual tokens equals the
    mean L2 norm of Qwen input embeddings of the TRAINING text tokens (token-frequency weighted). Training data only."""
    E = model.get_input_embeddings().weight.float()
    cnt = Counter()
    for ex in data.cap_train:
        cnt.update(ex["post"]); cnt.update(ex["target"])
    for name in ("presence", "counting", "vqa"):
        for pr in data.pair_ex[name]:
            for ex in pr:
                cnt.update(ex["post"]); cnt.update(ex["target"])
    ids = torch.tensor(list(cnt.keys()), device=E.device)
    w = torch.tensor([cnt[i] for i in cnt], dtype=torch.float32, device=E.device)
    text_norm = float((E[ids].norm(dim=-1) * w).sum() / w.sum())
    keys = sorted({e["image"] for e in data.cap_train})[:n_img]
    norms = []
    conn.scale.fill_(1.0)
    for b in range(0, len(keys), 32):
        z = conn(store.get(keys[b:b + 32], dev).float())
        norms.append(z.norm(dim=-1).flatten().cpu())
    conn_norm = float(torch.cat(norms).mean())
    scale = text_norm / conn_norm
    conn.scale.fill_(scale)
    return dict(rule="scale = mean ||text embedding|| (training text tokens, frequency weighted) / mean ||connector output|| "
                     f"at init ({len(keys)} training images x {store.n_tokens} tokens)",
                text_embedding_norm=text_norm, connector_output_norm_at_init=conn_norm, scale=scale,
                n_text_token_types=len(cnt), n_text_tokens=int(w.sum()), n_images=len(keys))


def save_ckpt(path, conn, model, opt, scaler, state, k):
    ck = dict(conn=conn.state_dict(), lora=lora_state(model) if lora_modules(model) else None,
              opt=opt.state_dict(), opt_stage=k, scaler=scaler.state_dict(), state=state,
              rng=dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                       cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    torch.save(ck, str(path) + ".tmp"); os.replace(str(path) + ".tmp", path)


def save_weights(wd, tag, conn, model):
    wd.mkdir(parents=True, exist_ok=True)
    torch.save(conn.state_dict(), wd / f"{tag}_connector.pt")
    if lora_modules(model):
        save_file(lora_state(model), str(wd / f"{tag}_lora.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--run", default=None, help="run name (default: main for 196 tokens, tok49 for 49)")
    ap.add_argument("--tokens", type=int, default=196, choices=[196, 49])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time_budget_h", type=float, default=1e9, help="checkpoint and exit cleanly after this")
    a = ap.parse_args()
    T0 = time.time()
    out = Path(a.out); run = out / "runs" / (a.run or ("main" if a.tokens == 196 else "tok49"))
    for d in ("ckpt", "weights", "logs"):
        (run / d).mkdir(parents=True, exist_ok=True)
    dev = get_device(); amp = dev.type == "cuda"
    set_seed()
    tok = load_tokenizer(); codec = PromptCodec(tok)
    data = Data(out, codec)
    store = FeatureStore(out, a.tokens)
    model = load_lm(torch.float16 if amp else torch.float32, dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    torch.manual_seed(SEED)
    conn = Connector(store.dim, model.config.hidden_size).to(dev)               # FP32 trainable
    plan_path = run / "plan.npz"
    if plan_path.exists():
        plan = dict(np.load(plan_path))
    else:
        plan = make_plan(data, np.random.default_rng(SEED)); np.savez(plan_path, **plan)
    ckpts = sorted((run / "ckpt").glob("update_*.pt"))        # one resumable checkpoint every 250 updates (all kept)
    logp, devp = run / "logs" / "train_log.jsonl", run / "logs" / "dev_log.jsonl"
    state = dict(update=0, stage_time={}, stage_peak_mem={}, dev_time=0.0, skipped=[], trainable_by_stage={},
                 tokens=a.tokens, smoke=SMOKE)
    ck = None
    if a.resume and ckpts:
        ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        conn.load_state_dict(ck["conn"])
        if ck["lora"] is not None:
            load_lora_state(model, ck["lora"])
        state = ck["state"]
        random.setstate(ck["rng"]["py"]); np.random.set_state(ck["rng"]["np"]); torch.set_rng_state(ck["rng"]["torch"])
        if torch.cuda.is_available() and ck["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        for f in (logp, devp):
            if f.exists():
                keep = [l for l in open(f) if json.loads(l)["update"] <= state["update"]]
                open(f, "w").writelines(keep)
        print(f"resumed at update {state['update']}")
    else:
        for f in (logp, devp):
            f.unlink(missing_ok=True)
        cal = calibrate(model, conn, data, store, dev)
        jdump(cal, run / "calibration.json"); print("calibration:", cal)
    if (run / "calibration.json").exists():
        assert abs(float(conn.scale) - jload(run / "calibration.json")["scale"]) < 1e-5

    # interface checks (once)
    if not (run / "interface_checks.json").exists():
        jdump(interface_checks(model, conn, data, store, codec, dev, amp), run / "interface_checks.json")

    logf, devf = open(logp, "a"), open(devp, "a")
    if state["update"] == 0 and 0 in DEV_EVAL_AT:
        t = time.time(); r = dev_eval(model, conn, data, store, dev, amp, 0); state["dev_time"] += time.time() - t
        devf.write(json.dumps(r) + "\n"); devf.flush()
    opt, opt_stage, scaler = None, None, torch.amp.GradScaler(dev.type, enabled=amp)
    model.train(); conn.train()
    u = state["update"]
    paused = False
    while u < TOTAL_UPDATES:
        k = stage_of(u)
        if opt_stage != k:
            if k >= 1 and not lora_modules(model):
                attach_lora(model)                                     # B = 0: function unchanged at stage start
            opt = make_optimizer(conn, model, k); opt_stage = k
            if ck is not None and ck["opt_stage"] == k and u == ck["state"]["update"]:
                opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
            ps = trainable_params(conn, model)
            frozen_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
            assert not frozen_trainable, f"backbone parameters are trainable: {frozen_trainable[:3]}"
            state["trainable_by_stage"][STAGES[k]["name"]] = dict(
                connector=sum(p.numel() for p in conn.parameters() if p.requires_grad),
                lora=sum(p.numel() for _, m in lora_modules(model) for p in (m.lora_A, m.lora_B)),
                total=sum(p.numel() for p in ps))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        s = u - stage_start(k); n_k = STAGES[k]["updates"]
        mult = lr_mult(s, n_k)
        for g in opt.param_groups:
            g["lr"] = g["peak"] * mult
        sync(); t0 = time.time()
        exs, mix = update_examples(data, plan, u)
        loss_tot = 0.0
        for mb in range(GRAD_ACCUM):
            chunk = exs[mb * MICRO_BS:(mb + 1) * MICRO_BS]
            feats = store.get([e["image"] for e in chunk], dev)
            batch = build_batch(codec, chunk, feats, store.n_tokens, device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                per_ex, _ = per_example_nll(model, conn, batch)
            loss = per_ex.sum() / EFF_BS                     # mean within example, then over the effective batch
            scaler.scale(loss).backward()
            loss_tot += float(loss)
        scaler.unscale_(opt)
        ps = trainable_params(conn, model)
        gn = float(torch.nn.utils.clip_grad_norm_(ps, CLIP))
        before = scaler.get_scale()
        scaler.step(opt); scaler.update()
        skipped = amp and scaler.get_scale() < before
        opt.zero_grad(set_to_none=True)
        sync(); dt = time.time() - t0
        u += 1; state["update"] = u
        name = STAGES[k]["name"]
        state["stage_time"][name] = state["stage_time"].get(name, 0.0) + dt
        if torch.cuda.is_available():
            state["stage_peak_mem"][name] = max(state["stage_peak_mem"].get(name, 0), torch.cuda.max_memory_allocated())
        if skipped:
            state["skipped"].append(u)
        logf.write(json.dumps(dict(update=u, stage=name, loss=loss_tot, lr_conn=opt.param_groups[0]["lr"],
                                   lr_lora=opt.param_groups[2]["lr"] if len(opt.param_groups) > 2 else None,
                                   grad_norm=gn, loss_scale=before, skipped=bool(skipped), mix=mix, step_s=dt)) + "\n")
        if u % 25 == 0 or u == 1:
            logf.flush()
            print(f"[{name}] update {u}/{TOTAL_UPDATES} loss={loss_tot:.4f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"gn={gn:.2f}{' SKIPPED' if skipped else ''}", flush=True)
        if u in DEV_EVAL_AT:
            t = time.time(); r = dev_eval(model, conn, data, store, dev, amp, u); state["dev_time"] += time.time() - t
            devf.write(json.dumps(r) + "\n"); devf.flush()
            print("dev", u, {kk: (round(v["delta_nll_swap"], 4) if isinstance(v, dict) else v) for kk, v in r.items()
                             if kk in ("caption_loss", "grounding_dev", "vqa_dev")}, flush=True)
        if u in STAGE_ENDS:
            save_weights(run / "weights", f"stage{STAGE_ENDS.index(u) + 1}", conn, model)
        over = (time.time() - T0) / 3600 > a.time_budget_h
        if u % CKPT_EVERY == 0 or u in STAGE_ENDS or over or u == TOTAL_UPDATES:
            logf.flush()
            save_ckpt(run / "ckpt" / f"update_{u:04d}.pt", conn, model, opt, scaler, state, k)
        if over and u < TOTAL_UPDATES:
            paused = True
            print(f"time budget reached at update {u}; rerun with --resume", flush=True)
            break
    logf.close(); devf.close()
    if not paused:
        save_weights(run / "weights", "final", conn, model)
        jdump(dict(run=run.name, tokens=a.tokens, updates=TOTAL_UPDATES, example_presentations=TOTAL_UPDATES * EFF_BS,
                   stages=STAGES, stage_ends=STAGE_ENDS, micro_bs=MICRO_BS, grad_accum=GRAD_ACCUM, eff_bs=EFF_BS,
                   weight_decay=WEIGHT_DECAY, clip=CLIP, warmup_frac=WARMUP_FRAC, min_lr_frac=MIN_LR_FRAC,
                   lora=LORA_CFG, caps=dict(question=Q_CAP, caption=CAPTION_CAP, answer=ANSWER_CAP),
                   train_time_s_by_stage=state["stage_time"], train_time_s=sum(state["stage_time"].values()),
                   dev_eval_time_s=state["dev_time"],
                   peak_gpu_mem_GB_by_stage={k_: v / 2**30 for k_, v in state["stage_peak_mem"].items()},
                   trainable_by_stage=state["trainable_by_stage"], skipped_updates=state["skipped"],
                   calibration=jload(run / "calibration.json"), smoke=SMOKE, seed=SEED,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   weights_bytes={p.name: p.stat().st_size for p in (run / "weights").iterdir()}),
              run / "train_summary.json")
        print("training complete")


def interface_checks(model, conn, data, store, codec, dev, amp):
    """Shapes, frozen sets, connector gradient flow through the frozen LM, visual-slot placement, no-image rebuild."""
    chk = {}
    exs = data.cap_train[:2]
    feats = store.get([e["image"] for e in exs], dev)
    chk["feature_shape"] = list(feats.shape)
    z = conn(feats.float())
    chk["connector_output_shape"] = list(z.shape)
    chk["lm_hidden"] = model.config.hidden_size
    b = build_batch(codec, exs, feats, store.n_tokens, device=dev)
    n0 = len(codec.pre_ids)
    chk["visual_slots_per_example"] = int(b["vmask"][0].sum())
    chk["visual_slots_start_after_user_header"] = bool(b["vmask"][0, n0:n0 + store.n_tokens].all() and not b["vmask"][0, :n0].any())
    chk["user_header_text"] = codec.tok.decode(codec.pre_ids)
    chk["labels_only_on_target"] = bool((b["labels"][b["vmask"]] == -100).all())
    model.train(); conn.train(); conn.zero_grad()
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
        l, _ = per_example_nll(model, conn, b)
    l.mean().backward()
    gnorm = sum(float(p.grad.norm()) for p in conn.parameters() if p.grad is not None)
    chk["connector_grad_norm"] = gnorm
    chk["connector_receives_gradient"] = gnorm > 0
    chk["backbone_trainable_params"] = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n)
    chk["backbone_has_grads"] = any(p.grad is not None for n, p in model.named_parameters() if "lora_" not in n)
    conn.zero_grad()
    bn = build_batch(codec, exs, None, store.n_tokens, with_image=False, device=dev)
    chk["no_image_length_reduction"] = int(b["att"][0].sum() - bn["att"][0].sum())
    chk["no_image_positions_contiguous"] = bool((bn["pos"][0, :int(bn["att"][0].sum())] ==
                                                 torch.arange(int(bn["att"][0].sum()), device=dev)).all())
    chk["split_isolation"] = "asserted in prepare_data.py (eval sets mutually image-disjoint, none in training/replay)"
    assert chk["connector_receives_gradient"] and chk["backbone_trainable_params"] == 0 and not chk["backbone_has_grads"]
    assert chk["visual_slots_per_example"] == store.n_tokens and chk["no_image_length_reduction"] == store.n_tokens
    print("interface checks:", chk)
    return chk


if __name__ == "__main__":
    main()

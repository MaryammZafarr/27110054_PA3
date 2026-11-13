"""Task 1 pipeline driver for Kaggle (also runs locally). MODE is filled in by make_kernels.py.

Safe to re-run: finished phases are skipped, interrupted training resumes from its latest recovery
checkpoint, and output from a previous kernel version (attached as a kernel source) is restored first.
Order: selftest, data, all 5 trainings, then evaluation one condition at a time (each resumable).
Disk (/kaggle/working is capped at ~20 GB) is logged after every phase, with the peak seen during it.
GPU hours are tracked across sessions; if the plan would exceed TOTAL_GPU_H the evaluation subset is trimmed
(never training) after the first condition has been timed, and the decision is written to TRIMMED.md.
"""
import glob, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

MODE = "__MODE__"                     # "smoke" (few updates, small eval) or "full"
BUDGET_H = 11.0                       # stop cleanly before Kaggle's ~12 h limit; rerun to continue
# GPU quota: ~19 h left at 09:40 on 2026-10-27, version 1 uses ~11 h in total (about 2.6 h after that time).
# Cumulative ceiling across all sessions = 11 h (v1) + 14 h = 25 h; trimming aims at 24.5 h, i.e. ~16 h
# of work from 09:40 including v1's remainder. HARD_STOP_H is the absolute stop that protects the quota.
TOTAL_GPU_H = 25.0                    # trim the evaluation subset if the timed projection exceeds this
SAFETY_H = 0.5
PRIOR_GPU_H = __PRIOR__               # GPU hours used by earlier versions, baked in at push time (floor if a restore fails)
HARD_STOP_H = 25.5                    # never start a condition that would end beyond this
CKPT_EVERY = 100                      # rolling recovery checkpoint every N updates (full mode)
METHODS = ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]
EVAL_ORDER = ["baseline", "full_ft", "l2sp", "lora4", "lora16", "oplora16", "full_ft_mid"]
SMOKE_UPDATES = 4
SMOKE = MODE == "smoke"
# train.py drops older rolling checkpoints first if a save would push /kaggle/working past this many GB.
# Smoke uses a deliberately small value so that fallback path gets exercised.
DISK_BUDGET_GB = 12 if SMOKE else 18
MID = 2 if SMOKE else 625             # full_ft mid-point update (T1_MID_UPDATE in smoke)

T0 = time.time()
SESSION = str(int(T0))
ON_KAGGLE = Path("/kaggle/working").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("work")
OUT = WORK / ("task1_out_smoke" if SMOKE else "task1_out")
CODE = WORK / "task1"
LOGS = WORK / "logs"
LOG = WORK / f"pipeline_{MODE}.log"
GB = 2**30


def say(msg):
    line = f"[{(time.time() - T0) / 60:7.1f} min] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def remaining_s():
    return BUDGET_H * 3600 - (time.time() - T0)


# ----------------------------------------------------------------------------- GPU-hour bookkeeping
def sessions_path():
    return OUT / "sessions.json"


def write_session():
    """Persist this session's elapsed seconds (survives in the kernel output, so later versions can sum them)."""
    if not OUT.exists():
        return
    try:
        d = json.loads(sessions_path().read_text()) if sessions_path().exists() else {}
        d[SESSION] = time.time() - T0
        tmp = sessions_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, sessions_path())
    except Exception:
        pass


def used_h():
    """Cumulative GPU hours over all sessions: the restored sessions.json total, but never less than
    PRIOR_GPU_H + this session, so a failed restore cannot reset the count to zero."""
    write_session()
    here = (time.time() - T0) / 3600
    try:
        total = sum(json.loads(sessions_path().read_text()).values()) / 3600
    except Exception:
        total = here
    return max(total, PRIOR_GPU_H + here)


# ----------------------------------------------------------------------------- disk accounting
def tree_size(p):
    tot = 0
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    tot += tree_size(e.path) if e.is_dir(follow_symlinks=False) else e.stat(follow_symlinks=False).st_size
                except OSError:
                    pass
    except (OSError, NotADirectoryError):
        pass
    return tot


class DiskPeak(threading.Thread):
    """Samples the size of the working tree in the background (and the session clock) and remembers the max."""
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self._stop_evt, self._n = 0, threading.Event(), 0

    def run(self):
        while not self._stop_evt.wait(10):
            self.peak = max(self.peak, tree_size(WORK))
            self._n += 1
            if self._n % 6 == 0:
                write_session()

    def take(self):
        p, self.peak = max(self.peak, tree_size(WORK)), 0
        return p


def disk_report(tag, peak=None):
    du = shutil.disk_usage(WORK)
    used = tree_size(WORK)
    say(f"DISK after {tag}: working tree {used / GB:.2f} GB"
        + (f", peak during phase {peak / GB:.2f} GB" if peak is not None else "")
        + f" | df: {du.used / GB:.1f}/{du.total / GB:.1f} GB used, {du.free / GB:.1f} GB free | GPU-h used {used_h():.2f}")
    rows = []
    for base in (WORK, OUT, OUT / "runs"):
        if base.exists():
            rows += [(c, tree_size(c)) for c in base.iterdir() if c.is_dir()]
    for r in (OUT / "runs").glob("*") if (OUT / "runs").exists() else []:
        rows += [(c, tree_size(c)) for c in r.iterdir() if c.is_dir()]
    seen, out = set(), []
    for c, s in sorted(rows, key=lambda x: -x[1]):
        if s >= 50 * 2**20 and c not in seen:
            seen.add(c); out.append(f"{s / GB:6.2f} GB  {c.relative_to(WORK)}")
    for line in out[:8]:
        say("    du " + line)


PEAK = None


# ----------------------------------------------------------------------------- code / restore
def find_code():
    if (CODE / "train.py").exists():
        return
    hits = [Path(p).parent for p in glob.glob("/kaggle/input/**/task1/train.py", recursive=True)]
    if not hits:
        hits = [Path(p).parent for p in glob.glob("/kaggle/input/**/train.py", recursive=True)]
    if not hits:
        sys.exit("task1 code not found under /kaggle/input (is the dataset attached?)")
    shutil.copytree(hits[0], CODE)
    say(f"copied code from {hits[0]} to {CODE}")


def restore_previous():
    """Copy output of an earlier version of this kernel (attached as a kernel source) into /kaggle/working."""
    if OUT.exists() or not ON_KAGGLE:
        return
    for p in glob.glob(f"/kaggle/input/**/{OUT.name}", recursive=True):
        if (Path(p) / "runs").exists() or (Path(p) / "data").exists():
            say(f"restoring previous output from {p} ({tree_size(p) / GB:.2f} GB)")
            shutil.copytree(p, OUT)
            prev = Path(p).parent                       # keep the earlier session's logs too
            keep = OUT / "session_logs" / f"before_{SESSION}"
            keep.mkdir(parents=True, exist_ok=True)
            for name in ("logs", f"pipeline_{MODE}.log"):
                if (prev / name).exists():
                    (shutil.copytree if (prev / name).is_dir() else shutil.copy)(prev / name, keep / name)
            return
    say("no previous output found; starting from scratch")


# ----------------------------------------------------------------------------- running scripts
def run(args, name, env=None):
    """Run a script in the code dir, teeing its output to logs/<name>.log. Returns True on success;
    on reaching the time budget, kills it and exits cleanly (training is checkpointed)."""
    cmd = [sys.executable, "-u", *args]
    say(f"START {name}: {' '.join(cmd)}")
    LOGS.mkdir(exist_ok=True)
    e = {**os.environ, "T1_DISK_BUDGET_GB": str(DISK_BUDGET_GB),
         "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **(env or {})}
    timed_out = []
    with open(LOGS / f"{name}.log", "a") as lf:          # append: a resumed phase keeps its earlier output
        proc = subprocess.Popen(cmd, cwd=CODE, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        timer = threading.Timer(max(60, remaining_s()), lambda: (timed_out.append(1), proc.kill()))
        timer.start()
        for line in proc.stdout:
            sys.stdout.write(line); lf.write(line); lf.flush()
        proc.wait(); timer.cancel()
    if timed_out:
        say(f"TIME BUDGET reached during {name}; progress is checkpointed. Push again to resume.")
        disk_report(f"{name} (timeout)", PEAK.take())
        sys.exit(0)
    say(f"{'DONE' if proc.returncode == 0 else 'FAILED (rc=%d)' % proc.returncode} {name}")
    return proc.returncode == 0


def marker(name):
    return OUT / f".done_{name}"


def is_done(name):
    """Hidden .done_* files may be dropped by a dataset upload (fallback restore), so accept a visible copy too."""
    return marker(name).exists() or (OUT / f"done_{name}").exists()


def phase(name, args, env=None, must=True):
    if is_done(name):
        say(f"skip {name} (already done)")
        return
    if remaining_s() < 600:
        say(f"not enough time left to start {name}; push again to continue")
        sys.exit(0)
    PEAK.take()
    ok = run(args, name, env)
    disk_report(name, PEAK.take())
    if ok:
        marker(name).touch()
    elif must:
        sys.exit(f"{name} failed; see logs/{name}.log")


# ----------------------------------------------------------------------------- evaluation with a GPU-hour plan
def plan_file():
    return OUT / "eval_plan.json"


def load_plan():
    return json.loads(plan_file().read_text()) if plan_file().exists() else {}


def decide_plan(ev, env):
    """After the first (baseline) condition has been timed: project the total GPU hours; if it exceeds
    TOTAL_GPU_H, trim the evaluation subset (every benchmark gets the same first-N limit) and redo baseline."""
    info = json.loads((OUT / "eval" / "baseline" / "eval_info.json").read_text())
    t1 = info.get("build_s", 0) + info.get("lmeval_s", 0) + info.get("qual_s", 0)
    used = used_h()
    t1_h = t1 / 3600
    projected = used + 6 * t1_h * 1.1          # 10% allowance for the merge checks LoRA conditions add
    plan = dict(decided=True, baseline_full_s=t1, used_h_at_decision=used, projected_total_h=projected,
                ceiling_h=TOTAL_GPU_H, limit=None, cond_s=t1 * 1.1)
    say(f"EVAL PLAN: baseline took {t1 / 60:.1f} min; GPU-h used {used:.2f}; projected total {projected:.2f} h "
        f"(ceiling {TOTAL_GPU_H} h)")
    if projected > TOTAL_GPU_H:
        f = (TOTAL_GPU_H - SAFETY_H - used) / (7 * t1_h * 1.1)     # baseline is redone at the trimmed size
        limit = max(100, int(1319 * max(f, 0) * 0.85))
        plan.update(limit=limit, cond_s=t1 * max(f, 0.05) * 1.1, trim_fraction=f)
        plan_file().write_text(json.dumps(plan, indent=2))
        shutil.move(str(OUT / "eval" / "baseline"), str(OUT / "eval_baseline_fullsplit"))
        marker("eval_baseline").unlink(missing_ok=True); (OUT / "done_eval_baseline").unlink(missing_ok=True)
        (OUT / "TRIMMED.md").write_text(
            "# Evaluation was trimmed (training was not)\n\n"
            f"- Decision taken after timing the baseline condition: {t1 / 60:.1f} min on the full benchmark splits.\n"
            f"- GPU hours already used at that point: {used:.2f} h. Projected total with full evaluation: "
            f"{projected:.2f} h, above the {TOTAL_GPU_H} h ceiling.\n"
            f"- Action: all 7 conditions are evaluated with `--limit {limit}` (the first {limit} examples of each of "
            "GSM8K, HellaSwag, BoolQ and ARC-Easy), identical for every condition so they stay comparable.\n"
            "- The 24 qualitative prompts, the merge checks and the diagnostics are NOT trimmed.\n"
            "- The baseline's full-split result (taken before the decision) is kept in `eval_baseline_fullsplit/`; "
            "the baseline was re-evaluated at the trimmed size for the comparison tables.\n"
            "- verify.py's 'full splits' / 'n = benchmark size' checks are expected to fail because of this.\n")
        say(f"EVAL TRIMMED: --limit {limit} for all conditions (see TRIMMED.md)")
        phase("eval_baseline", ev + ["--limit", str(limit), "--conditions", "baseline"], env)
    else:
        plan_file().write_text(json.dumps(plan, indent=2))
        say("evaluation fits: no trimming")


def eval_all(ev, env):
    for c in EVAL_ORDER:
        pl = load_plan()
        args = ev + (["--limit", str(pl["limit"])] if pl.get("limit") else []) + ["--conditions", c]
        if not is_done(f"eval_{c}") and pl.get("cond_s") and remaining_s() < 1.15 * pl["cond_s"]:
            say(f"too little time left for eval_{c} (~{pl['cond_s'] / 60:.0f} min); push again to continue")
            disk_report("pause before " + c)
            sys.exit(0)
        if not is_done(f"eval_{c}") and used_h() + (pl.get("cond_s") or 0) / 3600 > HARD_STOP_H:
            note = (f"QUOTA STOP before eval_{c}: GPU-h used {used_h():.2f} + ~{(pl.get('cond_s') or 0) / 3600:.2f} h "
                    f"would exceed the hard ceiling {HARD_STOP_H} h. Remaining conditions were not evaluated.")
            (OUT / "QUOTA_STOP.txt").write_text(note + "\n")
            say(note)
            disk_report("quota stop")
            sys.exit(0)
        phase(f"eval_{c}", args, env)
        if c == "baseline" and not load_plan().get("decided"):
            decide_plan(ev, env)


def main():
    global PEAK
    WORK.mkdir(exist_ok=True)
    say(f"mode={MODE} out={OUT} budget={BUDGET_H}h disk_budget={DISK_BUDGET_GB}GB session={SESSION}")
    PEAK = DiskPeak(); PEAK.start()
    find_code()
    restore_previous()
    OUT.mkdir(parents=True, exist_ok=True)
    disk_report("start")
    out = str(OUT)
    env = {"T1_MID_UPDATE": str(MID), "T1_FINAL_UPDATE": str(SMOKE_UPDATES)} if SMOKE else {}

    if ON_KAGGLE and not is_done("deps"):
        run(["-m", "pip", "install", "-q", "lm_eval>=0.4.9", "datasets", "pyyaml", "safetensors", "sentencepiece"], "pip_install")
        marker("deps").touch()
    say("env: " + subprocess.run([sys.executable, "-c",
        "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available(),"
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"],
        capture_output=True, text=True).stdout.strip())

    phase("selftest", ["selftest.py"], must=False)
    phase("prepare", ["prepare_data.py", "--out", out, "--qual_yaml", "first_plot_questions.yaml"])

    for m in METHODS:
        # smoke: lora4 is trained twice, the second pass resumes from the first one's checkpoint
        passes = 2 if (SMOKE and m == "lora4") else 1
        for i in range(passes):
            if not SMOKE and (OUT / "runs" / m / "train_summary.json").exists():
                say(f"skip train_{m} (summary exists)")
                continue
            a = ["train.py", "--method", m, "--out", out]
            if m == "l2sp":
                a += ["--l2sp_ref_device", "cpu"]
            if SMOKE:
                a += ["--smoke_updates", str(SMOKE_UPDATES), "--ckpt_every", "2"]
                if i == 0:
                    a.append("--fresh")
                name = f"train_{m}_{i}"
                marker(name).unlink(missing_ok=True)       # always re-run in smoke mode
                if not (m == "lora4" and i == 0):          # keep lora4's checkpoint for the resume pass
                    a.append("--prune_recovery_at_end")
            else:
                a += ["--ckpt_every", str(CKPT_EVERY), "--prune_recovery_at_end"]
                name = f"train_{m}"
            phase(name, a, env)

    ev = ["evaluate.py", "--out", out]
    if SMOKE:
        ev += ["--limit", "8", "--gen_batch_size", "8"]
        for c in EVAL_ORDER:
            phase(f"eval_{c}", ev + ["--conditions", c], env)
    else:
        eval_all(ev, env)
    phase("diagnostics", ["diagnostics.py", "--out", out], env)
    phase("analyze", ["analyze.py", "--out", out], env, must=False)
    if not SMOKE:
        phase("verify", ["verify.py", "--out", out], must=False)

    say("ALL PHASES COMPLETE")
    disk_report("end")
    for d in sorted(OUT.rglob("*")):
        if d.is_file() and d.stat().st_size > 50 * 2**20:
            say(f"large file: {d.relative_to(WORK)} {d.stat().st_size / GB:.2f} GB")


if __name__ == "__main__":
    main()

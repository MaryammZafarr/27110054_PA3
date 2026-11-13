# ===== PA3 Task 2 — Kaggle runner (one cell; safe to re-commit, resumes where it stopped) =====
import os, shutil, subprocess, time, json
from pathlib import Path

START = time.time()
BUDGET_H = 11.0                                   # Kaggle kills at 12 h; leave margin for saving output
CODE = next(p.parent for p in Path("/kaggle/input").rglob("selftest.py")) # wherever Kaggle mounted the code dataset
PREV = next((p for p in Path("/kaggle/input").rglob("task2_out") if p.is_dir()), Path("/nonexistent")) # previous output, if attached
OUT = Path("/kaggle/working/task2_out")
WORK = Path("/kaggle/working/task2"); WORK.mkdir(parents=True, exist_ok=True)
RAW = "/tmp/t2raw"                                # raw annotation zips stay OUT of the saved output
RUN_49 = False                                    # optional 49-token experiment (doubles training cost)

def hours(): return (time.time() - START) / 3600
def run(cmd):
    print(f"\n=== [{hours():.2f} h] {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

run("pip install -q -U 'transformers>=4.51' safetensors pillow matplotlib")
for f in CODE.rglob("*.py"):
    shutil.copy(f, WORK / f.name)
os.chdir(WORK)
print("code from", CODE, "| previous output:", PREV, "| files:", sorted(p.name for p in WORK.glob("*.py")))
prev_out = PREV if PREV.exists() else None
if prev_out and not OUT.exists():
    shutil.copytree(prev_out, OUT); print("restored", prev_out)

if not (OUT / "splits" / "selection_stats.json").exists():
    run("python selftest.py")
    run(f"python prepare_data.py --out {OUT} --raw_dir {RAW}")        # add --image_dir <attached COCO-2014> if you have one
if not (OUT / "features" / "cache_info.json").exists():
    run(f"python cache_features.py --out {OUT} --delete_train_images")

for name, tok in [("main", 196)] + ([("tok49", 49)] if RUN_49 else []):
    rd = OUT / "runs" / name
    if not (rd / "train_summary.json").exists():
        left = BUDGET_H - hours()
        if left < 0.3:
            print("budget used; commit again to continue"); break
        run(f"python train.py --out {OUT} --run {name} --tokens {tok} --resume --time_budget_h {left:.2f}")
    if (rd / "train_summary.json").exists() and not (OUT / "eval" / name / "predictions.jsonl").exists():
        run(f"python evaluate.py --out {OUT} --run {name}")
    if (OUT / "eval" / name / "predictions.jsonl").exists():
        run(f"python analyze.py --out {OUT} --run {name}")
        run(f"python verify.py --out {OUT} --run {name}")
print(f"done at {hours():.2f} h")

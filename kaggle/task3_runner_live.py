# ===== PA3 Task 3 — LIVE runner (Step 4, 50 MATH questions). Kernel settings: Accelerator GPU T4, Internet ON =====
import os, shutil, subprocess, time
from pathlib import Path

START = time.time()
BUDGET_H = 11.0
DRIVE = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"
CODE = next(p.parent for p in Path("/kaggle/input").rglob("selftest.py") if (p.parent / "live.py").exists())
PREV = next((p for p in Path("/kaggle/input").rglob("task3_out") if (p / "live").is_dir()), None)   # previous live output, if attached
OUT = Path("/kaggle/working/task3_out")
WORK = Path("/kaggle/working/task3"); WORK.mkdir(parents=True, exist_ok=True)
DATA = Path("/tmp/t3data")

def run(cmd):
    print(f"\n=== [{(time.time() - START) / 3600:.2f} h] {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

run("pip install -q 'math-verify[antlr4_13_2]' gdown pyarrow")
for f in CODE.glob("*.py"):
    shutil.copy(f, WORK / f.name)
os.chdir(WORK)
if PREV is not None and not OUT.exists():
    shutil.copytree(PREV, OUT); print("restored previous output from", PREV)
attached = next((p.parent.parent for p in Path("/kaggle/input").rglob("generations.parquet") if p.parent.name == "qwen25-math"), None)
if attached is None:
    run(f"gdown --folder '{DRIVE}' -O {DATA}")
root = attached or next(p.parent.parent for p in DATA.rglob("generations.parquet") if p.parent.name == "qwen25-math")
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("python selftest.py")
left = BUDGET_H - (time.time() - START) / 3600
run(f"python live.py --data {root} --out {OUT} --time_budget_h {left:.2f}")
run(f"python analyze.py --out {OUT}")
run(f"python verify.py --out {OUT} --part live")
print(f"done at {(time.time() - START) / 3600:.2f} h")

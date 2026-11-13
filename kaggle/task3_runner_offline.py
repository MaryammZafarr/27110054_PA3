# ===== PA3 Task 3 — OFFLINE runner (Steps 1, 2, 3, 5). Kernel settings: Accelerator NONE (no GPU quota), Internet ON =====
import os, shutil, subprocess, time
from pathlib import Path

START = time.time()
DRIVE = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"
CODE = next(p.parent for p in Path("/kaggle/input").rglob("selftest.py") if (p.parent / "offline.py").exists())
OUT = Path("/kaggle/working/task3_out")
WORK = Path("/kaggle/working/task3"); WORK.mkdir(parents=True, exist_ok=True)
DATA = Path("/tmp/t3data")                       # rollouts stay OUT of the saved output

def run(cmd):
    print(f"\n=== [{(time.time() - START) / 3600:.2f} h] {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

run("pip install -q 'math-verify[antlr4_13_2]' gdown datasets pyarrow")
for f in CODE.glob("*.py"):
    shutil.copy(f, WORK / f.name)
os.chdir(WORK)
attached = next((p.parent.parent for p in Path("/kaggle/input").rglob("generations.parquet") if p.parent.name == "qwen25-math"), None)
if attached is None:
    run(f"gdown --folder '{DRIVE}' -O {DATA}")
root = attached or next(p.parent.parent for p in DATA.rglob("generations.parquet") if p.parent.name == "qwen25-math")
print("data root:", root, flush=True)
run("python selftest.py")
if not (OUT / "prepared" / "prepare_info.json").exists():
    run(f"python prepare.py --data {root} --out {OUT}")
run(f"python offline.py --out {OUT}")
run(f"python analyze.py --out {OUT}")
run(f"python verify.py --out {OUT} --part offline")
print(f"done at {(time.time() - START) / 3600:.2f} h")

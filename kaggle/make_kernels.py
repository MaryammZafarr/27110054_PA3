"""Generate kernel_smoke/ and kernel_full/ (script + kernel-metadata.json) from pipeline_template.py.

Usage: python make_kernels.py [--prev_output maryamzafarkhan/task1-atml-full]
  --prev_output attaches an earlier version of the full kernel as a source, so a re-push restores its output.
  Leave it off for the very first push (the kernel does not exist yet).
"""
import argparse, json
from pathlib import Path

USER, DATASET = "maryamzafarkhan", "maryamzafarkhan/task1-atml"
here = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--prev_output", default=None)
ap.add_argument("--prior_gpu_h", type=float, default=0.0, help="GPU hours used by earlier versions of kernel_full")
a = ap.parse_args()

tpl = (here / "pipeline_template.py").read_text()
for mode in ("smoke", "full"):
    d = here / f"kernel_{mode}"
    d.mkdir(exist_ok=True)
    (d / "pipeline.py").write_text(tpl.replace("__MODE__", mode).replace("__PRIOR__", str(a.prior_gpu_h if mode == "full" else 0.0)))
    meta = {
        "id": f"{USER}/task1-atml-{mode}",
        "title": f"Task1 ATML {mode}",
        "code_file": "pipeline.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,        # needed: Hugging Face models/datasets and lm-eval task data
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [a.prev_output] if (mode == "full" and a.prev_output) else [],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", d)

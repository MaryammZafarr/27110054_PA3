"""
PA3 Task 1 -- verify LoRA-16 and OPLoRA-16 are distinct checkpoints.

Qualitative probing (annotation_sheet_42.csv) produced byte-identical text for
LoRA-16 and OPLoRA-16 on several prompts. This script rules out the trivial
explanation (same checkpoint written twice) before that similarity is reported
as a result.

Writes adapter_comparison.json next to the run outputs.
Pure CPU, no torch required.

Usage:  python verify_adapters.py
"""
import hashlib, json, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "full_v4" / "task1_out" / "runs"
EXPORT = "exports/update_1250"
A_NAME, B_NAME = "lora16", "oplora16"
OUT = ROOT / "outputs" / "full_v4" / "task1_out" / "adapter_comparison.json"

NP_DTYPE = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
            "I64": np.int64, "I32": np.int32, "I16": np.int16,
            "U8": np.uint8, "BOOL": np.bool_}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_safetensors(path):
    """Minimal safetensors reader -> {name: float32 ndarray}. Handles BF16."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        blob = fh.read()
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = blob[start:end]
        if meta["dtype"] == "BF16":
            bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = bits.view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=NP_DTYPE[meta["dtype"]])
        tensors[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return tensors


def main():
    a_dir, b_dir = RUNS / A_NAME / EXPORT, RUNS / B_NAME / EXPORT
    for d in (a_dir, b_dir):
        if not d.exists():
            sys.exit(f"missing export dir: {d}")

    a_w, b_w = a_dir / "adapter.safetensors", b_dir / "adapter.safetensors"
    result = {
        "checkpoints": {A_NAME: str(a_w), B_NAME: str(b_w)},
        "md5": {A_NAME: md5(a_w), B_NAME: md5(b_w)},
        "config": {n: json.loads((d / "adapter_config.json").read_text())
                   for n, d in ((A_NAME, a_dir), (B_NAME, b_dir))},
        "export_info": {n: json.loads((d / "export_info.json").read_text())
                        for n, d in ((A_NAME, a_dir), (B_NAME, b_dir))},
    }
    result["md5_identical"] = result["md5"][A_NAME] == result["md5"][B_NAME]

    A, B = read_safetensors(a_w), read_safetensors(b_w)
    shared = sorted(set(A) & set(B))

    per_tensor, n_identical = [], 0
    for k in shared:
        ta, tb = A[k], B[k]
        if ta.shape != tb.shape:
            per_tensor.append({"tensor": k, "shape_mismatch": True})
            continue
        abs_d = float(np.abs(ta - tb).max())
        rel_d = abs_d / (float(np.abs(ta).max()) + 1e-12)
        n_identical += abs_d == 0.0
        per_tensor.append({"tensor": k, "max_abs_diff": abs_d, "rel_diff": rel_d})

    rels = [t["rel_diff"] for t in per_tensor if "rel_diff" in t]
    norm_a = float(np.sqrt(sum((v.astype(np.float64) ** 2).sum() for v in A.values())))
    norm_b = float(np.sqrt(sum((v.astype(np.float64) ** 2).sum() for v in B.values())))
    norm_d = float(np.sqrt(sum(((A[k].astype(np.float64) - B[k].astype(np.float64)) ** 2).sum()
                               for k in shared)))

    result["weights"] = {
        "n_tensors": len(shared),
        "same_tensor_names": set(A) == set(B),
        "n_identical_tensors": int(n_identical),
        "max_abs_diff": max(rels and [t["max_abs_diff"] for t in per_tensor if "max_abs_diff" in t] or [0.0]),
        "median_rel_diff": float(np.median(rels)) if rels else None,
        "mean_rel_diff": float(np.mean(rels)) if rels else None,
        "frobenius_norm": {A_NAME: norm_a, B_NAME: norm_b, "difference": norm_d,
                           "relative": norm_d / norm_a},
    }
    result["per_tensor"] = sorted(
        [t for t in per_tensor if "rel_diff" in t],
        key=lambda t: -t["max_abs_diff"])

    identical = result["md5_identical"] or n_identical == len(shared)
    result["verdict"] = (
        "IDENTICAL - same checkpoint, OPLoRA is not a distinct condition"
        if identical else
        "DISTINCT - checkpoints differ; identical generations are a decoding "
        "artifact, not a duplicated run")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    w = result["weights"]
    print(f"md5 {A_NAME:10s} {result['md5'][A_NAME]}")
    print(f"md5 {B_NAME:10s} {result['md5'][B_NAME]}")
    print(f"config kind      {result['config'][A_NAME]['kind']} vs {result['config'][B_NAME]['kind']}")
    print(f"identical tensors {w['n_identical_tensors']}/{w['n_tensors']}")
    print(f"median rel diff   {w['median_rel_diff']:.2%}")
    print(f"||A-B||/||A||     {w['frobenius_norm']['relative']:.2%}")
    print(f"\n{result['verdict']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

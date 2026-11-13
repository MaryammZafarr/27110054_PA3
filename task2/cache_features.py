"""Step 1: cache frozen SigLIP patch features (196 x 768, final LayerNorm applied exactly once) in FP16.

python cache_features.py --out OUT [--batch 64] [--delete_train_images]
"""
import argparse
from pathlib import Path

from PIL import Image

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--delete_train_images", action="store_true", help="free disk after caching (eval images kept)")
    a = ap.parse_args()
    from transformers import SiglipImageProcessor, SiglipVisionModel
    out = Path(a.out); fd = out / "features"; fd.mkdir(parents=True, exist_ok=True)
    keys = jload(out / "splits" / "images.json")["all"]
    dev = get_device(); set_seed()
    proc = SiglipImageProcessor.from_pretrained(VISION_ID)
    enc = SiglipVisionModel.from_pretrained(VISION_ID, torch_dtype=torch.float32).to(dev).eval()
    post_ln = [m for n, m in enc.named_modules() if n.endswith("post_layernorm")][0]
    D = enc.config.hidden_size
    n_patch = (enc.config.image_size // enc.config.patch_size) ** 2
    assert n_patch == GRID * GRID, f"expected 196 patches, got {n_patch}"

    path = fd / "siglip_patch_fp16.npy"
    done_file = fd / "progress.json"
    start = jload(done_file)["next"] if done_file.exists() and path.exists() else 0
    arr = np.lib.format.open_memmap(path, mode="r+" if start else "w+", dtype=np.float16, shape=(len(keys), n_patch, D))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sync(); t0 = time.time()
    ln_check = None
    with torch.no_grad():
        for b in range(start, len(keys), a.batch):
            ks = keys[b:b + a.batch]
            ims = []
            for k in ks:
                split, iid = k.split(":")
                ims.append(Image.open(out / "images" / split / coco_filename(split, iid)).convert("RGB"))
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
            o = enc(pixel_values=px, output_hidden_states=(ln_check is None))
            feats = o.last_hidden_state                                  # already post_layernorm (applied once)
            if ln_check is None:
                pre = o.hidden_states[-1]
                ln_check = dict(
                    last_equals_postLN_of_last_block=bool(torch.allclose(post_ln(pre), feats, atol=1e-4)),
                    last_differs_from_pre_LN=bool(not torch.allclose(pre, feats, atol=1e-3)),
                    shape=list(feats.shape), token_order="row-major 14x14 patch grid (no CLS token in SigLIP)")
                assert ln_check["last_equals_postLN_of_last_block"] and ln_check["last_differs_from_pre_LN"]
            arr[b:b + len(ks)] = feats.float().cpu().numpy().astype(np.float16)
            if (b // a.batch) % 20 == 0:
                arr.flush(); jdump({"next": b + len(ks)}, done_file)
                print(f"cached {b + len(ks)}/{len(keys)}", flush=True)
    arr.flush(); sync()
    secs = time.time() - t0
    jdump({k: i for i, k in enumerate(keys)}, fd / "index.json")
    jdump({"next": len(keys)}, done_file)
    # determinism check: re-encode the first image and compare
    with torch.no_grad():
        split, iid = keys[0].split(":")
        px = proc(images=[Image.open(out / "images" / split / coco_filename(split, iid)).convert("RGB")],
                  return_tensors="pt")["pixel_values"].to(dev)
        again = enc(pixel_values=px).last_hidden_state.float().cpu().numpy().astype(np.float16)
    info = dict(vision_model=VISION_ID, n_images=len(keys), shape=[len(keys), n_patch, D], dtype="float16",
                seconds=secs, images_per_s=len(keys) / max(secs, 1e-9),
                peak_gpu_mem_GB=(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
                file_bytes=path.stat().st_size, preprocessing=proc.to_dict(), layernorm_check=ln_check,
                deterministic_reencode=bool(np.array_equal(again[0], np.asarray(arr[0]))),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", partial_resume_from=start)
    jdump(info, fd / "cache_info.json")
    print(json.dumps({k: v for k, v in info.items() if k != "preprocessing"}, indent=1))
    if a.delete_train_images:
        for k in jload(out / "splits" / "images.json")["train"]:
            split, iid = k.split(":")
            (out / "images" / split / coco_filename(split, iid)).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

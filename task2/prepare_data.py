"""Step 0: download COCO-2014 / VQAv2 annotations, select all splits (seed 6304), generate the
annotation-derived grounding pools, fix the mismatched-image mapping, download the needed images,
and write the label-audit sheets.

python prepare_data.py --out OUT [--image_dir /kaggle/input/<coco2014>]  [--raw_dir DIR]
"""
import argparse, io, urllib.request, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import *

URLS = {
    "annotations_trainval2014.zip": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "v2_Questions_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "v2_Questions_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "v2_Annotations_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "v2_Annotations_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "v2_Complementary_Pairs_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "v2_Complementary_Pairs_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
}
ZIP_MEMBERS = {
    "annotations_trainval2014.zip": ["captions_train2014.json", "captions_val2014.json", "instances_train2014.json",
                                     "instances_val2014.json"],
    "v2_Questions_Train_mscoco.zip": ["v2_OpenEnded_mscoco_train2014_questions.json"],
    "v2_Questions_Val_mscoco.zip": ["v2_OpenEnded_mscoco_val2014_questions.json"],
    "v2_Annotations_Train_mscoco.zip": ["v2_mscoco_train2014_annotations.json"],
    "v2_Annotations_Val_mscoco.zip": ["v2_mscoco_val2014_annotations.json"],
    "v2_Complementary_Pairs_Train_mscoco.zip": ["v2_mscoco_train2014_complementary_pairs.json"],
    "v2_Complementary_Pairs_Val_mscoco.zip": ["v2_mscoco_val2014_complementary_pairs.json"],
}
NEEDED = ["captions_train2014.json", "captions_val2014.json", "instances_train2014.json", "instances_val2014.json",
          "v2_OpenEnded_mscoco_train2014_questions.json", "v2_OpenEnded_mscoco_val2014_questions.json",
          "v2_mscoco_train2014_annotations.json", "v2_mscoco_val2014_annotations.json",
          "v2_mscoco_train2014_complementary_pairs.json", "v2_mscoco_val2014_complementary_pairs.json"]

IRREGULAR = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
             "knife": "knives", "broccoli": "broccoli", "bus": "buses", "couch": "couches", "bench": "benches",
             "sandwich": "sandwiches", "toothbrush": "toothbrushes", "wine glass": "wine glasses", "skateboard":
             "skateboards", "glass": "glasses"}


def plural(name):
    if name in IRREGULAR:
        return IRREGULAR[name]
    head, _, last = name.rpartition(" ")
    if last in IRREGULAR:
        return (head + " " if head else "") + IRREGULAR[last]
    if re.search(r"(s|x|z|ch|sh)$", last):
        return name + "es"
    return name + "s"


def article(name):
    return ("an " if name[0] in "aeiou" else "a ") + name


# ----------------------------------------------------------------------------- raw files
def ensure_raw(raw_dir):
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)

    def find(n):
        hits = list(raw_dir.rglob(n))
        return hits[0] if hits else None

    for zname, url in URLS.items():
        members = ZIP_MEMBERS[zname]
        if all(find(n) for n in members):
            continue
        print("downloading", url, flush=True)
        data = urllib.request.urlopen(url, timeout=600).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if Path(info.filename).name in members:
                    (raw_dir / Path(info.filename).name).write_bytes(z.read(info))
    paths = {n: find(n) for n in NEEDED}
    missing = [n for n, p in paths.items() if p is None]
    assert not missing, f"missing raw files: {missing}"
    return paths


# ----------------------------------------------------------------------------- VQAv2 pairs
def vqa_candidates(qs, anns, pairs, split, codec):
    qmap = {q["question_id"]: q for q in qs["questions"]}
    amap = {a["question_id"]: a for a in anns["annotations"]}
    out, reasons = [], Counter()
    for q1, q2 in pairs:
        if q1 not in amap or q2 not in amap:
            reasons["missing"] += 1; continue
        items = []
        for qid in (q1, q2):
            a, q = amap[qid], qmap[qid]
            raw = [x["answer"] for x in a["answers"]]
            canon, agree, target, norm = consensus(raw)
            items.append(dict(qid=qid, image=image_key(split, a["image_id"]), image_id=a["image_id"],
                              question=q["question"].strip(), answer_type=a["answer_type"], agree=agree,
                              canonical=canon, target=target, answers=raw, answers_norm=norm))
        a, b = items
        if a["agree"] < 8 or b["agree"] < 8:
            reasons["agreement<8"] += 1; continue
        if a["canonical"] == b["canonical"] or not a["canonical"] or not b["canonical"]:
            reasons["same_or_empty_consensus"] += 1; continue
        if a["question"] != b["question"]:
            reasons["question_text_differs"] += 1; continue
        if a["answer_type"] != b["answer_type"]:
            reasons["answer_type_differs"] += 1; continue
        if a["image"] == b["image"]:
            reasons["same_image"] += 1; continue
        if codec.n_tokens(a["question"]) > Q_CAP:
            reasons["question>80tok"] += 1; continue
        if max(codec.n_tokens(a["target"]), codec.n_tokens(b["target"])) > ANSWER_CAP:
            reasons["answer>16tok"] += 1; continue
        out.append(dict(pair_id=f"vqa-{split}-{q1}-{q2}", suite="vqa", kind=a["answer_type"], question=a["question"],
                        items=[{k: it[k] for k in ("qid", "image", "image_id", "target", "canonical", "answers",
                                                   "answers_norm", "agree")} for it in items]))
    reasons["kept"] = len(out)
    return sorted(out, key=lambda p: p["pair_id"]), dict(reasons)


def take_quota(cands, quota, rng, used=None):
    """Seeded selection per answer type; with `used`, enforce global image uniqueness (evaluation)."""
    order = rng.permutation(len(cands))
    got = {k: [] for k in quota}
    for i in order:
        p = cands[i]
        if p["kind"] not in got or len(got[p["kind"]]) >= quota[p["kind"]]:
            continue
        imgs = [it["image"] for it in p["items"]]
        if used is not None and any(im in used for im in imgs):
            continue
        got[p["kind"]].append(p)
        if used is not None:
            used.update(imgs)
        if all(len(got[k]) >= quota[k] for k in quota):
            break
    for k in quota:
        assert len(got[k]) == quota[k], f"not enough {k} pairs: {len(got[k])}/{quota[k]}"
    return [p for k in quota for p in got[k]]


# ----------------------------------------------------------------------------- COCO grounding
def index_instances(inst):
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    per = defaultdict(lambda: defaultdict(list))
    for a in inst["annotations"]:
        per[a["image_id"]][a["category_id"]].append((a["area"], a["iscrowd"]))
    return cats, per


def presence_sets(per, imgs, cid):
    pos = [i for i in imgs if any(c == 0 and ar >= MIN_AREA for ar, c in per[i].get(cid, []))]
    neg = [i for i in imgs if cid not in per[i]]
    return pos, neg


def count_sets(per, imgs, cid):
    by = defaultdict(list)
    for i in imgs:
        inst = per[i].get(cid)
        if not inst or any(c for _, c in inst) or any(ar < MIN_AREA for ar, _ in inst):
            continue                                   # no crowd boxes, no tiny instances -> unambiguous count
        if 1 <= len(inst) <= 4:
            by[len(inst)].append(i)
    return by


def g_pair(kind, split, cid, cname, ia, ib, ta, tb, extra=None):
    q = (f"Does this image contain {article(cname)}?" if kind == "presence" else f"How many {plural(cname)} can you see?")
    return dict(pair_id=f"{kind}-{split}-{cid}-{ia}-{ib}", suite="grounding", kind=kind, question=q, category=cname,
                items=[dict(image=image_key(split, ia), image_id=ia, target=ta, canonical=normalize_answer(ta),
                            answers=None, answers_norm=None),
                       dict(image=image_key(split, ib), image_id=ib, target=tb, canonical=normalize_answer(tb),
                            answers=None, answers_norm=None)], **(extra or {}))


def train_pools(cats, per, imgs, rng):
    imgs = sorted(imgs)
    pres, cnt = [], []
    for cid in sorted(cats):
        pos, neg = presence_sets(per, imgs, cid)
        if pos and neg:
            k = min(POOL_PER_CAT, len(pos))
            for ia in rng.choice(pos, size=k, replace=False):
                ib = int(rng.choice(neg))
                pres.append(g_pair("presence", "train2014", cid, cats[cid], int(ia), ib, "yes", "no"))
        by = count_sets(per, imgs, cid)
        avail = sorted(by)
        if len(avail) >= 2:
            seen = set()
            for _ in range(POOL_PER_CAT):
                a, b = rng.choice(avail, size=2, replace=False)
                ia, ib = int(rng.choice(by[a])), int(rng.choice(by[b]))
                if (ia, ib) in seen:
                    continue
                seen.add((ia, ib))
                cnt.append(g_pair("counting", "train2014", cid, cats[cid], ia, ib, str(a), str(b),
                                  dict(counts=[int(a), int(b)])))
    return pres, cnt


def eval_grounding(cats, per, imgs, quota, rng, used):
    imgs = sorted(i for i in imgs if image_key("val2014", i) not in used)
    out = []
    for kind, n in quota.items():
        got = []
        cat_order = [int(c) for c in rng.permutation(sorted(cats))]
        while len(got) < n:
            progress = False
            for cid in cat_order:
                if len(got) >= n:
                    break
                free = [i for i in imgs if image_key("val2014", i) not in used]
                if kind == "presence":
                    pos, neg = presence_sets(per, free, cid)
                    if not pos or not neg:
                        continue
                    ia, ib = int(rng.choice(pos)), int(rng.choice(neg))
                    p = g_pair("presence", "val2014", cid, cats[cid], ia, ib, "yes", "no")
                else:
                    by = count_sets(per, free, cid)
                    avail = sorted(by)
                    if len(avail) < 2:
                        continue
                    a, b = rng.choice(avail, size=2, replace=False)
                    ia, ib = int(rng.choice(by[a])), int(rng.choice(by[b]))
                    p = g_pair("counting", "val2014", cid, cats[cid], ia, ib, str(a), str(b), dict(counts=[int(a), int(b)]))
                if ia == ib:
                    continue
                used.update(it["image"] for it in p["items"])
                got.append(p); progress = True
            assert progress, f"cannot fill {kind} quota ({len(got)}/{n})"
        out += got
    return out


def mismatch_map(pairs, rng):
    """Pair-level derangement fixed before generation: item k of pair p gets image k of pair sigma(p)."""
    n = len(pairs)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            break
    m = {}
    for p, q in zip(range(n), perm):
        for k in range(2):
            src = pairs[p]["items"][k]
            m[f"{pairs[p]['pair_id']}#{k}"] = pairs[int(q)]["items"][k]["image"]
            assert m[f"{pairs[p]['pair_id']}#{k}"] not in {it["image"] for it in pairs[p]["items"]}
    return m


# ----------------------------------------------------------------------------- images
def fetch_images(keys, out, image_dirs, workers=32):
    dst_root = Path(out) / "images"
    todo = []
    for k in keys:
        split, iid = k.split(":")
        fn = coco_filename(split, iid)
        dst = dst_root / split / fn
        if dst.exists() and dst.stat().st_size > 0:
            continue
        src = None
        for d in image_dirs:
            for cand in (Path(d) / split / fn, Path(d) / "images" / split / fn, Path(d) / fn):
                if cand.exists():
                    src = cand; break
            if src:
                break
        todo.append((src, dst, f"http://images.cocodataset.org/{split}/{fn}"))

    def one(t):
        src, dst, url = t
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src is not None:
            dst.write_bytes(src.read_bytes()); return None
        for attempt in range(5):
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                dst.write_bytes(data); return None
            except Exception as e:
                err = e; time.sleep(2 * (attempt + 1))
        return f"{url}: {err}"

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        fails = [f for f in ex.map(one, todo) if f]
    assert not fails, f"{len(fails)} images failed, e.g. {fails[:3]}  (rerun to retry)"
    return dict(n_needed=len(keys), n_fetched=len(todo), seconds=time.time() - t0)


def image_path(out, key):
    split, iid = key.split(":")
    return Path(out) / "images" / split / coco_filename(split, iid)


def audit_sheets(out, pairs):
    """Label audit of the generated grounding evaluation pairs, to be checked by a person
    independently of any model prediction (COCO can omit objects / have ambiguous counts)."""
    import csv
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ad = Path(out) / "audit"; ad.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pairs:
        for k, it in enumerate(p["items"]):
            rows.append(dict(suite=p["split_name"], pair_id=p["pair_id"], item=k, kind=p["kind"], question=p["question"],
                             label=it["target"], image=str(image_path(out, it["image"]).relative_to(out)),
                             label_ok_YN="", note=""))
    with open(ad / "grounding_label_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    per_page = 6
    for pg in range(0, len(pairs), per_page):
        chunk = pairs[pg:pg + per_page]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(7, 3.3 * len(chunk)))
        axes = np.atleast_2d(axes)
        for r, p in enumerate(chunk):
            for k in range(2):
                ax = axes[r, k]
                ax.imshow(Image.open(image_path(out, p["items"][k]["image"])).convert("RGB"))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(f"label: {p['items'][k]['target']}", fontsize=10)
            axes[r, 0].set_title(f"{p['pair_id']}\n{p['question']}", fontsize=8, loc="left")
        fig.tight_layout(); fig.savefig(ad / f"grounding_audit_page{pg // per_page + 1:02d}.png", dpi=90); plt.close(fig)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--image_dir", nargs="*", default=[], help="attached COCO-2014 image folders (optional)")
    ap.add_argument("--skip_images", action="store_true")
    ap.add_argument("--small", action="store_true", help="SMOKE TEST ONLY: tiny quotas")
    a = ap.parse_args()
    out = Path(a.out); sd = out / "splits"; sd.mkdir(parents=True, exist_ok=True)
    global N_CAP_TRAIN_IMG, N_CAP_DEV, VQA_QUOTA, GROUND_QUOTA, POOL_PER_CAT
    if a.small:
        N_CAP_TRAIN_IMG, N_CAP_DEV, POOL_PER_CAT = 20, 4, 6
        VQA_QUOTA = {"train": {"yes/no": 4, "number": 2, "other": 4}, "dev": {"yes/no": 2, "number": 1, "other": 1},
                     "test": {"yes/no": 2, "number": 2, "other": 2}}
        GROUND_QUOTA = {"dev": {"presence": 2, "counting": 2}, "test": {"presence": 3, "counting": 3}}
    t0 = time.time()
    tok = load_tokenizer(); codec = PromptCodec(tok)
    paths = ensure_raw(a.raw_dir or out / "raw")
    L = {n: jload(p) for n, p in paths.items()}
    stats = {}

    # ---------------- natural VQAv2 complementary pairs
    vt, rt = vqa_candidates(L["v2_OpenEnded_mscoco_train2014_questions.json"], L["v2_mscoco_train2014_annotations.json"],
                            L["v2_mscoco_train2014_complementary_pairs.json"], "train2014", codec)
    vv, rv = vqa_candidates(L["v2_OpenEnded_mscoco_val2014_questions.json"], L["v2_mscoco_val2014_annotations.json"],
                            L["v2_mscoco_val2014_complementary_pairs.json"], "val2014", codec)
    stats["vqa_filter_train"], stats["vqa_filter_val"] = rt, rv
    used_eval = set()
    vqa_train = take_quota(vt, VQA_QUOTA["train"], np.random.default_rng(SEED))
    vqa_test = take_quota(vv, VQA_QUOTA["test"], np.random.default_rng(SEED + 1), used_eval)
    vqa_dev = take_quota(vv, VQA_QUOTA["dev"], np.random.default_rng(SEED + 2), used_eval)

    # ---------------- captions
    caps = {}
    for split, name in (("train2014", "captions_train2014.json"), ("val2014", "captions_val2014.json")):
        d = defaultdict(list)
        for c in sorted(L[name]["annotations"], key=lambda c: c["id"]):
            txt = " ".join(c["caption"].strip().split())
            if txt and codec.n_tokens(txt) <= CAPTION_CAP:
                d[c["image_id"]].append(dict(caption_id=c["id"], caption=txt))
        caps[split] = d
    rng = np.random.default_rng(SEED + 3)
    elig = sorted(i for i, v in caps["train2014"].items() if len(v) >= CAPS_PER_TRAIN_IMG)
    cap_imgs = [int(i) for i in rng.permutation(elig)[:N_CAP_TRAIN_IMG]]
    cap_train = []
    for iid in cap_imgs:
        pick = rng.choice(len(caps["train2014"][iid]), size=CAPS_PER_TRAIN_IMG, replace=False)
        for j in sorted(pick):
            c = caps["train2014"][iid][j]
            cap_train.append(dict(ex_id=f"cap-{c['caption_id']}", image=image_key("train2014", iid), image_id=iid,
                                  caption_id=c["caption_id"], target=c["caption"]))

    # ---------------- annotation-derived grounding
    cats_t, per_t = index_instances(L["instances_train2014.json"])
    cats_v, per_v = index_instances(L["instances_val2014.json"])
    train_img_ids = sorted(set(cap_imgs) | {it["image_id"] for p in vqa_train for it in p["items"]})
    pres_pool, cnt_pool = train_pools(cats_t, per_t, train_img_ids, np.random.default_rng(SEED + 4))
    val_imgs = sorted(im["id"] for im in L["instances_val2014.json"]["images"])
    g_test = eval_grounding(cats_v, per_v, val_imgs, GROUND_QUOTA["test"], np.random.default_rng(SEED + 5), used_eval)
    g_dev = eval_grounding(cats_v, per_v, val_imgs, GROUND_QUOTA["dev"], np.random.default_rng(SEED + 6), used_eval)

    # ---------------- caption development set (one caption per image, image-disjoint from all eval sets)
    rng = np.random.default_rng(SEED + 7)
    elig = [i for i in sorted(caps["val2014"]) if image_key("val2014", i) not in used_eval]
    cap_dev = []
    for iid in rng.permutation(elig):
        iid = int(iid)
        c = caps["val2014"][iid][int(rng.integers(len(caps["val2014"][iid])))]
        cap_dev.append(dict(ex_id=f"cap-{c['caption_id']}", image=image_key("val2014", iid), image_id=iid,
                            caption_id=c["caption_id"], target=c["caption"]))
        used_eval.add(image_key("val2014", iid))
        if len(cap_dev) == N_CAP_DEV:
            break

    # ---------------- isolation checks
    eval_sets = {"caption_dev": [e["image"] for e in cap_dev],
                 "grounding_dev": [it["image"] for p in g_dev for it in p["items"]],
                 "grounding_test": [it["image"] for p in g_test for it in p["items"]],
                 "vqa_dev": [it["image"] for p in vqa_dev for it in p["items"]],
                 "vqa_test": [it["image"] for p in vqa_test for it in p["items"]]}
    all_eval = [k for v in eval_sets.values() for k in v]
    assert len(all_eval) == len(set(all_eval)), "evaluation sets are not image-disjoint"
    train_keys = ({e["image"] for e in cap_train} | {it["image"] for p in vqa_train + pres_pool + cnt_pool for it in p["items"]})
    assert not (train_keys & set(all_eval)), "evaluation image leaked into training/replay"
    assert all(k.startswith("train2014:") for k in train_keys) and all(k.startswith("val2014:") for k in all_eval)

    mism = {"grounding_test": mismatch_map(g_test, np.random.default_rng(SEED + 8)),
            "vqa_test": mismatch_map(vqa_test, np.random.default_rng(SEED + 9))}
    for name, obj in (("captions_train", cap_train), ("captions_dev", cap_dev), ("vqa_train", vqa_train),
                      ("vqa_dev", vqa_dev), ("vqa_test", vqa_test), ("grounding_dev", g_dev), ("grounding_test", g_test),
                      ("grounding_pool_train", {"presence": pres_pool, "counting": cnt_pool}), ("mismatch_test", mism)):
        jdump(obj, sd / f"{name}.json")
    images = sorted(train_keys | set(all_eval))
    jdump({"all": images, "train": sorted(train_keys), "eval": sorted(all_eval)}, sd / "images.json")
    stats.update(n_caption_train=len(cap_train), n_caption_train_images=len(cap_imgs), n_caption_dev=len(cap_dev),
                 n_vqa=dict(train=len(vqa_train), dev=len(vqa_dev), test=len(vqa_test)),
                 n_grounding=dict(dev=len(g_dev), test=len(g_test)),
                 grounding_pool=dict(presence=len(pres_pool), counting=len(cnt_pool), images=len(train_img_ids)),
                 n_images=dict(train=len(train_keys), eval=len(all_eval)),
                 label_filters=dict(min_area_px=MIN_AREA, counting="no crowd boxes and every instance >= min area, count 1-4",
                                    presence="positive: >=1 non-crowd instance >= min area; negative: no annotation of the category"),
                 prompt_template=codec.template_text, caption_instruction=CAPTION_INSTR, qa_instruction=QA_INSTR,
                 seconds_selection=time.time() - t0)
    print(json.dumps({k: v for k, v in stats.items() if k != "prompt_template"}, indent=1))
    if not a.skip_images:
        stats["image_fetch"] = fetch_images(images, out, a.image_dir)
        for p in g_dev: p["split_name"] = "grounding_dev"
        for p in g_test: p["split_name"] = "grounding_test"
        stats["audit_rows"] = audit_sheets(out, g_dev + g_test)
    jdump(stats, sd / "selection_stats.json")


if __name__ == "__main__":
    main()

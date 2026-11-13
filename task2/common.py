"""Shared configuration and utilities for PA3 Task 2 (SigLIP -> MLP connector -> Qwen3-0.6B VLM)."""
import json, math, os, random, re, time, hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- constants (from the handout)
MODEL_ID = os.environ.get("T2_MODEL_ID", "Qwen/Qwen3-0.6B")
VISION_ID = os.environ.get("T2_VISION_ID", "google/siglip-base-patch16-224")
SEED = 6304
GRID = 14                                  # 14 x 14 = 196 patch features
MICRO_BS, GRAD_ACCUM = 8, 2
EFF_BS = MICRO_BS * GRAD_ACCUM             # 16
WEIGHT_DECAY, CLIP, BETAS, EPS = 0.01, 1.0, (0.9, 0.999), 1e-8
WARMUP_FRAC, MIN_LR_FRAC = 0.05, 0.10
Q_CAP, CAPTION_CAP, ANSWER_CAP = 80, 64, 16
GEN_MAX_NEW, GEN_BS = 16, 8
CAPTION_GEN_MAX = 64
N_BOOT = 4000
LORA_CFG = dict(r=16, alpha=32, dropout=0.05, targets=["q_proj", "k_proj", "v_proj", "o_proj"])

_smoke = os.environ.get("T2_SMOKE_UPDATES")          # e.g. "12,8,4" -- ONLY for the offline smoke test
_upd = [int(x) for x in _smoke.split(",")] if _smoke else [2250, 1500, 750]
STAGES = [
    dict(name="caption_alignment", updates=_upd[0], lr_conn=1e-3, lr_lora=None, mix={"caption": 1.0}),
    dict(name="visual_instruction", updates=_upd[1], lr_conn=1e-4, lr_lora=2e-5,
         mix={"presence": 0.50, "counting": 0.25, "caption": 0.25}),
    dict(name="natural_vqa", updates=_upd[2], lr_conn=5e-5, lr_lora=1e-5,
         mix={"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}),
]
STAGE_ENDS = list(np.cumsum([s["updates"] for s in STAGES]))     # 2250, 3750, 4500
TOTAL_UPDATES = STAGE_ENDS[-1]
CKPT_EVERY = int(os.environ.get("T2_CKPT_EVERY", 250))
DEV_EVERY = int(os.environ.get("T2_DEV_EVERY", 500))
DEV_EVAL_AT = sorted(set(range(0, TOTAL_UPDATES + 1, DEV_EVERY)) | set(STAGE_ENDS))
SMOKE = bool(_smoke)

# data sizes
N_CAP_TRAIN_IMG, CAPS_PER_TRAIN_IMG, N_CAP_DEV = 6000, 3, 96
VQA_QUOTA = {"train": {"yes/no": 640, "number": 256, "other": 640},
             "dev": {"yes/no": 12, "number": 8, "other": 12},
             "test": {"yes/no": 36, "number": 24, "other": 36}}
GROUND_QUOTA = {"dev": {"presence": 12, "counting": 12}, "test": {"presence": 48, "counting": 48}}
POOL_PER_CAT = 150             # training pool pairs per category per question family
MIN_AREA = 32 * 32             # label-quality filter: ignore objects smaller than COCO "small"

CAPTION_INSTR = "Write a short caption describing this image."
QA_INSTR = "Answer the question using a single word or phrase."
SENT = "<<<IMAGE_SLOT>>>"


def default_out():
    return "/kaggle/working/task2_out" if Path("/kaggle/working").exists() else "outputs/task2"


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def jdump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def dir_size(p):
    p = Path(p)
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def stage_of(update_idx):
    """0-based optimizer-update index -> stage index (0, 1, 2)."""
    for k, e in enumerate(STAGE_ENDS):
        if update_idx < e:
            return k
    return len(STAGES) - 1


def stage_start(k):
    return 0 if k == 0 else STAGE_ENDS[k - 1]


def lr_mult(local_step, n_updates):
    """Per-stage: linear warmup over 5% of updates, then cosine decay to 10% of peak."""
    warm = max(1, int(round(WARMUP_FRAC * n_updates)))
    if local_step < warm:
        return (local_step + 1) / warm
    prog = (local_step - warm) / max(1, n_updates - warm - 1)
    return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


# ----------------------------------------------------------------------------- VQA answer normalisation
# Port of the official VQA evaluation normalisation (vqaEval.py, VQA team, BSD licence).
CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've"}
MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
              "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(text):
    out = text
    for p in PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out)


def _process_digit_article(text):
    words = []
    for w in text.lower().split():
        w = MANUAL_MAP.get(w, w)
        if w not in ARTICLES:
            words.append(w)
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize_answer(ans):
    if ans is None:
        return ""
    a = str(ans).replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punctuation(a)).strip()


def vqa_soft_accuracy(pred_norm, answers_norm):
    """s(a) = 1/10 * sum_i min(1, 1/3 * sum_{j != i} 1[a == a_j]) over the 10 human answers."""
    m = [a == pred_norm for a in answers_norm]
    tot = sum(m)
    return sum(min(1.0, (tot - mi) / 3.0) for mi in m) / len(answers_norm)


def consensus(raw_answers):
    """Most frequent normalised answer (earliest wins ties), its agreement count, and the most
    frequent *original* string inside that group (earliest wins ties) to use as the training target."""
    norm = [normalize_answer(a) for a in raw_answers]
    cnt = Counter(norm)
    best = max(cnt, key=lambda a: (cnt[a], -norm.index(a)))
    raws = [r.strip() for r, n in zip(raw_answers, norm) if n == best]
    rc = Counter(raws)
    target = max(rc, key=lambda r: (rc[r], -raws.index(r)))
    return best, cnt[best], target, norm


# ----------------------------------------------------------------------------- text / prompts
def load_tokenizer(model_id=MODEL_ID):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def end_of_turn_id(tok):
    i = tok.convert_tokens_to_ids("<|im_end|>")
    assert i is not None and i != tok.unk_token_id
    return i


def user_text(kind, question=None):
    return CAPTION_INSTR if kind == "caption" else f"{question}\n{QA_INSTR}"


class PromptCodec:
    """Builds [pre | visual slots | post | target] token sequences that follow Qwen's chat template:
    <|im_start|>user\\n <196 visual tokens> \\n{question}\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
    Visual tokens are inserted inside the user message, before the question."""

    def __init__(self, tok):
        self.tok = tok
        full = tok.apply_chat_template([{"role": "user", "content": SENT + "\n" + "QQQ"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        pre, post = full.split(SENT)
        self.pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
        self.post_tpl = post            # "\nQQQ<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.eot = end_of_turn_id(tok)
        self._cache = {}
        self.template_text = full

    def post_ids(self, utext):
        if utext not in self._cache:
            self._cache[utext] = self.tok(self.post_tpl.replace("QQQ", utext), add_special_tokens=False)["input_ids"]
        return self._cache[utext]

    def target_ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"] + [self.eot]

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])


def make_example(codec, image_key, kind, question, target, meta=None):
    return dict(image=image_key, kind=kind, post=codec.post_ids(user_text(kind, question)),
                target=codec.target_ids(target) if target is not None else None, meta=meta or {})


def pair_examples(codec, pair):
    """A complementary pair -> two examples (same question, two images, two targets)."""
    return [make_example(codec, it["image"], pair["kind"], pair["question"], it["target"],
                         dict(pair_id=pair["pair_id"], item=i)) for i, it in enumerate(pair["items"])]


# ----------------------------------------------------------------------------- features
def image_key(split, image_id):
    return f"{split}:{int(image_id)}"


def coco_filename(split, image_id):
    return f"COCO_{split}_{int(image_id):012d}.jpg"


class FeatureStore:
    """FP16 SigLIP patch features (N, 196, D) in a memory-mapped .npy, keyed by 'split:image_id'."""

    def __init__(self, out, n_tokens=196):
        out = Path(out)
        self.index = jload(out / "features" / "index.json")
        self.arr = np.load(out / "features" / "siglip_patch_fp16.npy", mmap_mode="r")
        self.n_tokens = n_tokens
        assert n_tokens in (196, 49)

    @property
    def dim(self):
        return self.arr.shape[-1]

    def get(self, keys, device):
        rows = [self.index[k] for k in keys]
        x = torch.from_numpy(np.stack([self.arr[r] for r in rows])).to(device)      # (B,196,D) fp16
        if self.n_tokens == 49:                                                     # 2x2 average pooling
            B, N, D = x.shape
            x = x.float().view(B, GRID // 2, 2, GRID // 2, 2, D).mean((2, 4)).reshape(B, 49, D).half()
        return x


# ----------------------------------------------------------------------------- model pieces
class Connector(nn.Module):
    """g_phi: Linear(768->1024) -> GELU -> Linear(1024->1024), then a fixed calibration scale."""

    def __init__(self, d_vis, d_lm):
        super().__init__()
        self.fc1 = nn.Linear(d_vis, d_lm)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_lm, d_lm)
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))) * self.scale


class LoRALinear(nn.Module):
    """y = W0 x + (alpha/r) B A dropout(x); W0 frozen (FP16), A random, B zero, both FP32, no bias."""

    def __init__(self, base, r, alpha, dropout, A_init):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(A_init.to(dev, torch.float32).clone())
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A)
        return self.base(x) + (self.scale * F.linear(h, self.lora_B)).to(self.base.weight.dtype)


def lora_targets(model):
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in LORA_CFG["targets"] and ".layers." in name:
            out.append((name, mod))
    return sorted(out, key=lambda x: x[0])


def attach_lora(model, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    r = LORA_CFG["r"]
    for name, lin in lora_targets(model):
        bound = 1.0 / math.sqrt(lin.in_features)                    # kaiming-uniform(a=sqrt 5), as in PEFT
        A = (torch.rand(r, lin.in_features, generator=g) * 2 - 1) * bound
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child,
                LoRALinear(lin, r, LORA_CFG["alpha"], LORA_CFG["dropout"], A))


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_state(model):
    sd = {}
    for n, m in lora_modules(model):
        sd[f"{n}.lora_A"] = m.lora_A.detach().float().cpu().clone()
        sd[f"{n}.lora_B"] = m.lora_B.detach().float().cpu().clone()
    return sd


@torch.no_grad()
def load_lora_state(model, sd):
    if not lora_modules(model):
        attach_lora(model)
    for n, m in lora_modules(model):
        m.lora_A.copy_(sd[f"{n}.lora_A"]); m.lora_B.copy_(sd[f"{n}.lora_B"])


def load_lm(dtype=torch.float16, device=None):
    from transformers import AutoModelForCausalLM
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    m.config.use_cache = False
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device or get_device())


# ----------------------------------------------------------------------------- batching
def build_batch(codec, examples, feats, n_vis, with_image=True, with_target=True, left_pad=False,
                exclude_eot=False, device=None):
    """Returns ids, attention mask, position ids, labels, visual-slot mask. With with_image=False the
    visual slots are removed entirely (shorter prefix; masks/positions rebuilt), text unchanged."""
    seqs, labs, vis = [], [], []
    for ex in examples:
        img = [0] * n_vis if with_image else []
        tgt = ex["target"] if (with_target and ex["target"] is not None) else []
        ids = codec.pre_ids + img + ex["post"] + tgt
        lab = [-100] * (len(ids) - len(tgt)) + tgt
        if exclude_eot and tgt:
            lab[-1] = -100
        vm = [False] * len(codec.pre_ids) + [True] * len(img) + [False] * (len(ex["post"]) + len(tgt))
        seqs.append(ids); labs.append(lab); vis.append(vm)
    L = max(len(s) for s in seqs)
    B = len(seqs)
    ids = torch.full((B, L), codec.tok.pad_token_id, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    vmask = torch.zeros((B, L), dtype=torch.bool)
    for i, (s, l, v) in enumerate(zip(seqs, labs, vis)):
        n = len(s)
        sl = slice(L - n, L) if left_pad else slice(0, n)
        ids[i, sl] = torch.tensor(s); lab[i, sl] = torch.tensor(l)
        att[i, sl] = 1; vmask[i, sl] = torch.tensor(v)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    dev = device or get_device()
    out = dict(ids=ids.to(dev), att=att.to(dev), pos=pos.to(dev), labels=lab.to(dev), vmask=vmask.to(dev))
    out["feats"] = feats if with_image else None
    return out


def embed(model, connector, batch):
    emb = model.get_input_embeddings()(batch["ids"])
    if batch["feats"] is not None:
        z = connector(batch["feats"].to(next(connector.parameters()).dtype)
                      if not torch.is_autocast_enabled() else batch["feats"])
        assert z.shape[0] * z.shape[1] == int(batch["vmask"].sum()), "visual slot count mismatch"
        emb = emb.masked_scatter(batch["vmask"].unsqueeze(-1), z.to(emb.dtype))
    return emb


def per_example_nll(model, connector, batch, chunk=1024):
    """Mean token NLL over supervised positions within each example -> (B,), plus token counts."""
    from torch.utils.checkpoint import checkpoint
    emb = embed(model, connector, batch)
    h = model.model(inputs_embeds=emb, attention_mask=batch["att"], position_ids=batch["pos"]).last_hidden_state
    h = h[:, :-1]; tgt = batch["labels"][:, 1:]
    mask = tgt != -100
    rows = torch.nonzero(mask, as_tuple=True)[0]
    hs, ts = h[mask], tgt[mask]
    head = model.lm_head
    ce = []
    for i in range(0, hs.shape[0], chunk):
        hc, tc = hs[i:i + chunk], ts[i:i + chunk]
        if torch.is_grad_enabled():
            ce.append(checkpoint(lambda a, b: F.cross_entropy(head(a).float(), b, reduction="none"), hc, tc,
                                 use_reentrant=False))
        else:
            ce.append(F.cross_entropy(head(hc).float(), tc, reduction="none"))
    ce = torch.cat(ce)
    B = batch["ids"].shape[0]
    s = torch.zeros(B, device=ce.device, dtype=torch.float32).index_add(0, rows, ce)
    n = torch.zeros(B, device=ce.device, dtype=torch.float32).index_add(0, rows, torch.ones_like(ce))
    return s / n.clamp(min=1), n


@torch.no_grad()
def greedy_generate(model, connector, codec, examples, feats, n_vis, with_image, max_new, device):
    """Batched greedy decoding with KV cache from inputs_embeds (left padding, explicit positions)."""
    b = build_batch(codec, examples, feats, n_vis, with_image=with_image, with_target=False, left_pad=True,
                    device=device)
    emb = embed(model, connector, b)
    out = model.model(inputs_embeds=emb, attention_mask=b["att"], position_ids=b["pos"], use_cache=True)
    pkv = out.past_key_values
    logits = model.lm_head(out.last_hidden_state[:, -1]).float()
    att, pos = b["att"], b["pos"][:, -1]
    stop = {codec.eot, codec.tok.eos_token_id}
    B = len(examples)
    gen = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for step in range(max_new):
        nxt = logits.argmax(-1)
        for i in range(B):
            if not done[i]:
                t = int(nxt[i])
                if t in stop:
                    done[i] = True
                else:
                    gen[i].append(t)
        if bool(done.all()) or step == max_new - 1:
            break
        att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=device)], 1)
        pos = pos + 1
        out = model.model(input_ids=nxt[:, None], attention_mask=att, position_ids=pos[:, None],
                          past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        logits = model.lm_head(out.last_hidden_state[:, -1]).float()
    hit_cap = [not bool(done[i]) for i in range(B)]
    return [codec.tok.decode(g, skip_special_tokens=True).strip() for g in gen], [len(g) for g in gen], hit_cap


# ----------------------------------------------------------------------------- bootstrap
def pair_bootstrap(values_by_pair, n_boot=N_BOOT, seed=SEED, stat=np.mean):
    """values_by_pair: (n_pairs, k) array (k = 2 questions per pair, or 1 for pair-level metrics).
    Resamples complete pairs. Returns (point, lo, hi)."""
    v = np.asarray(values_by_pair, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    rng = np.random.default_rng(seed)
    n = v.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(v[i].ravel()) for i in idx])
    return float(stat(v.ravel())), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

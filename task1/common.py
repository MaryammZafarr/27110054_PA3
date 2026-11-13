import json, math, os, random, re, time, hashlib, unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = os.environ.get("T1_MODEL_ID", "Qwen/Qwen3-0.6B")
SEED = 6304
N_TRAIN = 40_000
MAX_LEN = 2048
MICRO_BS = 2
GRAD_ACCUM = 16
EFF_BS = MICRO_BS * GRAD_ACCUM            # 32
TOTAL_UPDATES = N_TRAIN // EFF_BS          # 1250
WARMUP = 38
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8
CLIP = 1.0
CKPT_UPDATES = [312, 625, 937, 1250]  
# env overrides exist ONLY so the smoke test can point at its short runs; never set them for real runs
MID_UPDATE = int(os.environ.get("T1_MID_UPDATE", 625))
FINAL_UPDATE = int(os.environ.get("T1_FINAL_UPDATE", 1250))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DIAG_LAYERS = [0, 13, 27]
PROTECT_K = 128

METHODS = {
    "full_ft":  dict(kind="full", lr=2e-5, lam=0.0),
    "l2sp":     dict(kind="full", lr=2e-5, lam=1e-3),
    "lora4":    dict(kind="lora", lr=2e-4, r=4,  alpha=8,  dropout=0.05),
    "lora16":   dict(kind="lora", lr=2e-4, r=16, alpha=32, dropout=0.05),
    "oplora16": dict(kind="oplora", lr=2e-4, r=16, alpha=32, dropout=0.05, k=PROTECT_K),
}
# the seven evaluation conditions: (name, method, update)
CONDITIONS = [
    ("baseline", None, 0),
    ("full_ft_mid", "full_ft", MID_UPDATE),
    ("full_ft", "full_ft", FINAL_UPDATE),
    ("l2sp", "l2sp", FINAL_UPDATE),
    ("lora4", "lora4", FINAL_UPDATE),
    ("lora16", "lora16", FINAL_UPDATE),
    ("oplora16", "oplora16", FINAL_UPDATE),
]
COND = {c[0]: c for c in CONDITIONS}

BENCH = {  # task name in lm-eval -> (expected n, primary metric, extra metrics)
    "gsm8k_cot": (1319, "exact_match,flexible-extract", ["exact_match,strict-match"]),
    "hellaswag": (10042, "acc_norm,none", ["acc,none"]),
    "boolq":     (3270, "acc,none", []),
    "arc_easy":  (2376, "acc_norm,none", ["acc,none"]),
}
GSM_MAX_NEW = 1024
QUAL_MAX_NEW = 512


def default_out():
    return "/kaggle/working/task1_out" if Path("/kaggle/working").exists() else "outputs/task1"


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jdump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def dir_size(path):
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def sha(t: torch.Tensor):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def normalize_text(s: str) -> str:
    """Text normalisation used for GSM8K-test decontamination."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def load_tokenizer(model_id=MODEL_ID):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_prompt(tok, user, system=None):
    """Non-thinking chat prompt ending with the assistant generation header."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def end_of_turn_id(tok):
    i = tok.convert_tokens_to_ids("<|im_end|>")
    assert i is not None and i != tok.unk_token_id, "<|im_end|> token not found"
    return i


def encode_example(tok, question, response):
    p = tok(chat_prompt(tok, question), add_special_tokens=False)["input_ids"]
    r = tok(response, add_special_tokens=False)["input_ids"] + [end_of_turn_id(tok)]
    return p, r


def collate(examples, pad_id, device):
    """Right-pad a microbatch. labels = -100 on prompt and padding."""
    L = max(len(p) + len(r) for p, r in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for i, (p, r) in enumerate(examples):
        seq = p + r
        ids[i, :len(seq)] = torch.tensor(seq)
        lab[i, len(p):len(seq)] = torch.tensor(r)
        att[i, :len(seq)] = 1
    return ids.to(device), lab.to(device), att.to(device)


# ----------------------------------------------------------------------------- model loading
def load_model(dtype=torch.float16, model_id=MODEL_ID, device=None, use_cache=False):
    from transformers import AutoModelForCausalLM
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}
    m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    m.config.use_cache = use_cache          # off for training, on for generation
    return m.to(device or get_device())


def unique_params(model):
    """named_parameters() already de-duplicates tied tensors (embed/lm_head counted once)."""
    return list(model.named_parameters())


def target_linears(model):
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGETS and ".layers." in name:
            out.append((name, mod))
    return sorted(out, key=lambda x: x[0])


def get_submodule(model, name):
    return model.get_submodule(name)


def set_submodule(model, name, new):
    parent, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), child, new)


def down_proj_name(layer):
    return f"model.layers.{layer}.mlp.down_proj"


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r, alpha, dropout, A_init, Uk=None, Vk=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(A_init.to(dev, torch.float32).clone())             
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))       
        self.dropout = nn.Dropout(dropout)
        self.protected = Uk is not None
        if self.protected:
            self.register_buffer("Uk", Uk.to(dev, torch.float32), persistent=False)
            self.register_buffer("Vk", Vk.to(dev, torch.float32), persistent=False)

    def effective_factors(self):
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            A, B = self.lora_A.float(), self.lora_B.float()
            if self.protected:
                B = B - self.Uk @ (self.Uk.T @ B)
                A = A - (A @ self.Vk) @ self.Vk.T
        return A, B

    def delta_weight(self):
        A, B = self.effective_factors()
        return self.scale * (B @ A)

    def forward(self, x):
        out = self.base(x)
        A, B = self.effective_factors()
        # under autocast these matmuls run in FP16; without autocast (merge check) they run in FP32
        h = F.linear(self.dropout(x).to(A.dtype), A)
        return out + (self.scale * F.linear(h, B)).to(out.dtype)


def lora_init_A(names_shapes, r, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, din in names_shapes:  # sorted order
        bound = 1.0 / math.sqrt(din)
        out[name] = (torch.rand(r, din, generator=g) * 2 - 1) * bound
    return out


@torch.no_grad()
def compute_bases(model, k=PROTECT_K):
    """Leading-k singular bases of every target W0, computed once in FP32."""
    bases = {}
    for name, lin in target_linears(model):
        W = lin.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        bases[name] = (U[:, :k].contiguous(), Vh[:k].T.contiguous())
    return bases


def attach_adapters(model, r, alpha, dropout, bases=None, seed=SEED):
    lins = target_linears(model)
    inits = lora_init_A([(n, l.in_features) for n, l in lins], r, seed)
    for name, lin in lins:
        Uk, Vk = bases[name] if bases is not None else (None, None)
        set_submodule(model, name, LoRALinear(lin, r, alpha, dropout, inits[name], Uk, Vk))
    return inits


def adapter_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


@torch.no_grad()
def export_adapter_state(model):
    sd = {}
    for n, m in adapter_modules(model):
        A, B = m.effective_factors()
        sd[f"{n}.lora_A"] = m.lora_A.detach().float().cpu().clone()
        sd[f"{n}.lora_B"] = m.lora_B.detach().float().cpu().clone()
        sd[f"{n}.eff_A"] = A.detach().cpu().clone()          # == raw factors for plain LoRA
        sd[f"{n}.eff_B"] = B.detach().cpu().clone()
    return sd


@torch.no_grad()
def merge_adapter_into(model, adapter_sd, scale):
    names = sorted({k.rsplit(".", 1)[0] for k in adapter_sd})
    for n in names:
        lin = get_submodule(model, n)
        A, B = adapter_sd[f"{n}.eff_A"], adapter_sd[f"{n}.eff_B"]
        dW = scale * (B.to(lin.weight.device) @ A.to(lin.weight.device))
        lin.weight.copy_((lin.weight.float() + dW).to(lin.weight.dtype))
    return model


@torch.no_grad()
def load_unmerged_adapter(model, adapter_sd, r, alpha):
    names = sorted({k.rsplit(".", 1)[0] for k in adapter_sd})
    for n in names:
        lin = get_submodule(model, n)
        m = LoRALinear(lin, r, alpha, 0.0, adapter_sd[f"{n}.eff_A"])
        m.lora_B.copy_(adapter_sd[f"{n}.eff_B"])
        set_submodule(model, n, m)
    return model


def sft_loss_sum(model, ids, labels, att, chunk=1024):
    from torch.utils.checkpoint import checkpoint
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    h = h[:, :-1]; tgt = labels[:, 1:]
    mask = tgt != -100
    hs, ts = h[mask], tgt[mask]
    head = model.lm_head

    def piece(hc, tc):
        return F.cross_entropy(head(hc).float(), tc, reduction="sum")

    total = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], chunk):
        total = total + checkpoint(piece, hs[i:i + chunk], ts[i:i + chunk], use_reentrant=False)
    return total, int(mask.sum())


def hellaswag_preprocess(text):
    try:
        from lm_eval.tasks.hellaswag.utils import preprocess
        return preprocess(text)
    except Exception:
        text = text.strip().replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        return text.replace("  ", " ")


def hellaswag_context(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return hellaswag_preprocess(doc["activity_label"] + ": " + ctx)


def gsm8k_cot_user_text(question):
    return f"Q: {question}\nA:"   # gsm8k_cot doc_to_text, zero-shot
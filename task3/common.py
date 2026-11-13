"""Shared code for PA3 Task 3 (test-time compute): data loading, splits, answer normalisation/equivalence,
answer groups, quality scores and selection rules."""
import json, math, os, re, time
from pathlib import Path

import numpy as np

SEED = 6304
BUDGETS = [1, 2, 4, 8, 16, 32]
N_ORDERS = 10
N_ROLL = 32
N_DEV = 200
N_LIVE = 50
N_BOOT = 1000
CHECKS = [4, 8, 16]
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]
NO_ANSWER = "NO_ANSWER"
DATASETS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
SCORER_VERSION = "t3-scorer-1 (math-verify for MATH; MathVista normalize_extracted_answer port, letters for MC)"
LIVE_MODEL = os.environ.get("T3_LIVE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LIVE_REVISION = os.environ.get("T3_LIVE_REVISION", "aa8e72537993ba99e69dfaafa59ed015b17504d1")
SYSTEM_MSG = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")


def default_out():
    return "/kaggle/working/task3_out" if Path("/kaggle/working").exists() else "outputs/task3"


def jdump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def find_data_root(hint=None):
    """Directory that contains qwen25-math/ and qwen25vl-mathvista/."""
    cands = [Path(hint)] if hint else []
    cands += [Path("/kaggle/input"), Path("/kaggle/working/ATML-PA3-Data"), Path("ATML-PA3-Data")]
    for c in cands:
        if not c.exists():
            continue
        for g in sorted(c.rglob("generations.parquet")):
            if g.parent.name == DATASETS["math"]:
                return g.parent.parent
    return None


# ----------------------------------------------------------------------------- splits / orders
def make_splits(qids):
    """Sort IDs lexicographically, permute with default_rng(6304): first 200 dev, remaining 800 eval."""
    s = sorted(str(q) for q in qids)
    perm = np.random.default_rng(SEED).permutation(len(s))
    order = [s[i] for i in perm]
    return dict(permuted=order, dev=order[:N_DEV], eval=order[N_DEV:])


def make_orders(qids_sorted):
    """10 seeded permutations of the 32 rollout IDs per question (questions in lexicographic order)."""
    rng = np.random.default_rng(SEED)
    return {q: [rng.permutation(N_ROLL).tolist() for _ in range(N_ORDERS)] for q in qids_sorted}


# ----------------------------------------------------------------------------- answer extraction (live)
def last_boxed(text):
    """README rule: last literal '\\boxed{', scan balanced braces, keep nested LaTeX, strip whitespace.
    Absent / empty / unclosed last box -> None. No numeric fallback."""
    if text is None:
        return None
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{"); depth = 1; k = j
    while k < len(text):
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                s = text[j:k].strip()
                return s if s else None
        k += 1
    return None


# ----------------------------------------------------------------------------- MATH equivalence (math-verify)
class MathScorer:
    def __init__(self):
        from math_verify import parse, verify
        self._parse, self._verify = parse, verify
        self._pc = {}

    def p(self, s):
        if s not in self._pc:
            try:
                self._pc[s] = self._parse("\\boxed{" + s + "}")
            except Exception:
                self._pc[s] = []
        return self._pc[s]

    def eq(self, a, b):
        """Equivalence of two predictions (no gold involved): string identity or math-verify either way."""
        if a == b:
            return True
        pa, pb = self.p(a), self.p(b)
        if not pa or not pb:
            return False
        try:
            return bool(self._verify(pa, pb) or self._verify(pb, pa))
        except Exception:
            return False

    def grade(self, gold, pred):
        if pred is None or pred == NO_ANSWER:
            return False
        g = gold if gold is not None else ""
        g = last_boxed(g) or g
        if g.strip() == pred.strip():
            return True
        pg, pp = self.p(g), self.p(pred)
        if not pg or not pp:
            return False
        try:
            return bool(self._verify(pg, pp))
        except Exception:
            return False

    @staticmethod
    def version():
        import importlib.metadata as m
        return m.version("math-verify")


def clean_extraction(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    return s if s else None


# ----------------------------------------------------------------------------- MathVista rules (ported)
def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mathvista_normalize(extraction, meta):
    """Port of MathVista evaluation/calculate_score.py::normalize_extracted_answer, except that multiple-choice
    answers are represented by option letters (A, B, ...) as the handout requires. Returns None if unparseable."""
    qt, at = meta["question_type"], meta["answer_type"]
    choices = list(meta.get("choices") or [])
    if qt == "multi_choice":
        e = str(extraction).strip() if extraction is not None else ""
        letter = re.findall(r"\(([a-zA-Z])\)", e)
        if letter:
            e = letter[0].upper()
        seq = [chr(ord("A") + i) for i in range(len(choices))]
        if e in seq:
            return e
        d = [levenshtein(e, str(c)) for c in choices]
        return seq[d.index(min(d))]
    if at == "integer":
        try:
            return str(int(float(extraction)))
        except Exception:
            return None
    if at == "float":
        try:
            return str(round(float(extraction), int(meta.get("precision") or 0)))
        except Exception:
            return None
    try:
        return str(extraction)
    except Exception:
        return None


def mathvista_gold(gold, meta):
    if meta["question_type"] == "multi_choice":
        choices = [str(c) for c in (meta.get("choices") or [])]
        if str(gold) in choices:
            return chr(ord("A") + choices.index(str(gold)))
        return mathvista_normalize(gold, meta)
    return mathvista_normalize(gold, meta) or str(gold)


# ----------------------------------------------------------------------------- groups
def build_groups(answers, eq):
    """answers: list of 32 cleaned strings or None (rollout-id order). Groups formed WITHOUT gold:
    None -> NO_ANSWER; otherwise join the first existing group whose representative is equivalent."""
    reps, gid = [], []
    no_id = None
    for a in answers:
        if a is None:
            if no_id is None:
                reps.append(NO_ANSWER); no_id = len(reps) - 1
            gid.append(no_id); continue
        for k, r in enumerate(reps):
            if r != NO_ANSWER and eq(a, r):
                gid.append(k); break
        else:
            reps.append(a); gid.append(len(reps) - 1)
    return gid, reps


# ----------------------------------------------------------------------------- ranking / selection
def avg_rank_desc(q):
    """Descending average ranks (1 = best). NaN scores are treated as worst."""
    q = np.where(np.isnan(q), -np.inf, np.asarray(q, dtype=float))
    n = len(q)
    order = np.argsort(-q, kind="stable")
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and q[order[j + 1]] == q[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pick_max(scores):
    """Index (within prefix) of the highest score; ties -> earliest position."""
    s = np.where(np.isnan(scores), -np.inf, scores)
    return int(np.flatnonzero(s == s.max())[0])


def vote_winner(groups, weights):
    """groups/weights in current reveal order. Returns (winning group, W dict). Ties -> group whose earliest
    supporting rollout comes first."""
    W, first = {}, {}
    for pos, (g, w) in enumerate(zip(groups, weights)):
        W[g] = W.get(g, 0.0) + float(w)
        first.setdefault(g, pos)
    best = max(W.values())
    tied = [g for g in W if abs(W[g] - best) <= 1e-12]
    return min(tied, key=lambda g: first[g]), W


def rwc_weights(scores):
    n = len(scores)
    if n == 1:
        return np.ones(1)
    r = avg_rank_desc(scores)
    return (n - r) / (n - 1)


def rwc_margin(W):
    v = sorted(W.values(), reverse=True)
    tot = sum(v)
    if tot <= 0:
        return 0.0
    return (v[0] - (v[1] if len(v) > 1 else 0.0)) / tot


# ----------------------------------------------------------------------------- VisConf
class TokenCal:
    """Per-rollout sorted KL values with prefix sums for O(log T) evaluation of mean clip((k-a)/(b-a))."""

    def __init__(self, k, v):
        self.k = np.asarray(k, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.T = len(self.k)
        self.ks = np.sort(self.k)
        self.cs = np.concatenate([[0.0], np.cumsum(self.ks)])

    def mean_c(self, a, b):
        if self.T == 0:
            return float("nan")
        d = max(b - a, 1e-8)
        lo = np.searchsorted(self.ks, a, side="right")       # k <= a -> 0
        hi = np.searchsorted(self.ks, b, side="left")        # k >= b -> 1
        mid = (self.cs[hi] - self.cs[lo]) - a * (hi - lo)
        return float((mid / d + (self.T - hi)) / self.T)

    def mean_cv(self, a, b):
        if self.T == 0:
            return float("nan")
        c = np.clip((self.k - a) / max(b - a, 1e-8), 0.0, 1.0)
        return float(np.mean(c * self.v))


def calib_bounds(cals):
    pool = np.concatenate([c.k for c in cals if c.T]) if any(c.T for c in cals) else np.zeros(1)
    return float(np.percentile(pool, 5)), float(np.percentile(pool, 95))      # linear interpolation (default)


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=SEED):
    """Question-level paired bootstrap of mean(a) - mean(b) (inputs already averaged over orders)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

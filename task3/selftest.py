"""Unit tests (no data, no GPU).  python selftest.py"""
import numpy as np

from common import *

OK = []


def ok(name, cond, detail=""):
    OK.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail != "" else ""))


def main():
    ok("last_boxed: nested braces, last box, strip", last_boxed("a \\boxed{1} b \\boxed{ \\frac{1}{2} }") == "\\frac{1}{2}")
    ok("last_boxed: absent / empty / unclosed -> None",
       last_boxed("no box 5") is None and last_boxed("\\boxed{ }") is None and last_boxed("\\boxed{1} \\boxed{2") is None)
    sc = MathScorer()
    ok("math-verify equivalence: 1/2 == 0.5 == \\frac{1}{2}", sc.eq("\\frac{1}{2}", "0.5") and sc.eq("1/2", "\\frac12"))
    ok("math-verify grading uses gold; NO_ANSWER always wrong", sc.grade("\\frac{1}{2}", "0.5") and not sc.grade("2", "3")
       and not sc.grade("2", NO_ANSWER) and not sc.grade("2", None))
    gid, reps = build_groups(["0.5", None, "1/2", "3", None, "\\frac{1}{2}"], sc.eq)
    ok("groups: equivalents merged, all missing -> one NO_ANSWER group", gid == [0, 1, 0, 2, 1, 0] and reps[1] == NO_ANSWER, gid)
    meta_mc = dict(question_type="multi_choice", answer_type="text", choices=["red", "blue", "green"])
    ok("MathVista MC: letter, '(B) text', free text -> option letters",
       mathvista_normalize("B", meta_mc) == "B" and mathvista_normalize("(c) green", meta_mc) == "C"
       and mathvista_normalize("blue.", meta_mc) == "B" and mathvista_gold("green", meta_mc) == "C")
    ok("MathVista integer/float precision rules",
       mathvista_normalize("3.0", dict(question_type="free_form", answer_type="integer")) == "3"
       and mathvista_normalize("1.456", dict(question_type="free_form", answer_type="float", precision=2)) == "1.46"
       and mathvista_normalize("abc", dict(question_type="free_form", answer_type="integer")) is None)
    ok("average descending ranks with ties", list(avg_rank_desc(np.array([3., 1., 3., 2.]))) == [1.5, 4, 1.5, 3])
    w = rwc_weights(np.array([3., 1., 3., 2.]))
    ok("RWC weights (N - r)/(N - 1)", np.allclose(w, [(4 - 1.5) / 3, 0, (4 - 1.5) / 3, (4 - 3) / 3]))
    win, W = vote_winner([5, 7, 7, 5], [1, 1, 1, 1])
    ok("vote tie -> earliest supporting rollout", win == 5)
    ok("highest score tie -> earliest", pick_max(np.array([1., 3., 3.])) == 1)
    ok("RWC margin", abs(rwc_margin({1: 3.0, 2: 1.0}) - 0.5) < 1e-12 and rwc_margin({1: 2.0}) == 1.0)
    rng = np.random.default_rng(0)
    k = rng.normal(5, 2, 300); v = rng.random(300)
    c = TokenCal(k, v)
    a, b = np.percentile(k, 5), np.percentile(k, 95)
    brute = np.mean(np.clip((k - a) / (b - a), 0, 1))
    ok("fast calibrated-confidence mean == brute force", abs(c.mean_c(a, b) - brute) < 1e-10, (c.mean_c(a, b), brute))
    ct = np.clip((k - a) / (b - a), 0, 1)
    ok("tokenwise fusion identity Z_tok - V*C == Cov(c, v)",
       abs(c.mean_cv(a, b) - v.mean() * c.mean_c(a, b) - (np.mean(ct * v) - ct.mean() * v.mean())) < 1e-12)
    cals = [TokenCal(rng.random(10), rng.random(10)) for _ in range(3)]
    pa, pb = calib_bounds(cals)
    pool = np.concatenate([x.k for x in cals])
    ok("calibration pools only the given candidates (linear percentiles)", pa == np.percentile(pool, 5) and pb == np.percentile(pool, 95))
    ids = [f"test:{i}" for i in range(1000)]
    s1, s2 = make_splits(ids), make_splits(list(reversed(ids)))
    ok("splits: lexicographic sort then default_rng(6304) permutation; 200/800; order-invariant",
       s1 == s2 and len(s1["dev"]) == 200 and len(s1["eval"]) == 800 and not set(s1["dev"]) & set(s1["eval"]))
    o = make_orders(sorted(ids[:5]))
    ok("10 orders per question, each a permutation of 0..31", all(len(v) == 10 and all(sorted(p) == list(range(32)) for p in v)
                                                                 for v in o.values()))
    d, lo, hi = paired_bootstrap(np.array([1, 0, 1, 1.]), np.array([0, 0, 1, 0.]))
    ok("paired bootstrap point estimate and CI", abs(d - 0.5) < 1e-12 and lo <= d <= hi)
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    raise SystemExit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()

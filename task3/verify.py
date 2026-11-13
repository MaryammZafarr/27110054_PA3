"""Check Task 3 requirements against saved artefacts (whichever parts exist).

python verify.py --out OUT [--part offline|live|all]
"""
import argparse
from pathlib import Path

import pandas as pd

from common import *

RES = []


def check(name, ok, detail="", warn=False):
    st = "PASS" if ok else ("WARN" if warn else "FAIL")
    RES.append(dict(check=name, status=st, detail=str(detail)[:300]))
    print(f"[{st}] {name}" + (f" — {str(detail)[:200]}" if detail != "" else ""))


def offline_checks(out):
    pi = jload(out / "prepared" / "prepare_info.json")
    for key in ("math", "mathvista"):
        c = pi["checks"][key]
        check(f"{key}: 32,000 rows, 1,000 questions x 32 rollouts, files joined on (dataset, sample_id, rollout_id)",
              c["rows"] == 32000 and c["questions"] == 1000 and c["rollouts_per_question_32"] and c["keys_joined"], c["rows"])
        lens = [c["ids_len_eq_n"], c["logp_len_eq_n"], c["kl_len_eq_n"]] + ([c["att_len_eq_n"]] if key == "mathvista" else [])
        check(f"{key}: every per-token list aligned with generated_token_ids", all(x == c["rows"] for x in lens), lens)
        if key == "mathvista":
            check("mathvista: attention n_generated_text_tokens lists are 0..N-1; first generated-text attention is 0",
                  c["att_counts_are_0_to_N_minus_1"] and c["first_token_generated_attention_zero"])
        r = c["readme"]
        check(f"{key}: totals match README (tokens {r['tokens']:,}, capped {r['capped']}, null {r['null']})",
              (c["total_tokens"], c["capped"], c["null_extractions"]) == (r["tokens"], r["capped"], r["null"]),
              (c["total_tokens"], c["capped"], c["null_extractions"]), warn=True)
    si = jload(out / "prepared" / "scorer_info.json")
    check("scorer fixed and versioned before evaluation", "math_verify" in si and "frozen_at" in si, si["math_verify"])
    for key in ("math", "mathvista"):
        sp = jload(out / "splits" / f"{key}_splits.json")
        check(f"{key}: 200 dev / 800 eval, disjoint, from sorted IDs permuted with default_rng(6304)",
              len(sp["dev"]) == 200 and len(sp["eval"]) == 800 and not set(sp["dev"]) & set(sp["eval"])
              and make_splits(sp["permuted"]) == {k: sp[k] for k in ("permuted", "dev", "eval")})
        od = jload(out / "splits" / f"{key}_orders.json")
        check(f"{key}: 10 saved reveal orders per question", len(od) == 1000 and all(len(v) == 10 for v in od.values()))
    live = jload(out / "splits" / "live_questions.json")["live_math_questions"]
    check("live subset = first 50 MATH evaluation questions", live == jload(out / "splits" / "math_splits.json")["eval"][:50])
    od = out / "offline"
    for key in ("math", "mathvista"):
        S = pd.read_parquet(od / f"{key}_selection.parquet")
        need = {"first", "sc", "maxlogp", "maxsc", "sc_rwc", "shortest"} | (
            {"vision_max", "vision_rwc", "visconf_max", "visconf_rwc", "tokfuse_max", "tokfuse_rwc"} if key == "mathvista" else set())
        check(f"{key}: all required methods at N in {{1,2,4,8,16,32}} x 10 orders x 1,000 questions",
              set(S.method) == need and sorted(S.N.unique()) == BUDGETS and len(S) == 1000 * 10 * 6 * len(need), len(S))
        n1 = S[S.N == 1].pivot_table(index=["qid", "order"], columns="method", values="selected_rollout")
        check(f"{key}: at N=1 every rule returns the only candidate", bool((n1.nunique(axis=1) == 1).all()))
        sc = S[S.method == "sc"].sort_values(["qid", "order", "N"])
        mono = sc.groupby(["qid", "order"]).oracle.apply(lambda x: bool((x.astype(int).diff().fillna(0) >= 0).all())).all()
        check(f"{key}: oracle coverage non-decreasing over nested prefixes", bool(mono))
        f = S[(S.method == "first")]
        check(f"{key}: FirstAns charged only its first candidate; others charged all N",
              bool((f.groupby(["qid", "order"]).token_cost.nunique() == 1).all()))
        sel = jload(od / f"{key}_adaptive_selected.json")
        fx = sel["fixed32_dev"]["acc"]
        check(f"{key}: adaptive settings chosen on dev within 1 pp of fixed-32 SC and frozen before held-out evaluation",
              all(v["dev"]["acc"] >= fx - 0.01 - 1e-9 for v in sel["chosen"].values()), sel["chosen"])
        A = pd.read_parquet(od / f"{key}_adaptive_eval.parquet")
        check(f"{key}: adaptive held-out results for 800 questions x 10 orders x 2 controllers", len(A) == 800 * 10 * 2, len(A))
        check(f"{key}: stopping only at 4, 8, 16 or 32", set(A.n_used.unique()) <= {4, 8, 16, 32})
        if key == "mathvista":
            idn = jload(od / f"{key}_fusion_identity.json")
            check("tokenwise fusion identity Z_token - Z = Cov(c, v) holds numerically", idn["max_abs_identity_error"] < 1e-4,
                  idn["max_abs_identity_error"])


def live_checks(out):
    ld = out / "live"
    if not (ld / "calls.jsonl").exists():
        check("live generation run", False, "no live/calls.jsonl"); return
    info = jload(ld / "live_info.json")
    L = [json.loads(l) for l in open(ld / "calls.jsonl")]
    keys = {(r["qid"], r["call"]) for r in L}
    q = [x["qid"] for x in jload(ld / "live_questions.json")]
    check("live: 50 questions x 5 calls (single, 3 parallel, revision)", len(q) == 50 and len(keys) == 250 and info["complete"],
          f"{len(keys)} calls")
    check("live: pinned model revision, FP16, batch size 1", info["revision"] == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
          and info["dtype"] == "float16" and info["batch_size"] == 1, info["revision"])
    check("live: token caps 3072 / 1024 / 2048 respected", all(r["output_tokens"] <= r["cap"] for r in L)
          and {r["cap"] for r in L if r["call"] == "single"} == {3072} and {r["cap"] for r in L if r["call"] == "revision"} == {2048})
    check("live: independent seeds per call", len({r["seed"] for r in L}) == len(L))
    rev = [r for r in L if r["call"] == "revision"]
    init = {r["qid"]: r["response"] for r in L if r["call"] == "parallel0"}
    check("live: revision conditioned on the first parallel response + fixed review instruction",
          all(r["messages"][2]["content"] == init[r["qid"]] and r["messages"][3]["content"] == REVIEW for r in rev))
    check("live: prompts, responses, seeds, lengths, truncation, runtime saved",
          all(k in L[0] for k in ("prompt", "response", "seed", "input_tokens", "output_tokens", "truncated", "seconds")))
    check("live: GPU", "T4" in str(info.get("gpu")), info.get("gpu"), warn=True)


def report_checks(out, part):
    rd = out / "report_t3"
    files = []
    if part in ("offline", "all"):
        for k in ("math", "mathvista"):
            files += [f"fig_{k}_accuracy_vs_budget.png", f"{k}_accuracy_by_budget.csv", f"{k}_method_comparison.csv",
                      f"{k}_failure_decomposition.csv", f"{k}_signal_associations.csv", f"{k}_adaptive_vs_fixed.csv",
                      f"{k}_adaptive_stopping.csv", f"fig_{k}_adaptive_cost.png", f"{k}_qualitative.md"]
        files += ["mathvista_confidence_vision.csv", "fig_mathvista_trajectories.png"]
    if part in ("live", "all"):
        files += ["live_strategies.csv", "live_transitions.json", "live_qualitative.md"]
    for f in files:
        check(f"report artefact {f}", (rd / f).exists())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=default_out())
    ap.add_argument("--part", default="all", choices=["offline", "live", "all"])
    a = ap.parse_args(); out = Path(a.out)
    fns = ([offline_checks] if a.part in ("offline", "all") else []) + ([live_checks] if a.part in ("live", "all") else [])
    for f in fns:
        try:
            f(out)
        except Exception as e:
            check(f"{f.__name__} crashed", False, repr(e))
    report_checks(out, a.part)
    jdump(RES, out / f"verify_t3_{a.part}.json")
    n = {s: sum(r["status"] == s for r in RES) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed")

"""Offline experiments on the supplied rollouts (Steps 1, 2, 3, 5): repeated sampling, quality scores and
aggregation, signal analysis, adaptive stopping with dev-selected thresholds. CPU only.

python offline.py --out OUT
"""
import argparse, pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common import *

BASE = ["first", "sc", "maxlogp", "maxsc", "sc_rwc", "shortest"]
VLM = ["vision_max", "vision_rwc", "visconf_max", "visconf_rwc", "tokfuse_max", "tokfuse_rwc"]


def evaluate_prefix(e, idx, key, cals):
    """All selection methods on one revealed prefix. Returns {method: (selected rollout, group)} and extras."""
    g = e["gid"][idx]
    N = len(idx)
    res = {}

    def by_max(scores):
        p = pick_max(np.asarray(scores, float))
        return idx[p], g[p]

    def by_vote(weights):
        win, W = vote_winner(g.tolist(), weights)
        p = int(np.flatnonzero(g == win)[0])
        return (idx[p], win), W

    res["first"] = (idx[0], g[0])
    (res["sc"], Wsc) = by_vote(np.ones(N))
    res["maxlogp"] = by_max(e["L"][idx])
    res["maxsc"] = by_max(e["K"][idx])
    (res["sc_rwc"], Wk) = by_vote(rwc_weights(e["K"][idx]))
    res["shortest"] = by_max(-e["T"][idx].astype(float))
    extra = {}
    if key == "mathvista":
        V = e["V"][idx]
        res["vision_max"] = by_max(V)
        res["vision_rwc"], _ = by_vote(rwc_weights(V))
        cs = [cals[i] for i in idx]
        a, b = calib_bounds(cs)
        C = np.array([c.mean_c(a, b) for c in cs])
        Z = V * C
        Zt = np.array([c.mean_cv(a, b) for c in cs])
        res["visconf_max"] = by_max(Z)
        (res["visconf_rwc"], Wz) = by_vote(rwc_weights(Z))
        res["tokfuse_max"] = by_max(Zt)
        res["tokfuse_rwc"], _ = by_vote(rwc_weights(Zt))
        extra.update(Z=Z, C=C, Zt=Zt, a=a, b=b)
        Wq, qwin = Wz, res["visconf_rwc"][1]
    else:
        Wq, qwin = Wk, res["sc_rwc"][1]
    # stopping statistics (current self-consistency + quality-aware RWC)
    cnt = {}
    for x in g.tolist():
        cnt[x] = cnt.get(x, 0) + 1
    top = max(cnt.values())
    leaders = [x for x in cnt if cnt[x] == top]
    lead = leaders[0] if len(leaders) == 1 else None
    extra.update(sc_group=res["sc"][1], top_count=top, unique_leader=lead,
                 leader_is_noanswer=lead is not None and e["reps"][lead] == NO_ANSWER,
                 rwc_group=qwin, rwc_margin=rwc_margin(Wq))
    return res, extra


def run_dataset(key, out):
    P = pickle.load(open(out / "prepared" / f"{key}.pkl", "rb"))
    Q, orders, sp = P["Q"], P["orders"], P["splits"]
    split_of = {q: "dev" for q in sp["dev"]}
    split_of.update({q: "eval" for q in sp["eval"]})
    methods = BASE + (VLM if key == "mathvista" else [])
    rows, stops = [], []
    t0 = time.time()
    for qi, q in enumerate(sp["permuted"]):
        e = Q[q]
        cals = [TokenCal(k, v) for k, v in zip(e["k"], e["v"])] if key == "mathvista" else None
        for o, perm in enumerate(orders[q]):
            perm = np.array(perm)
            for N in BUDGETS:
                idx = perm[:N]
                res, ex = evaluate_prefix(e, idx, key, cals)
                tok_all = int(e["T"][idx].sum())
                oracle = bool(e["correct"][idx].any())
                ng = len(set(e["gid"][idx].tolist()))
                for m in methods:
                    sel, grp = res[m]
                    rows.append((q, split_of[q], o, N, m, int(sel), e["reps"][grp], bool(e["gcorr"][grp]),
                                 int(e["T"][idx[0]]) if m == "first" else tok_all, oracle, ng))
                if N >= 4:
                    stops.append((q, split_of[q], o, N, int(ex["sc_group"]), bool(e["gcorr"][ex["sc_group"]]),
                                  ex["top_count"], -1 if ex["unique_leader"] is None else int(ex["unique_leader"]),
                                  bool(ex["leader_is_noanswer"]), int(ex["rwc_group"]), float(ex["rwc_margin"]), tok_all))
        if qi % 100 == 0:
            print(f"{key}: {qi}/{len(sp['permuted'])} questions, {time.time() - t0:.0f}s", flush=True)
    sel = pd.DataFrame(rows, columns=["qid", "split", "order", "N", "method", "selected_rollout", "answer", "correct",
                                      "token_cost", "oracle", "n_groups"])
    st = pd.DataFrame(stops, columns=["qid", "split", "order", "n", "sc_group", "sc_correct", "top_count", "unique_leader",
                                      "leader_is_noanswer", "rwc_group", "rwc_margin", "token_cost"])
    od = out / "offline"; od.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(od / f"{key}_selection.parquet", index=False)
    st.to_parquet(od / f"{key}_stopping_stats.parquet", index=False)
    signals(key, Q, sp, od)
    adaptive(key, st, Q, od)
    return time.time() - t0


# ----------------------------------------------------------------------------- Step 3: signals
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    rx, ry = avg_rank_desc(x), avg_rank_desc(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def signals(key, Q, sp, od):
    rec, cv, traj, ident = [], [], [], []
    for q in sp["eval"]:
        e = Q[q]
        S = dict(length=e["T"].astype(float), neg_length=-e["T"].astype(float), mean_logp=e["L"], self_certainty=e["K"])
        if key == "mathvista":
            cals = [TokenCal(k, v) for k, v in zip(e["k"], e["v"])]
            a, b = calib_bounds(cals)
            C = np.array([c.mean_c(a, b) for c in cals])
            S.update(vision=e["V"], calibrated_conf=C, visconf=e["V"] * C,
                     tokfuse=np.array([c.mean_cv(a, b) for c in cals]))
            for i, c in enumerate(cals):
                if c.T >= 3:
                    ct = np.clip((c.k - a) / max(b - a, 1e-8), 0, 1)
                    cov = float(np.mean(ct * c.v) - np.mean(ct) * np.mean(c.v))
                    ident.append(abs(S["tokfuse"][i] - S["visconf"][i] - cov))
                    r_c = np.corrcoef(ct, c.v)[0, 1] if np.ptp(ct) > 0 and np.ptp(c.v) > 0 else np.nan
                    r_k = np.corrcoef(c.k, c.v)[0, 1] if np.ptp(c.k) > 0 and np.ptp(c.v) > 0 else np.nan
                    cv.append((q, i, r_c, r_k, cov, bool(e["correct"][i])))
        corr = e["correct"].astype(float)
        mixed = 0 < corr.mean() < 1
        for name, s in S.items():
            o = np.argsort(-np.where(np.isnan(s), -np.inf, s), kind="stable")
            rec.append(dict(qid=q, score=name, mixed=mixed, top_minus_bottom=corr[o[:8]].mean() - corr[o[-8:]].mean(),
                            spearman_correct=spearman(s, corr), spearman_length=spearman(s, e["T"])))
    df = pd.DataFrame(rec)
    df.to_parquet(od / f"{key}_signals_per_question.parquet", index=False)
    if cv:
        pd.DataFrame(cv, columns=["qid", "rollout", "r_calconf_vision", "r_kl_vision", "cov_c_v", "correct"]).to_parquet(
            od / f"{key}_conf_vision_corr.parquet", index=False)
        jdump(dict(max_abs_identity_error=float(np.max(ident)), n=len(ident),
                   identity="Z_token - Z = Cov_t(c, v) (population covariance)"), od / f"{key}_fusion_identity.json")
        # fixed illustrative trajectories: first 3 MathVista evaluation questions (permutation order), rollouts 0 and 1
        for q in sp["eval"][:3]:
            e = Q[q]
            cals = [TokenCal(k, v) for k, v in zip(e["k"], e["v"])]
            a, b = calib_bounds(cals)
            for i in (0, 1):
                c = cals[i]
                traj.append(dict(qid=q, rollout=i, correct=bool(e["correct"][i]), answer=e["answers_norm"][i],
                                 gold=str(e["gold"]), a=a, b=b, c=np.clip((c.k - a) / max(b - a, 1e-8), 0, 1).tolist(),
                                 v=c.v.tolist()))
        jdump(dict(rule="first 3 MathVista evaluation questions in the seed-6304 permutation; rollouts 0 and 1; "
                        "calibration pooled over all 32 rollouts", trajectories=traj), od / f"{key}_trajectories.json")


# ----------------------------------------------------------------------------- Step 5: adaptive stopping
def apply_policy(st, family, tau, gamma):
    """Returns per (qid, order) outcome: correct, token_cost, n_used."""
    st = st.sort_values(["qid", "order", "n"])
    out = []
    for (q, o), g in st.groupby(["qid", "order"], sort=False):
        g = g.set_index("n")
        stop = 32
        if tau is not None:
            for n in CHECKS:
                r = g.loc[n]
                ok = (r.unique_leader >= 0 and not r.leader_is_noanswer and r.top_count >= tau * n - 1e-9)
                if ok and family == "quality":
                    ok = (r.rwc_group == r.unique_leader) and (r.rwc_margin >= gamma - 1e-12)
                if ok:
                    stop = n; break
        r = g.loc[stop]
        out.append((q, o, bool(r.sc_correct), int(r.token_cost), stop))
    return pd.DataFrame(out, columns=["qid", "order", "correct", "token_cost", "n_used"])


def summarise(df):
    per_q = df.groupby("qid")[["correct", "token_cost", "n_used"]].mean()
    return dict(acc=float(per_q.correct.mean()), cost=float(per_q.token_cost.mean()), rollouts=float(per_q.n_used.mean()))


def adaptive(key, st, Q, od):
    dev, ev = st[st.split == "dev"], st[st.split == "eval"]
    fixed_dev = summarise(apply_policy(dev, "agreement", None, None))
    grid = []
    for fam in ("agreement", "quality"):
        settings = [(None, None)] + [(t, None) for t in TAUS] if fam == "agreement" else \
                   [(None, None)] + [(t, gm) for t in TAUS for gm in GAMMAS]
        for tau, gm in settings:
            s = summarise(apply_policy(dev, fam, tau, gm))
            grid.append(dict(family=fam, tau=tau, gamma=gm, **{f"dev_{k}": v for k, v in s.items()},
                             eligible=s["acc"] >= fixed_dev["acc"] - 0.01 - 1e-12))
    G = pd.DataFrame(grid)
    G.to_csv(od / f"{key}_adaptive_dev_grid.csv", index=False)
    chosen = {}
    for fam in ("agreement", "quality"):
        c = G[(G.family == fam) & G.eligible].copy()
        c["tau_k"] = c.tau.fillna(9.0); c["gamma_k"] = c.gamma.fillna(9.0)
        c = c.sort_values(["dev_cost", "dev_acc", "tau_k", "gamma_k"], ascending=[True, False, True, True])
        best = c.iloc[0]
        tau = None if pd.isna(best.tau) else float(best.tau)
        gm = None if pd.isna(best.gamma) else float(best.gamma)
        chosen[fam] = dict(tau=tau, gamma=gm, dev=dict(acc=best.dev_acc, cost=best.dev_cost, rollouts=best.dev_rollouts))
    jdump(dict(selection_rule="lowest mean dev token cost with dev accuracy >= fixed-32 SC accuracy - 1 pp; ties: higher "
                              "accuracy, then smaller tau, then smaller gamma; always-32 fallback included",
               fixed32_dev=fixed_dev, chosen=chosen, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               quality_score="self-certainty" if key == "math" else "VisConf"), od / f"{key}_adaptive_selected.json")
    # held-out evaluation with frozen settings
    res = {}
    full = apply_policy(ev, "agreement", None, None).rename(columns={"correct": "fixed32_correct"})
    for fam, ch in chosen.items():
        r = apply_policy(ev, fam, ch["tau"], ch["gamma"]).merge(full[["qid", "order", "fixed32_correct"]], on=["qid", "order"])
        corr_frac = {q: float(Q[q]["correct"].mean()) for q in r.qid.unique()}
        r["pool_correct_frac"] = r.qid.map(corr_frac)
        r["family"] = fam
        res[fam] = r
    pd.concat(res.values()).to_parquet(od / f"{key}_adaptive_eval.parquet", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=default_out())
    ap.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    a = ap.parse_args(); out = Path(a.out)
    info = {}
    for key in a.datasets:
        info[key] = dict(seconds=run_dataset(key, out))
        print(key, "done", info[key], flush=True)
    jdump(info, out / "offline" / "offline_info.json")


if __name__ == "__main__":
    main()

"""
Established functionally grounded metrics computed on exactly the explanations that HCT evaluated
(added in the revision).

  python baselines.py <dataset> <rf|mlp> [seed]

The script rebuilds the context of the main study (same split, model and explained instances) and reads the
attributions stored by run_experiments.py, so every metric below scores the same explanations as the main study (Table 4).

Removal-based faithfulness, per explained instance and explainer
  PGI_marg, PGI_cond  prediction gap on important features (OpenXAI): mean |f(x) - f(x')| when the three features
                      with the largest |a| are replaced by the values of random training rows (marginal) or of the
                      K nearest training rows on the remaining features (conditional). Sign-blind.
  DEL_marg, DEL_cond  deletion curve (area over the perturbation curve): features are removed one at a time in order
                      of their signed support for the predicted class, for k = 1..ceil(d/2), and the drop of the
                      predicted-class probability is averaged over k. Sign-aware.
  FC_mean, FC_cond    faithfulness correlation (Bhatt et al., 2020): Pearson correlation, over 50 random feature subsets
                      of size max(2, round(d/4)), between the attribution mass of the subset and the change in f when the
                      subset is set to the training mean (Bhatt's baseline) or to conditional values. Sign-aware.
  hor_*               share of the evaluation points of each metric that lie inside the HCT horizon.
Disagreement (Krishna et al.), per instance and pair of explainers (and each explainer against the whole)
  FA, RA, SA, SRA     feature, rank, sign and signed-rank agreement over the top three features
  RC                  Spearman correlation of |a| over all features
"""
import os
import sys
import time
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hct  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
EXPL = ["LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H", "Random", "SignFlip"]
K = hct.K_COND          # replacement draws per evaluation point set (as in the claim test)
KTOP = hct.TOPK         # top-k used by PGI and by the agreement metrics
N_SUB = 50              # random subsets for faithfulness correlation


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _nn(ctx, z, S):
    """K nearest rows of D_fit measured on the features outside S (conditional replacement)."""
    keep = np.setdiff1d(np.arange(ctx.d), np.atleast_1d(S))
    D = ((ctx.Zfit[:, keep] - z[keep]) ** 2).sum(axis=1)
    return np.argpartition(D, K)[:K]


def _pearson(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.std() < 1e-12 or v.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(u, v)[0, 1])


def instance_metrics(ctx, x, A, rng, horizon=True):
    """Removal-based metrics for one instance; A maps explainer name -> attribution vector (towards class 1)."""
    d = ctx.d
    z = ctx.z(x)
    kdel = int(np.ceil(d / 2))
    msub = max(2, int(round(d / 4)))
    p1 = float(ctx.f(x)[0])
    sgn = 1.0 if p1 >= 0.5 else -1.0
    marg = ctx.Xfit[rng.choice(len(ctx.Xfit), K, replace=True)]       # common random rows for every explainer
    subs = [np.sort(rng.choice(d, msub, replace=False)) for _ in range(N_SUB)]
    Q, spans, n_rows = [], {}, 0

    def put(key, Xr):
        nonlocal n_rows
        spans[key] = (n_rows, len(Xr))
        Q.append(Xr)
        n_rows += len(Xr)

    def repl(S, rows):
        Xr = np.repeat(x[None, :], len(rows), axis=0)
        Xr[:, S] = rows[:, S]
        return Xr

    for i, S in enumerate(subs):
        xm = x.copy()
        xm[S] = ctx.mu[S]
        put(("FCm", i), xm[None, :])
        put(("FCc", i), repl(S, ctx.Xfit[_nn(ctx, z, S)]))
    orders = {}
    for n, a in A.items():
        S3 = np.array(hct.topk(a, KTOP))
        put((n, "PGIm"), repl(S3, marg))
        put((n, "PGIc"), repl(S3, ctx.Xfit[_nn(ctx, z, S3)]))
        order = np.argsort(-(np.asarray(a) * sgn), kind="stable")
        orders[n] = order
        for k in range(1, kdel + 1):
            S = order[:k]
            put((n, "DELm", k), repl(S, marg))
            put((n, "DELc", k), repl(S, ctx.Xfit[_nn(ctx, z, S)]))
    Xall = np.vstack(Q)
    P = ctx.f(Xall)
    H = ctx.on_horizon(ctx.z(Xall)) if horizon else np.full(len(Xall), np.nan)

    def get(key):
        o, l_ = spans[key]
        return P[o:o + l_], H[o:o + l_]

    dfm = np.array([p1 - get(("FCm", i))[0].mean() for i in range(N_SUB)])
    dfc = np.array([p1 - get(("FCc", i))[0].mean() for i in range(N_SUB)])
    hor_fcm = float(np.mean([get(("FCm", i))[1].mean() for i in range(N_SUB)]))
    hor_fcc = float(np.mean([get(("FCc", i))[1].mean() for i in range(N_SUB)]))
    out = {}
    for n, a in A.items():
        a = np.asarray(a, float)
        mass = np.array([a[S].sum() for S in subs])
        pm, hm = get((n, "PGIm"))
        pc, hc = get((n, "PGIc"))
        dm = [sgn * (p1 - get((n, "DELm", k))[0].mean()) for k in range(1, kdel + 1)]
        dc = [sgn * (p1 - get((n, "DELc", k))[0].mean()) for k in range(1, kdel + 1)]
        out[n] = dict(PGI_marg=float(np.abs(p1 - pm).mean()), PGI_cond=float(np.abs(p1 - pc).mean()),
                      DEL_marg=float(np.mean(dm)), DEL_cond=float(np.mean(dc)),
                      FC_mean=_pearson(mass, dfm), FC_cond=_pearson(mass, dfc),
                      hor_PGI_marg=float(hm.mean()), hor_PGI_cond=float(hc.mean()),
                      hor_DEL_marg=float(np.mean([get((n, "DELm", k))[1].mean() for k in range(1, kdel + 1)])),
                      hor_DEL_cond=float(np.mean([get((n, "DELc", k))[1].mean() for k in range(1, kdel + 1)])),
                      hor_FC_mean=hor_fcm, hor_FC_cond=hor_fcc, pred=p1)
    return out


def agreement(a, b, k=KTOP):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ta, tb = hct.topk(a, k), hct.topk(b, k)
    common = set(ta) & set(tb)
    fa = len(common) / k
    ra = sum(ta[i] == tb[i] for i in range(k)) / k
    sa = sum(1 for j in common if np.sign(a[j]) == np.sign(b[j]) and a[j] != 0) / k
    sra = sum(1 for i in range(k) if ta[i] == tb[i] and np.sign(a[ta[i]]) == np.sign(b[tb[i]]) and a[ta[i]] != 0) / k
    rc = spearmanr(np.abs(a), np.abs(b)).correlation if np.abs(a).std() > 0 and np.abs(b).std() > 0 else np.nan
    return dict(FA=fa, RA=ra, SA=sa, SRA=sra, RC=float(rc))


def run_main(ds, mk, seed=0):
    t0 = time.time()
    X, y, cat = hct.load_dataset(ds, DATA, seed)
    ctx = hct.Context(X, y, cat, mk, seed, n_explain=100)
    Z = np.load(os.path.join(RES, f"main_{ds}_{mk}_s{seed}_arrays.npz"))
    assert np.array_equal(ctx.E, Z["E"]), "explained instances differ from the main study"
    A = {n: Z[f"attr_{n}"] for n in EXPL}
    G = Z["G_1.0"]
    rng = np.random.default_rng(7000 + seed)
    rows, drows = [], []
    for r, ti in enumerate(ctx.E):
        x = ctx.Xte[ti]
        m = instance_metrics(ctx, x, {n: A[n][r] for n in EXPL}, rng)
        for n in EXPL:
            rows.append(dict(explainer=n, inst=r, **m[n]))
        for e1, e2 in itertools.combinations(EXPL, 2):
            drows.append(dict(inst=r, e1=e1, e2=e2, **agreement(A[e1][r], A[e2][r])))
        for e1 in EXPL:
            drows.append(dict(inst=r, e1=e1, e2="Whole", **agreement(A[e1][r], G[r])))
        if (r + 1) % 25 == 0:
            log(f"  {ds}/{mk}: {r + 1}/{len(ctx.E)}")
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"baselines_{ds}_{mk}_s{seed}.csv"), index=False)
    pd.DataFrame(drows).to_csv(os.path.join(RES, f"disagree_{ds}_{mk}_s{seed}.csv"), index=False)
    log(f"{ds}/{mk}/s{seed} baselines done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run_main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)

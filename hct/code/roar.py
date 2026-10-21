"""
RemOve And Retrain (ROAR; Hooker et al., 2019) for the explainers of the main study (added in the revision).

  python roar.py <dataset> <rf|mlp> [seed]

Every training and test row is explained with the same explainers and settings as in the main study. For each
fraction t of the features, the t*d features with the largest |a| of each row are replaced by their training mean,
the model is retrained from scratch on the modified training rows (three seeds), and its accuracy is measured on
the modified test rows. A faster loss of accuracy indicates a ranking that removes more information.
Random and Sign-flip rankings are included as controls; Sign-flip has the magnitudes of HermAI-H by construction.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hct  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
NAMES = ["LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H"]
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
REPS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def explain_all(ctx, Xall, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return {k[5:]: z[k] for k in z.files}
    exps = hct.build_explainers(ctx, NAMES)
    A = {}
    for n, e in exps.items():
        t0 = time.time()
        A[n] = np.array([e.explain(Xall[i], 20000 + i)["attr"] for i in range(len(Xall))])
        log(f"  explained {len(Xall)} rows with {n} in {time.time() - t0:.0f}s")
    rng = np.random.default_rng(ctx.seed + 31)
    A["Random"] = rng.standard_normal(Xall.shape)
    A["SignFlip"] = -A["HermAI-H"]
    np.savez_compressed(cache, **{f"attr_{k}": v for k, v in A.items()})
    return A


def run(ds, mk, seed=0):
    t0 = time.time()
    X, y, cat = hct.load_dataset(ds, DATA, seed)
    ctx = hct.Context(X, y, cat, mk, seed, n_explain=100)
    Xall = np.vstack([ctx.Xtr, ctx.Xte])
    ntr = len(ctx.Xtr)
    A = explain_all(ctx, Xall, os.path.join(RES, f"roar_attr_{ds}_{mk}_s{seed}.npz"))
    rows = []
    for rep in range(REPS):
        m = hct.train_model(mk, ctx.Xtr, ctx.ytr, rep)
        rows.append(dict(explainer="none", frac=0.0, t=0, rep=rep,
                         acc=float(((m.predict_proba(ctx.Xte)[:, 1] >= 0.5).astype(int) == ctx.yte).mean())))
    idx = np.arange(len(Xall))[:, None]
    for frac in FRACS:
        t = max(1, int(round(frac * ctx.d)))
        for n, a in A.items():
            top = np.argsort(-np.abs(a), axis=1, kind="stable")[:, :t]
            Xm = Xall.copy()
            Xm[idx, top] = ctx.mu[top]
            for rep in range(REPS):
                m = hct.train_model(mk, Xm[:ntr], ctx.ytr, rep)
                acc = float(((m.predict_proba(Xm[ntr:])[:, 1] >= 0.5).astype(int) == ctx.yte).mean())
                rows.append(dict(explainer=n, frac=frac, t=t, rep=rep, acc=acc))
        log(f"  {ds}/{mk}: fraction {frac} done")
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"roar_{ds}_{mk}_s{seed}.csv"), index=False)
    log(f"ROAR {ds}/{mk}/s{seed} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)

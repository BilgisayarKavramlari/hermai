"""
Run-to-run consistency of each explainer (Co-12 'consistency'): every instance is explained
five times with different random states; we report the mean pairwise sign agreement@3 and the
Jaccard overlap of the top-3 feature sets between runs.

  python audit_consistency.py <dataset> <rf|mlp> [n_instances]
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hct  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")


def main(dataset, model_kind, n_inst=25, runs=5, seed=0):
    X, y, cat = hct.load_dataset(dataset, DATA, seed)
    ctx = hct.Context(X, y, cat, model_kind, seed, n_explain=100)
    inst = ctx.E[:n_inst]
    names = ["HermAI-H", "LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov"]
    exps = hct.build_explainers(ctx, names)
    A = {n: np.zeros((runs, n_inst, ctx.d)) for n in names}
    for run in range(runs):
        exps["LIME"].e = LimeTabularExplainer(ctx.Xtr, feature_names=ctx.names, categorical_features=ctx.cat_idx,
                                              mode="classification", discretize_continuous=True,
                                              random_state=1000 + run)
        exps["HermAI-H"].rng = np.random.default_rng(2000 + run)
        for r, ti in enumerate(inst):
            x = ctx.Xte[ti]
            for n in names:
                np.random.seed(3000 + 97 * run + r)          # KernelSHAP draws coalitions from the global RNG
                i_arg = 10000 * (run + 1) + r                # hermai 0.1.8 seeds from this index
                A[n][run, r] = exps[n].explain(x, i_arg)["attr"]
    np.savez_compressed(os.path.join(OUT, f"consistency_{dataset}_{model_kind}_arrays.npz"), **{n: A[n] for n in names})
    rows = []
    for n in names:
        sa, jac, sau, jacu = [], [], [], []
        for r in range(n_inst):
            for a, b in itertools.combinations(range(runs), 2):
                va, vb = A[n][a, r], A[n][b, r]
                sa.append(hct.sign_agreement(va, vb))
                ta, tb = set(hct.topk(va)), set(hct.topk(vb))
                jac.append(len(ta & tb) / len(ta | tb))
                # the same quantities over meaning units (sums of attributions within a unit), as used by PWA
                ga, gb = hct.group_sum(ctx, va), hct.group_sum(ctx, vb)
                sau.append(hct.sign_agreement(ga, gb))
                ua, ub = set(hct.topk(ga)), set(hct.topk(gb))
                jacu.append(len(ua & ub) / len(ua | ub))
        rows.append(dict(dataset=dataset, model=model_kind, explainer=n,
                         run_sign_agreement=float(np.mean(sa)), run_jaccard=float(np.mean(jac)),
                         run_sign_agreement_unit=float(np.mean(sau)), run_jaccard_unit=float(np.mean(jacu))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"consistency_{dataset}_{model_kind}.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25)

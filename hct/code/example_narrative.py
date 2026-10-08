"""
Finds one Pima Diabetes case (random forest, seed 0) that illustrates the reading problem of hermai 0.1.8
narratives and writes the sentences of both HermAI versions with their test outcomes to results/example.json.
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hct  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")


def main():
    X, y, cat = hct.load_dataset("pima", DATA, 0)
    ctx = hct.Context(X, y, cat, "rf", 0, n_explain=100)
    old = hct.HermaiV018Explainer(ctx, hct.HermaiPublicGen, "HermAI-0.1.8")
    rng = np.random.default_rng(5)
    for r, ti in enumerate(ctx.E):
        x = ctx.Xte[ti]
        eo = old.explain(x, r)
        w = eo["slope"]
        inv = [j for (j, s, _) in eo["claims"] if np.sign(w[j]) != np.sign(eo["attr"][j])]
        if not inv:
            continue
        new = hct.HermaiHExplainer(ctx)
        en = new.explain(x, r)
        rows_old, rows_new = [], []
        for (j, s, _) in eo["claims"]:
            lit = hct.test_slope(ctx, x, j, s)
            nat, _ = hct.test_contrib(ctx, x, j, s, "cond", rng, level="group")
            rows_old.append(dict(feature=ctx.names[j], value=float(x[j]), mean=float(ctx.mu[j]),
                                 sentence=f"Feature '{ctx.names[j]}' {'increases' if s > 0 else 'decreases'} the probability.",
                                 literal_pass=bool(lit), natural_pass=bool(nat)))
        for (j, s, _) in en["claims"]:
            nat, _ = hct.test_contrib(ctx, x, j, s, "cond", rng, level="group")
            rows_new.append(dict(feature=ctx.names[j], value=float(x[j]),
                                 sentence=f"The value of {ctx.names[j]} ({x[j]:g}) {'raises' if s > 0 else 'lowers'} the predicted probability.",
                                 natural_pass=bool(nat)))
        if sum(r_["natural_pass"] for r_ in rows_new) < 2:
            continue
        cf_old = eo["cf"]
        cf_new = en["cf"]
        out = dict(instance=int(ti), prob=float(ctx.f(x)[0]), decision=int(ctx.f(x)[0] >= 0.5),
                   old=rows_old, new=rows_new,
                   old_cf=None if cf_old is None else dict(zip(["valid", "on_horizon", "changed", "l1"], hct.test_cf(ctx, x, cf_old))),
                   new_cf=None if cf_new is None else dict(
                       zip(["valid", "on_horizon", "changed", "l1"], hct.test_cf(ctx, x, cf_new)),
                       changes={ctx.names[j]: [float(x[j]), float(cf_new[j])] for j in range(ctx.d) if abs(cf_new[j] - x[j]) > 1e-9}))
        json.dump(out, open(os.path.join(OUT, "example.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        return


if __name__ == "__main__":
    main()

"""
Runs every experiment reported in the paper.

  python run_experiments.py main <dataset> <rf|mlp> <seed> [n_explain]
  python run_experiments.py planted <seed>
  python run_experiments.py permuted <dataset> <seed>
  python run_experiments.py adversarial <seed>

Outputs go to ../results/.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hct  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(tag, df, readers, meta):
    df.to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
    with open(os.path.join(OUT, f"{tag}.json"), "w") as fh:
        json.dump(dict(meta=meta, readers=readers), fh, indent=1)


# --------------------------------------------------------------------------- main grid
def main_setting(dataset, model_kind, seed, n_explain=100):
    t0 = time.time()
    X, y, cat = hct.load_dataset(dataset, DATA, seed)
    ctx = hct.Context(X, y, cat, model_kind, seed, n_explain=n_explain)
    log(f"{dataset}/{model_kind}/s{seed}: acc={ctx.test_acc:.3f} whole_fid={ctx.whole_fidelity:.3f} "
        f"HP(real test)={ctx.hp_real_test:.3f} n_E={len(ctx.E)}")
    df, readers, meta, attrs = hct.run_setting(ctx, log=log)
    meta.update(dataset=dataset, model=model_kind, seed=seed, seconds=time.time() - t0,
                n_cat=len(cat), n_total=int(len(X)), pos_rate=float(np.mean(y)))
    save(f"main_{dataset}_{model_kind}_s{seed}", df, readers, meta)
    np.savez_compressed(os.path.join(OUT, f"main_{dataset}_{model_kind}_s{seed}_arrays.npz"),
                        E=ctx.E, **{f"attr_{k}": v for k, v in attrs.items()},
                        **{f"G_{s}": v for s, v in ctx._G.items()}, **{f"GA_{s}": v for s, v in ctx._GA.items()})
    log(f"done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- planted structure
def make_planted(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    d = 6
    C = np.eye(d)
    C[0, 1] = C[1, 0] = 0.6
    C[2, 5] = C[5, 2] = 0.4
    Xa = rng.multivariate_normal(np.zeros(d), C, size=n)
    region = (Xa[:, 3] > 0.5) & (Xa[:, 4] > 0.5)
    y = np.where(region, Xa[:, 2] > 0, Xa[:, 0] + Xa[:, 1] > 0).astype(int)
    flip = rng.random(n) < 0.02
    y[flip] = 1 - y[flip]
    X = pd.DataFrame(Xa, columns=[f"x{j}" for j in range(d)])
    return X, y


def planted(seed=0, n_explain=80):
    t0 = time.time()
    X, y = make_planted(seed=seed)
    ctx = hct.Context(X, y, [], "rf", seed, n_explain=n_explain)
    Xte = ctx.Xte
    inR = (Xte[:, 3] > 0.5) & (Xte[:, 4] > 0.5)
    rng = np.random.default_rng(seed + 3)
    idx_in, idx_out = np.where(inR)[0], np.where(~inR)[0]
    E = np.concatenate([rng.choice(idx_in, n_explain // 2, replace=False),
                        rng.choice(idx_out, n_explain - n_explain // 2, replace=False)])
    ctx.E = rng.permutation(E)
    rest = np.setdiff1d(np.arange(len(Xte)), ctx.E)
    ctx.R = rng.choice(rest, min(1000, len(rest)), replace=False)
    acc_in = float((((ctx.f_te >= 0.5).astype(int)) == ctx.yte)[inR].mean())
    acc_out = float((((ctx.f_te >= 0.5).astype(int)) == ctx.yte)[~inR].mean())
    whole_in = float(((ctx.whole.predict(ctx.Zte) >= 0.5) == (ctx.f_te >= 0.5))[inR].mean())
    whole_out = float(((ctx.whole.predict(ctx.Zte) >= 0.5) == (ctx.f_te >= 0.5))[~inR].mean())
    log(f"planted: acc={ctx.test_acc:.3f} (in R {acc_in:.3f}, out {acc_out:.3f}) whole fid in {whole_in:.3f} out {whole_out:.3f}")
    names = ["LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H", "Random", "SignFlip"]
    df, readers, meta, attrs = hct.run_setting(ctx, names, with_readers=False, log=log)
    df["in_region"] = [bool(inR[ctx.E[r]]) for r in df["inst"]]
    # reference: exact TreeSHAP attributions of the forest itself (faithful to the model by construction)
    import shap
    sv = shap.TreeExplainer(ctx.model).shap_values(ctx.Xte[ctx.E])
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    attrs["TreeSHAP-forest"] = sv
    G, GA = ctx._G, ctx._GA
    orow = []
    for r, ti in enumerate(ctx.E):
        pw = {s_: hct.sign_agreement(hct.group_sum(ctx, sv[r]), hct.group_sum(ctx, G[s_][r])) * float(GA[s_][r]) for s_ in hct.SCALES}
        hhr = next((s_ for s_ in hct.SCALES if pw[s_] >= hct.TAU_PWA - 1e-9), 0.0)
        orow.append(dict(explainer="TreeSHAP-forest", inst=r, PWA=pw[1.0], HHR=hhr, in_region=bool(inR[ti]),
                         **{f"PWA_{s_}": v for s_, v in pw.items()}))
    df = pd.concat([df, pd.DataFrame(orow)], ignore_index=True)
    top1 = []
    for r, ti in enumerate(ctx.E):
        truth = {2} if inR[ti] else {0, 1}
        mech = {2, 3, 4} if inR[ti] else {0, 1}
        for n in list(names) + ["TreeSHAP-forest"]:
            t1 = int(np.argmax(np.abs(attrs[n][r])))
            top1.append(dict(explainer=n, inst=r, in_region=bool(inR[ti]), top1=t1,
                             top1_correct=int(t1 in truth), top1_mechanism=int(t1 in mech),
                             agree_forest=int(t1 == int(np.argmax(np.abs(sv[r]))))))
    pd.DataFrame(top1).to_csv(os.path.join(OUT, f"planted_top1_s{seed}.csv"), index=False)
    meta.update(acc_in=acc_in, acc_out=acc_out, whole_fid_in=whole_in, whole_fid_out=whole_out,
                share_region=float(inR.mean()), seconds=time.time() - t0)
    save(f"planted_s{seed}", df, readers, meta)
    log(f"planted done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- label permutation
def permuted(dataset, seed=0, n_explain=80):
    t0 = time.time()
    X, y, cat = hct.load_dataset(dataset, DATA, seed)
    ctx = hct.Context(X, y, cat, "rf", seed, n_explain=n_explain, permute_labels=True)
    log(f"permuted {dataset}: acc vs true labels={ctx.test_acc:.3f} whole_fid={ctx.whole_fidelity:.3f}")
    names = ["LIME", "KernelSHAP", "HermAI-H", "Random"]
    df, readers, meta, _ = hct.run_setting(ctx, names, with_readers=True, log=log)
    meta.update(dataset=dataset, model="rf", seed=seed, permuted=True, seconds=time.time() - t0)
    save(f"permuted_{dataset}_s{seed}", df, readers, meta)
    log(f"permuted done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- adversarial scaffolding
class ConstantModel:
    def predict_proba(self, X):
        X = np.atleast_2d(X)
        return np.c_[np.full(len(X), 0.5), np.full(len(X), 0.5)]


class ScaffoldModel:
    """Slack et al. (2020)-style scaffold: biased on the data manifold, innocuous off it."""
    def __init__(self, mu, sd, detector, j_age, j_unrel, thr=0.5):
        self.mu, self.sd, self.det, self.ja, self.ju, self.thr = mu, sd, detector, j_age, j_unrel, thr

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        real = self.det.predict_proba((X - self.mu) / self.sd)[:, 1] >= self.thr
        pb = 1.0 / (1.0 + np.exp((X[:, self.ja] - 30.0) / 3.0))        # depends on age only
        pi = 1.0 / (1.0 + np.exp(-(X[:, self.ju] - 0.5) * 12.0))      # depends on an unrelated column
        p = np.where(real, pb, pi)
        return np.c_[1 - p, p]


def adversarial(seed=0, n_explain=60):
    from sklearn.ensemble import RandomForestClassifier
    from lime.lime_tabular import LimeTabularExplainer
    import shap
    t0 = time.time()
    X, y, cat = hct.load_dataset("compas", DATA, seed)
    rng = np.random.default_rng(seed + 21)
    X = X.copy()
    X["unrelated"] = rng.random(len(X))
    ctx = hct.Context(X, y, cat, "rf", seed, n_explain=n_explain, model=ConstantModel())
    ja, ju = ctx.names.index("age"), ctx.names.index("unrelated")
    # the attacker's detector is trained against LIME and KernelSHAP evidence points
    lime_e = LimeTabularExplainer(ctx.Xtr, feature_names=ctx.names, categorical_features=ctx.cat_idx,
                                  mode="classification", discretize_continuous=True, random_state=seed)
    rows = ctx.Xfit[rng.choice(len(ctx.Xfit), 300, replace=False)]
    neg = [lime_e._LimeTabularExplainer__data_inverse(r, 21)[1][1:] for r in rows]
    bg = shap.kmeans(ctx.Xfit, 10)
    B, w = bg.data, np.asarray(bg.weights, float) / np.sum(bg.weights)
    for r in rows:
        S = rng.random((20, ctx.d)) < 0.5
        neg.append(np.where(S, r[None, :], B[rng.choice(len(B), 20, p=w)]))
    Xneg = np.vstack(neg)
    Zpos, Zneg = ctx.Zfit, ctx.z(Xneg)
    det = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1)
    det.fit(np.vstack([Zpos, Zneg]), np.r_[np.ones(len(Zpos)), np.zeros(len(Zneg))])
    thr = float(np.quantile(det.predict_proba(ctx.Zcal)[:, 1], 0.05))
    det_real_test = float((det.predict_proba(ctx.Zte)[:, 1] >= thr).mean())
    det_neg_caught = float((det.predict_proba(Zneg)[:, 1] < thr).mean())
    ctx.set_model(ScaffoldModel(ctx.mu, ctx.sd, det, ja, ju, thr))
    log(f"adversarial: threshold {thr:.3f}; detector accepts {det_real_test:.3f} of real test rows and rejects {det_neg_caught:.3f} of its training perturbations")
    names = ["LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H"]
    df, readers, meta, attrs = hct.run_setting(ctx, names, with_readers=False, log=log)
    # how much of each explainer's evidence the scaffold treats as real
    exps = hct.build_explainers(ctx, names)
    acc = []
    for n, e in exps.items():
        fr = []
        for r, ti in enumerate(ctx.E[:20]):
            ex = e.explain(ctx.Xte[ti], 1000 + r)
            fr.append(float((det.predict_proba(ctx.z(ex["samples"]))[:, 1] >= thr).mean()))
        acc.append(dict(explainer=n, evidence_seen_as_real=float(np.mean(fr))))
    top = []
    for n in names:
        A = attrs[n]
        t1 = np.argmax(np.abs(A), axis=1)
        top.append(dict(explainer=n, age_top1=float(np.mean(t1 == ja)), unrelated_top1=float(np.mean(t1 == ju)),
                        unrelated_in_top3=float(np.mean([ju in hct.topk(a) for a in A]))))
    pd.DataFrame(top).merge(pd.DataFrame(acc), on="explainer").to_csv(os.path.join(OUT, f"adversarial_top_s{seed}.csv"), index=False)
    meta.update(det_real_test=det_real_test, det_threshold=thr, det_train_neg_caught=det_neg_caught, j_age=ja, j_unrel=ju,
                seconds=time.time() - t0)
    save(f"adversarial_s{seed}", df, readers, meta)
    log(f"adversarial done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "main":
        n = int(sys.argv[5]) if len(sys.argv) > 5 else 100
        main_setting(sys.argv[2], sys.argv[3], int(sys.argv[4]), n)
    elif mode == "planted":
        planted(int(sys.argv[2]))
    elif mode == "permuted":
        permuted(sys.argv[2], int(sys.argv[3]))
    elif mode == "adversarial":
        adversarial(int(sys.argv[2]))
    else:
        raise SystemExit(__doc__)

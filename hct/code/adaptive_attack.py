"""
Adaptive scaffold attacks on the COMPAS data (added in the revision).

  python adaptive_attack.py <A0|A1|A2|A3> [seed]

The scaffold of Slack et al. (2020) returns a prediction that depends only on age for inputs its detector accepts as
real and only on a uniformly distributed decoy column otherwise. The attacks differ in what the detector is trained
to reject.
  A0  the attack of the original submission: evidence points of LIME and KernelSHAP.
  A1  horizon-aware attacker: A0, plus perturbations generated strictly inside the horizon H_alpha, namely LIME and
      KernelSHAP points snapped to the measurement scale and filtered by the horizon test, and HermAI-H samples.
  A2  test-aware attacker: A1, plus the intervention points of the HCT claim test (one feature replaced by the value
      of one of its 30 nearest neighbours on the other features) and of its marginal variant.
  A3  memorising attacker: the scaffold is biased exactly on the stored records of the dataset and innocuous at every
      other point. This is the limiting case of an attacker who can tell any synthetic point from a record.
For every attack the full HCT profile (including machine readers) is computed, the established metrics of
baselines.py are evaluated on the same explanations, and two oracle explanations are scored: the truth
(attribution on age only) and the decoy (attribution on the decoy column only).
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hct  # noqa: E402
import baselines as bl  # noqa: E402
from run_experiments import ScaffoldModel, ConstantModel  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
NAMES = ["LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ExactMatchDetector:
    """Accepts a point as real only if it coincides with a stored record (A3)."""
    def __init__(self, Zrec):
        self.nn = NearestNeighbors(n_neighbors=1).fit(Zrec)

    def predict_proba(self, Z):
        dist = self.nn.kneighbors(np.atleast_2d(Z), n_neighbors=1)[0][:, 0]
        r = (dist < 1e-6).astype(float)
        return np.c_[1 - r, r]


def cvr_hybrids(ctx, rows, rng, n_cond=20, n_marg=10):
    """Intervention points of the HCT claim test (conditional) and of its marginal variant."""
    out_c, out_m = [], []
    Zr = ctx.z(rows)
    for x, z in zip(rows, Zr):
        for _ in range(n_cond):
            j = int(rng.integers(ctx.d))
            keep, nn = ctx.nn_minus[j]
            idx = nn.kneighbors(z[keep].reshape(1, -1), n_neighbors=hct.K_COND + 1)[1][0]
            idx = [i for i in idx if not np.allclose(ctx.Xfit[i], x)][:hct.K_COND]
            h = x.copy()
            h[j] = ctx.Xfit[rng.choice(idx), j]
            out_c.append(h)
        for _ in range(n_marg):
            j = int(rng.integers(ctx.d))
            h = x.copy()
            h[j] = ctx.Xfit[rng.integers(len(ctx.Xfit)), j]
            out_m.append(h)
    return np.array(out_c), np.array(out_m)


def claim_stats(df, ja, ju):
    res = {}
    for n, g in df.groupby("explainer"):
        rec = []
        for cd in g.claim_detail.dropna():
            for it in str(cd).split(";"):
                if it:
                    j, sg, ok = it.split(":")
                    rec.append((int(j), int(ok)))
        rec = np.array(rec)
        res[n] = dict(decoy_claims=int((rec[:, 0] == ju).sum()),
                      decoy_pass=float(rec[rec[:, 0] == ju, 1].mean()) if (rec[:, 0] == ju).any() else np.nan,
                      age_claims=int((rec[:, 0] == ja).sum()),
                      age_pass=float(rec[rec[:, 0] == ja, 1].mean()) if (rec[:, 0] == ja).any() else np.nan)
    return res


def run(attack, seed=0, n_explain=60):
    t0 = time.time()
    from lime.lime_tabular import LimeTabularExplainer
    import shap
    X, y, cat = hct.load_dataset("compas", DATA, seed)
    rng = np.random.default_rng(seed + 21)
    X = X.copy()
    X["unrelated"] = rng.random(len(X))
    ctx = hct.Context(X, y, cat, "rf", seed, n_explain=n_explain, model=ConstantModel())
    ja, ju = ctx.names.index("age"), ctx.names.index("unrelated")
    # --- negatives of A0, drawn exactly as in the original submission
    lime_e = LimeTabularExplainer(ctx.Xtr, feature_names=ctx.names, categorical_features=ctx.cat_idx,
                                  mode="classification", discretize_continuous=True, random_state=seed)
    rows = ctx.Xfit[rng.choice(len(ctx.Xfit), 300, replace=False)]
    neg_lime = [lime_e._LimeTabularExplainer__data_inverse(r, 21)[1][1:] for r in rows]
    bg = shap.kmeans(ctx.Xfit, 10)
    B, w = bg.data, np.asarray(bg.weights, float) / np.sum(bg.weights)
    neg_ks = []
    for r in rows:
        S = rng.random((20, ctx.d)) < 0.5
        neg_ks.append(np.where(S, r[None, :], B[rng.choice(len(B), 20, p=w)]))
    neg = neg_lime + neg_ks
    arng = np.random.default_rng(seed + 977)          # the attacker's own stream (the auditor's seeds are unknown)
    extra = {}
    if attack in ("A1", "A2"):
        P = np.vstack(neg_lime + neg_ks)
        Psnap = ctx.x_from_z(ctx.z(P))                 # snap to the measurement scale ...
        Psnap = Psnap[ctx.on_horizon(ctx.z(Psnap))]     # ... and keep only points inside H_alpha
        hh = hct.HermaiHExplainer(ctx)
        hh.rng = np.random.default_rng(seed + 978)
        Hs = np.vstack([ctx.x_from_z(hh.sample(r, n=200))[arng.choice(200, 20, replace=False)] for r in rows])
        extra["horizon_snapped"] = Psnap
        extra["hermai_h_samples"] = Hs
        neg += [Psnap, Hs]
    if attack == "A2":
        hc, hm = cvr_hybrids(ctx, rows, arng)
        extra["cvr_cond_hybrids"] = hc
        extra["cvr_marg_hybrids"] = hm
        neg += [hc, hm]
    Xneg = np.vstack(neg)
    if attack == "A3":
        det = ExactMatchDetector(ctx.z(X.values.astype(float)))
        thr = 0.5
        det_neg_caught = float((det.predict_proba(ctx.z(Xneg))[:, 1] < thr).mean())
    else:
        Zpos, Zneg = ctx.Zfit, ctx.z(Xneg)
        det = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1)
        det.fit(np.vstack([Zpos, Zneg]), np.r_[np.ones(len(Zpos)), np.zeros(len(Zneg))])
        thr = float(np.quantile(det.predict_proba(ctx.Zcal)[:, 1], 0.05))
        det_neg_caught = float((det.predict_proba(Zneg)[:, 1] < thr).mean())
    det_real_test = float((det.predict_proba(ctx.Zte)[:, 1] >= thr).mean())
    ctx.set_model(ScaffoldModel(ctx.mu, ctx.sd, det, ja, ju, thr))
    log(f"{attack}: n_neg={len(Xneg)} threshold={thr:.3f} real test accepted={det_real_test:.3f} "
        f"training negatives rejected={det_neg_caught:.3f}")

    # --- what the detector accepts, for held-out points of every type (20 explained instances)
    probe_rng = np.random.default_rng(seed + 555)
    test_rows = ctx.Xte[ctx.E[:20]]
    exps = hct.build_explainers(ctx, NAMES)
    acc_types = {}
    for n, e in exps.items():
        fr = [float((det.predict_proba(ctx.z(e.explain(ctx.Xte[ti], 1000 + r)["samples"]))[:, 1] >= thr).mean())
              for r, ti in enumerate(ctx.E[:20])]
        acc_types[f"evidence_{n}"] = float(np.mean(fr))
    hc, hm = cvr_hybrids(ctx, test_rows, probe_rng)
    acc_types["cvr_cond_hybrids"] = float((det.predict_proba(ctx.z(hc))[:, 1] >= thr).mean())
    acc_types["cvr_marg_hybrids"] = float((det.predict_proba(ctx.z(hm))[:, 1] >= thr).mean())
    meanpts = []
    for x in test_rows:
        for _ in range(10):
            S = probe_rng.choice(ctx.d, 2, replace=False)
            xm = x.copy()
            xm[S] = ctx.mu[S]
            meanpts.append(xm)
    acc_types["mean_baseline_points"] = float((det.predict_proba(ctx.z(np.array(meanpts)))[:, 1] >= thr).mean())
    # detector AUC: held-out real test rows against held-out conditional hybrids
    s_real = det.predict_proba(ctx.Zte[ctx.R[:600]])[:, 1]
    s_hyb = det.predict_proba(ctx.z(hc))[:, 1]
    auc_hyb = float(roc_auc_score(np.r_[np.ones(len(s_real)), np.zeros(len(s_hyb))], np.r_[s_real, s_hyb]))
    log(f"{attack}: acceptance {json.dumps({k: round(v, 3) for k, v in acc_types.items()})}; AUC real vs hybrids {auc_hyb:.3f}")

    # --- HCT on the five explainers, with machine readers
    df, readers, meta, attrs = hct.run_setting(ctx, NAMES, with_readers=True, log=log)
    cs = claim_stats(df, ja, ju)
    whole_top1 = np.argmax(np.abs(ctx._G[1.0]), axis=1)
    top = {}
    for n in NAMES:
        A = attrs[n]
        t1 = np.argmax(np.abs(A), axis=1)
        top[n] = dict(age_top1=float(np.mean(t1 == ja)), unrelated_top1=float(np.mean(t1 == ju)),
                      unrelated_in_top3=float(np.mean([ju in hct.topk(a) for a in A])),
                      top1_agrees_with_whole=float(np.mean(t1 == whole_top1)))
    # --- oracle explanations and established metrics on the same explanations
    brng = np.random.default_rng(seed + 7000)
    orng = np.random.default_rng(seed + 5)
    trng = np.random.default_rng(seed + 6)          # breaks ties among the zero attributions of the oracles at random
    brows, orows = [], []
    for r, ti in enumerate(ctx.E):
        x = ctx.Xte[ti]
        xa = x.copy()
        xa[ja] = ctx.mu[ja]
        xu = x.copy()
        xu[ju] = 0.5
        p = ctx.f(x)[0]
        tie = 1e-9 * trng.standard_normal(ctx.d)
        truth = tie.copy()
        truth[ja] = 1.0 / (1.0 + np.exp((x[ja] - 30.0) / 3.0)) - 1.0 / (1.0 + np.exp((ctx.mu[ja] - 30.0) / 3.0))
        decoy = tie.copy()
        decoy[ju] = 1.0 / (1.0 + np.exp(-(x[ju] - 0.5) * 12.0)) - 0.5
        truth[ja] = truth[ja] if abs(truth[ja]) > 1e-6 else 1e-6
        decoy[ju] = decoy[ju] if abs(decoy[ju]) > 1e-6 else 1e-6
        Ar = {n: attrs[n][r] for n in NAMES}
        Ar["Oracle-truth"], Ar["Oracle-decoy"] = truth, decoy
        m = bl.instance_metrics(ctx, x, Ar, brng)
        for n in Ar:
            brows.append(dict(explainer=n, inst=r, **m[n]))
        for n, a, j in [("Oracle-truth", truth, ja), ("Oracle-decoy", decoy, ju)]:
            ok, _ = hct.test_contrib(ctx, x, j, float(np.sign(a[j])), "cond", orng, level="group")
            okm, _ = hct.test_contrib(ctx, x, j, float(np.sign(a[j])), "marg", orng)
            orows.append(dict(explainer=n, inst=r, CVR_dir=float(ok), CVR_dir_marg=float(okm)))
    bdf = pd.DataFrame(brows)
    odf = pd.DataFrame(orows)
    tag = f"adaptive_{attack}_s{seed}"
    df.to_csv(os.path.join(RES, f"{tag}.csv"), index=False)
    bdf.to_csv(os.path.join(RES, f"{tag}_baselines.csv"), index=False)
    odf.to_csv(os.path.join(RES, f"{tag}_oracles.csv"), index=False)
    np.savez_compressed(os.path.join(RES, f"{tag}_arrays.npz"), E=ctx.E, **{f"attr_{k}": v for k, v in attrs.items()},
                        G=ctx._G[1.0])
    out = dict(attack=attack, seed=seed, n_neg=int(len(Xneg)), threshold=thr, det_real_test=det_real_test,
               det_train_neg_caught=det_neg_caught, acceptance=acc_types, auc_real_vs_hybrids=auc_hyb,
               claims=cs, top=top, readers=readers, whole_fidelity=ctx.whole_fidelity,
               profile=df.groupby("explainer")[["HP", "CVR_dir", "CVR_dir_marg", "PWA", "HHR", "cos_whole"]].mean().to_dict(),
               baselines=bdf.groupby("explainer").mean(numeric_only=True).drop(columns="inst").to_dict(),
               oracles=odf.groupby("explainer")[["CVR_dir", "CVR_dir_marg"]].mean().to_dict(),
               extra_negatives={k: int(len(v)) for k, v in extra.items()}, seconds=time.time() - t0)
    with open(os.path.join(RES, f"{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log(f"{attack} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)

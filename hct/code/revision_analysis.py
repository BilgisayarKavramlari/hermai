"""
Aggregates the experiments added in the revision into results/revision_summary.json and draws the new figures.
  python revision_analysis.py
Inputs: baselines_*.csv, disagree_*.csv, roar_*.csv, adaptive_A*_s0.json/csv, planted_s0.csv and the main study.
"""
import os
import json
import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
SET =[(d, m) for d in ["breast", "pima", "compas", "adult"] for m in ["rf", "mlp"]]
EX = ["LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H", "Random", "SignFlip"]
MAIN = EX[:5]
BM = ["PGI_marg", "PGI_cond", "DEL_marg", "DEL_cond", "FC_mean", "FC_cond"]
AGR = ["FA", "RA", "SA", "SRA", "RC"]
EMPTY = {}   # dataset -> explainer -> number of all-zero explanations (both models)


def r4(v):
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else round(v, 4)


def msd(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return dict(mean=r4(np.mean(vals)), sd=r4(np.std(vals, ddof=1)) if len(vals) > 1 else None, n=len(vals)) if vals else None


def load_inst():
    S = json.load(open(os.path.join(RES, "summary.json")))
    rows, inst = [], {}
    for ds, mk in SET:
        b = pd.read_csv(os.path.join(RES, f"baselines_{ds}_{mk}_s0.csv"))
        h = pd.read_csv(os.path.join(RES, f"main_{ds}_{mk}_s0.csv"))
        h["GI"] = (h.HHR == 1.0).astype(float)
        m = b.merge(h[["explainer", "inst", "HP", "CVR_dir", "PWA", "GI", "HHR"]], on=["explainer", "inst"])
        # explanations that attribute nothing (all-zero vectors) make no ranking: excluded from ranking-based metrics,
        # as they are excluded from CVR, which they give no claims to test
        Z = np.load(os.path.join(RES, f"main_{ds}_{mk}_s0_arrays.npz"))
        empty = {n: np.abs(Z[f"attr_{n}"]).max(axis=1) == 0 for n in EX}
        m["is_empty"] = [bool(empty[n][r]) for n, r in zip(m.explainer, m.inst)]
        for c in BM + [c for c in m.columns if c.startswith("hor_") and c not in ("hor_FC_mean", "hor_FC_cond")]:
            m.loc[m["is_empty"], c] = np.nan
        EMPTY.setdefault(ds, {})
        for n in EX:
            EMPTY[ds][n] = EMPTY[ds].get(n, 0) + int(empty[n].sum())
        inst[(ds, mk)] = m
        g = m.groupby("explainer").mean(numeric_only=True)
        for n in EX:
            r = dict(ds=ds, mk=mk, explainer=n, **g.loc[n].to_dict())
            ps = S["per_setting"].get(f"{ds}|{mk}|{n}", {})
            r["HG"] = ps.get("HG_mean", np.nan)
            rows.append(r)
    return S, pd.DataFrame(rows), inst


def comparison(S, D, inst):
    out = {}
    out["profile"] = {n: {k: msd(D[D.explainer == n][k].tolist()) for k in BM + ["HP", "CVR_dir", "PWA", "GI", "HG"]} for n in EX}
    hor = [c for c in D.columns if c.startswith("hor_")]
    out["horizon_of_eval_points"] = {ds: {c: r4(D[D.ds == ds][c].mean()) for c in hor} for ds in ["breast", "pima", "compas", "adult"]}
    out["horizon_of_eval_points_all"] = {c: r4(D[c].mean()) for c in hor}
    # Kendall tau between metric rankings of the five main explainers, mean over settings
    mets = BM + ["CVR_dir", "PWA", "GI", "HG"]
    tau = {}
    for a, b in itertools.combinations(mets, 2):
        t = []
        for (ds, mk), g in D.groupby(["ds", "mk"]):
            g = g.set_index("explainer").loc[MAIN]
            t.append(kendalltau(g[a], g[b]).correlation)
        tau[f"{a}|{b}"] = r4(np.nanmean(t))
    out["kendall_main"] = tau
    # instance-level Spearman between HCT components and each baseline (within explainer and setting)
    sp = {}
    for hm in ["CVR_dir", "PWA"]:
        for bm in BM:
            vals = []
            for m in inst.values():
                for n in MAIN:
                    g = m[m.explainer == n][[bm, hm]].dropna()
                    vals.append(spearmanr(g[bm], g[hm]).correlation)
            sp[f"{hm}|{bm}"] = msd(vals)
    out["spearman_instance"] = sp
    # sign-flip control against HermAI-H
    sf = {}
    for k in BM + ["CVR_dir", "PWA", "HP", "HG"]:
        a = D[D.explainer == "HermAI-H"][k].values
        b = D[D.explainer == "SignFlip"][k].values
        sf[k] = dict(hermai_h=r4(np.nanmean(a)), signflip=r4(np.nanmean(b)), max_abs_diff=r4(np.nanmax(np.abs(a - b))))
    out["signflip_vs_hermaih"] = sf
    # narrative inversion: HermAI 0.1.8 against LIME on vector metrics and on claims
    P = S["profile"]
    out["inversion_case"] = {n: dict(FC_mean=out["profile"][n]["FC_mean"]["mean"], DEL_marg=out["profile"][n]["DEL_marg"]["mean"],
                                     PGI_marg=out["profile"][n]["PGI_marg"]["mean"], CVR_dir=P[n]["CVR_dir"]["mean"],
                                     inversion=P[n].get("inversion", {}).get("mean"))
                             for n in ["LIME", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H", "KernelSHAP"]}
    # main explainers that score highest / lowest on each removal-based metric (mean over settings)
    out["extremes"] = {k: dict(highest=max(MAIN, key=lambda n: out["profile"][n][k]["mean"]),
                               lowest=min(MAIN, key=lambda n: out["profile"][n][k]["mean"])) for k in BM + ["CVR_dir", "PWA"]}
    out["empty_explanations"] = {ds: dict(v) for ds, v in EMPTY.items()}
    out["empty_explanations_total"] = {n: int(sum(EMPTY[ds][n] for ds in EMPTY)) for n in EX}
    return out


def disagreement():
    allD, adj = [], []
    for ds, mk in SET:
        d = pd.read_csv(os.path.join(RES, f"disagree_{ds}_{mk}_s0.csv"))
        d["ds"], d["mk"] = ds, mk
        h = pd.read_csv(os.path.join(RES, f"main_{ds}_{mk}_s0.csv")).set_index(["explainer", "inst"])
        Z = np.load(os.path.join(RES, f"main_{ds}_{mk}_s0_arrays.npz"))
        empty = {n: np.abs(Z[f"attr_{n}"]).max(axis=1) == 0 for n in EX}
        keep = [not (empty[e1][r] or (e2 in empty and empty[e2][r])) for e1, e2, r in zip(d.e1, d.e2, d.inst)]
        allD.append(d[keep])
        for e1, e2 in itertools.combinations(MAIN, 2):
            A1, A2 = Z[f"attr_{e1}"], Z[f"attr_{e2}"]
            for r in range(len(A1)):
                if empty[e1][r] or empty[e2][r]:
                    continue
                adj.append(dict(e1=e1, e2=e2, top1_differ=int(np.argmax(np.abs(A1[r])) != np.argmax(np.abs(A2[r]))),
                                c1=h.loc[(e1, r), "CVR_dir"], c2=h.loc[(e2, r), "CVR_dir"]))
    D = pd.concat(allD)
    mp = D[D.e1.isin(MAIN) & D.e2.isin(MAIN)]
    out = dict(main_pairs={k: r4(mp[k].mean()) for k in AGR},
               main_pairs_by_setting_sd={k: r4(mp.groupby(["ds", "mk"])[k].mean().std(ddof=1)) for k in AGR},
               vs_whole={n: {k: r4(D[(D.e1 == n) & (D.e2 == "Whole")][k].mean()) for k in AGR} for n in EX},
               signflip_vs_hermaih={k: r4(D[(D.e1 == "HermAI-H") & (D.e2 == "SignFlip")][k].mean()) for k in AGR})
    A = pd.DataFrame(adj)
    A["sep"] = (np.abs(A.c1 - A.c2) >= 1 / 3 - 1e-9).astype(int)
    sub = A[A.top1_differ == 1]
    lk = sub[(sub.e1 == "LIME") & (sub.e2 == "KernelSHAP")]
    out["adjudication"] = dict(share_top1_differ=r4(A.top1_differ.mean()), n_differ=int(len(sub)),
                               cvr_separates=r4(sub.sep.mean()),
                               lime_vs_kernelshap=dict(n=int(len(lk)), kernelshap_more=r4(((lk.c2 - lk.c1) >= 1 / 3 - 1e-9).mean()),
                                                       lime_more=r4(((lk.c1 - lk.c2) >= 1 / 3 - 1e-9).mean())))
    return out


def roar():
    out, curves = {}, {}
    for ds, mk in [("pima", "rf"), ("pima", "mlp"), ("breast", "rf"), ("breast", "mlp")]:
        f = os.path.join(RES, f"roar_{ds}_{mk}_s0.csv")
        if not os.path.exists(f):
            continue
        r = pd.read_csv(f)
        base = r[r.explainer == "none"].acc.mean()
        g = r[r.explainer != "none"].groupby(["explainer", "frac"]).acc.mean().unstack()
        curves[f"{ds}|{mk}"] = dict(base=r4(base), fracs=[float(c) for c in g.columns],
                                    acc={n: [r4(v) for v in g.loc[n].values] for n in g.index})
        out[f"{ds}|{mk}"] = {n: dict(mean_acc=r4(g.loc[n].mean()), acc50=r4(g.loc[n, 0.5]),
                                     drop50=r4(base - g.loc[n, 0.5])) for n in g.index}
        out[f"{ds}|{mk}"]["_base"] = r4(base)
    summ = {}
    if out:
        for n in EX:
            vals = [v[n]["mean_acc"] for v in out.values() if n in v]
            drops = [v[n]["drop50"] for v in out.values() if n in v]
            summ[n] = dict(mean_acc=msd(vals), drop50=msd(drops))
        summ["_identical_signflip"] = bool(all(v["SignFlip"]["mean_acc"] == v["HermAI-H"]["mean_acc"] for v in out.values()))
        summ["_settings"] = list(out.keys())
    return dict(per_setting=out, summary=summ, curves=curves)


def planted():
    p = pd.read_csv(os.path.join(RES, "planted_s0.csv"))
    out = {}
    for n in ["LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H", "Random", "SignFlip"]:
        q = p[(p.explainer == n) & p.CVR_dir.notna()].copy()
        lo, hi = q.HHR < 1.0, q.CVR_dir >= 2 / 3 - 1e-9
        y = q.in_region.astype(int).values
        auc_h = roc_auc_score(y, 1 - q.HHR.values)
        auc_hc = roc_auc_score(y, (1 - q.HHR.values) * q.CVR_dir.values)
        out[n] = dict(lowH_hiC_in=r4((lo & hi)[q.in_region].mean()), lowH_hiC_out=r4((lo & hi)[~q.in_region].mean()),
                      lowH_loC_in=r4((lo & ~hi)[q.in_region].mean()), lowH_loC_out=r4((lo & ~hi)[~q.in_region].mean()),
                      cvr_lowH_in=r4(q[lo & q.in_region].CVR_dir.mean()), cvr_lowH_out=r4(q[lo & ~q.in_region].CVR_dir.mean()),
                      n_lowH_in=int((lo & q.in_region).sum()), n_lowH_out=int((lo & ~q.in_region).sum()),
                      auc_hhr=r4(auc_h), auc_hhr_cvr=r4(auc_hc))
    return out


def adaptive():
    out = {}
    for a in ["A0", "A1", "A2", "A3"]:
        f = os.path.join(RES, f"adaptive_{a}_s0.json")
        if not os.path.exists(f):
            continue
        J = json.load(open(f))
        hg = {n: r4(np.mean([v["HG"] for v in J["readers"][n].values()])) for n in J["readers"]}
        hgmin = {n: r4(np.min([v["HG"] for v in J["readers"][n].values()])) for n in J["readers"]}
        out[a] = dict(det_real_test=r4(J["det_real_test"]), det_train_neg_caught=r4(J["det_train_neg_caught"]),
                      n_neg=J["n_neg"], threshold=r4(J["threshold"]), acceptance={k: r4(v) for k, v in J["acceptance"].items()},
                      auc_real_vs_hybrids=r4(J["auc_real_vs_hybrids"]),
                      claims={n: {k: (r4(v) if isinstance(v, float) else v) for k, v in c.items()} for n, c in J["claims"].items()},
                      top={n: {k: r4(v) for k, v in t.items()} for n, t in J["top"].items()},
                      profile={k: {n: r4(v) for n, v in d.items()} for k, d in J["profile"].items()},
                      HG=hg, HG_min=hgmin,
                      baselines={k: {n: r4(v) for n, v in d.items()} for k, d in J["baselines"].items()},
                      oracles={k: {n: r4(v) for n, v in d.items()} for k, d in J["oracles"].items()})
    return out


# ----------------------------------------------------------------------------- figures
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"HermAI-H": "#2a78d6", "LIME": "#eb6834", "KernelSHAP": "#1baf7a", "HermAI-0.1.8": "#eda100",
         "HermAI-0.1.8-cov": "#e87ba4", "Random": "#9a9893", "SignFlip": "#52514e"}
LABEL = {"HermAI-H": "HermAI-H", "LIME": "LIME", "KernelSHAP": "KernelSHAP", "HermAI-0.1.8": "HermAI 0.1.8",
         "HermAI-0.1.8-cov": "HermAI 0.1.8 (cov)", "Random": "Random (control)", "SignFlip": "Sign-flip (control)"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "figure.dpi": 110})


def fig_framework():
    """Figure 1: (a) where HCT sits in the life cycle of an explanation; (b) structure of the test."""
    fig = plt.figure(figsize=(7.2, 7.4))
    axa = fig.add_axes([0.0, 0.60, 1.0, 0.40])
    axb = fig.add_axes([0.0, 0.0, 1.0, 0.56])
    for ax in (axa, axb):
        ax.axis("off")

    def box(ax, x, y, w, h, title, body, fc, fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=fs + 1.0, weight="bold", color=INK)
        if body:
            ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.3)

    def arrow(ax, p, q, style="-|>", ls="-"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=INK2, lw=0.9, linestyle=ls,
                                                         shrinkA=0, shrinkB=0))

    ax = axa
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.text(0.05, 4.55, "(a) Where the test sits in the life cycle of an explanation", fontsize=8.8, va="top", color=INK)
    box(ax, 0.05, 2.25, 1.55, 1.25, "Data and model", "training data,\nclassifier f", "#f0efec")
    box(ax, 1.95, 2.25, 1.75, 1.25, "Explainer", "method, sampler,\nsettings, seed", "#f0efec")
    box(ax, 4.05, 2.25, 1.85, 1.25, "Explanation", "attributions, sentences,\nevidence points,\ncounterfactual", "#f0efec")
    box(ax, 6.25, 2.1, 1.6, 1.55, "HCT screen", "no human raters;\nquery access to f\nand its training data", "#e6f0fb")
    box(ax, 8.2, 2.25, 1.75, 1.25, "People", "user study,\ndeployment,\naudit", "#fdf1e9")
    for x0, x1 in [(1.6, 1.95), (3.7, 4.05), (5.9, 6.25), (7.85, 8.2)]:
        arrow(ax, (x0, 2.88), (x1, 2.88))
    # repair loop above the row: from the screen back to the explainer
    ax.annotate("", xy=(2.83, 3.5), xytext=(7.05, 3.65), arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=1.0,
                connectionstyle="arc3,rad=0.12", linestyle="--", shrinkA=0, shrinkB=0))
    ax.text(5.3, 4.12, "repair loop: change the sampler, the wording or the configuration and test again",
            fontsize=6.6, color="#2a78d6", ha="center", va="center")
    outs = [("Per explanation", "verdict for each sentence;\nhorizon and coherence scale\nattached to what people see"),
            ("Per case", "cases that cohere only\nlocally or nowhere are\nrouted to closer reading"),
            ("Per explainer", "profile across cases:\nchoose or repair sampler,\nwording, configuration"),
            ("Per study", "which explanations deserve\na study, and which predictions\nthe study should test")]
    xs = [0.05, 2.55, 5.05, 7.55]
    bus = 1.72
    ax.plot([7.05, 7.05], [2.1, bus], color=INK2, lw=0.9)
    ax.plot([xs[0] + 1.17, xs[-1] + 1.17], [bus, bus], color=INK2, lw=0.9)
    for (t, b), x in zip(outs, xs):
        box(ax, x, 0.05, 2.35, 1.35, t, b, "#ffffff", fs=6.6)
        arrow(ax, (x + 1.17, bus), (x + 1.17, 1.42))

    ax = axb
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.3)
    ax.text(0.05, 6.25, "(b) Structure of the test", fontsize=8.8, va="top", color=INK)
    box(ax, 0.1, 2.15, 1.95, 1.7, "Explanation", "attributions\nnarrative claims\ncounterfactual\nevidence points", "#f0efec")
    mids = [("Interpreter 1: the data", "Horizon Plausibility (HP): evidence points\ninside the conformal, scale-aware horizon"),
            ("Interpreter 2: the model", "Claim Verification Rate (CVR): every sentence\nbecomes an on-horizon intervention test"),
            ("The whole at four context scales", "Part-Whole Agreement (PWA), radius (HHR):\nthe local account vs. the surrogate whole"),
            ("Interpreter 3: machine readers", "Hermeneutic Gain (HG): readers learn the\nmodel from explanations vs. chance")]
    ys = [4.6, 3.1, 1.6, 0.1]
    for (t, b), y in zip(mids, ys):
        box(ax, 2.75, y, 4.3, 1.25, t, b, "#e6f0fb", fs=7.1)
        arrow(ax, (2.05, 3.0), (2.75, y + 0.62))
        arrow(ax, (7.05, y + 0.62), (7.75, 3.0))
    box(ax, 7.75, 2.0, 2.15, 2.0, "HCT profile", "HP, CVR, PWA,\nGI / AR, HG, HG$_{min}$\n\nweakest-link index\n(for ranking only)", "#fdf1e9", fs=7.1)
    fig.savefig(os.path.join(FIG, "fig1_framework.png"), bbox_inches="tight")
    plt.close(fig)


def fig_comparison(cmp_, roar_):
    """Figure 6: what established metrics see. (a) sign-flip control; (b) ROAR curves; (c) horizon of evaluation points."""
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0], wspace=0.55))
    ax = axs[0]
    sf = cmp_["signflip_vs_hermaih"]
    keys = [("PGI_marg", "PGI (marg.)"), ("PGI_cond", "PGI (cond.)"), ("DEL_marg", "deletion"), ("FC_mean", "faithf. corr."),
            ("CVR_dir", "CVR"), ("PWA", "PWA"), ("HG", "HG")]
    y = np.arange(len(keys))[::-1]
    h = 0.36
    ax.barh(y + h / 2, [sf[k]["hermai_h"] for k, _ in keys], h, color=COLOR["HermAI-H"], zorder=3, label="HermAI-H")
    ax.barh(y - h / 2, [sf[k]["signflip"] for k, _ in keys], h, color=COLOR["SignFlip"], zorder=3, label="Sign-flip")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in keys], fontsize=7.2)
    ax.set_xlim(-0.75, 0.85)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("score (mean of eight settings)")
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    ax.set_title("(a) Sign-flip control", fontsize=8.4, loc="left", color=INK)
    ax = axs[1]
    key = "pima|rf" if "pima|rf" in roar_["curves"] else next(iter(roar_["curves"]), None)
    if key:
        c = roar_["curves"][key]
        fr = [0.0] + c["fracs"]
        for n in ["KernelSHAP", "LIME", "HermAI-H", "Random"]:
            ax.plot(fr, [c["base"]] + c["acc"][n], color=COLOR[n], lw=2, marker="o", ms=3.5, mec="white", mew=0.8, label=LABEL[n].split(" (")[0])
        ax.plot(fr, [c["base"]] + c["acc"]["SignFlip"], color=COLOR["SignFlip"], lw=1.0, ls="--", label="Sign-flip")
        ds, mk = key.split("|")
        ax.set_title(f"(b) ROAR, {'Pima' if ds == 'pima' else 'Breast'} {'forest' if mk == 'rf' else 'MLP'}", fontsize=8.4, loc="left", color=INK)
        ax.set_xlabel("share of features removed")
        ax.set_ylabel("accuracy after retraining")
        ax.set_xticks([0, 0.3, 0.5, 0.7, 0.9])
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    ax = axs[2]
    hor = cmp_["horizon_of_eval_points"]
    dsl = ["breast", "pima", "compas", "adult"]
    mk_ = [("hor_FC_mean", "mean baseline (FC)", "#9a9893"), ("hor_PGI_marg", "marginal draws (PGI)", "#eda100"),
           ("hor_PGI_cond", "conditional (PGI, CVR)", "#2a78d6")]
    x = np.arange(len(dsl))
    w = 0.26
    for i, (k, lab, col) in enumerate(mk_):
        vals = [hor[d][k] for d in dsl]
        ax.bar(x + (i - 1) * w, vals, w, color=col, zorder=3, label=lab)
        for xi, v in zip(x, vals):
            if v < 0.1:
                ax.text(xi + (i - 1) * w, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6, color=INK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["Breast", "Pima", "COMPAS", "Adult"], fontsize=6.8, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("evaluation points inside horizon")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1)
    ax.set_title("(c) Where metrics query f", fontsize=8.4, loc="left", color=INK)
    fig.savefig(os.path.join(FIG, "fig6_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    S, D, inst = load_inst()
    R = dict(comparison=comparison(S, D, inst), disagreement=disagreement(), roar=roar(), planted=planted(), adaptive=adaptive())
    json.dump(R, open(os.path.join(RES, "revision_summary.json"), "w"), indent=1)
    fig_framework()
    fig_comparison(R["comparison"], R["roar"])
    print("revision summary written; ROAR settings:", R["roar"]["summary"].get("_settings"), "; attacks:", list(R["adaptive"]))


if __name__ == "__main__":
    main()

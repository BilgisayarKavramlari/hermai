"""
Aggregates every result file into results/summary.json and draws the figures in figures/.
  python aggregate_and_plot.py
"""
import os
import re
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

ORDER = ["HermAI-H", "LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "Random", "SignFlip"]
LABEL = {"HermAI-H": "HermAI-H", "LIME": "LIME", "KernelSHAP": "KernelSHAP",
         "HermAI-0.1.8": "HermAI 0.1.8", "HermAI-0.1.8-cov": "HermAI 0.1.8 (cov)",
         "Random": "Random (control)", "SignFlip": "Sign-flip (control)", "TreeSHAP-forest": "TreeSHAP of forest (reference)"}
COLOR = {"HermAI-H": "#2a78d6", "LIME": "#eb6834", "KernelSHAP": "#1baf7a", "HermAI-0.1.8": "#eda100",
         "HermAI-0.1.8-cov": "#e87ba4", "Random": "#9a9893", "SignFlip": "#52514e", "TreeSHAP-forest": "#4a3aa7"}
DATASETS = ["breast", "pima", "compas", "adult"]
DNAME = {"breast": "Breast Cancer", "pima": "Pima Diabetes", "compas": "COMPAS", "adult": "Adult Income"}
MODELS = ["rf", "mlp"]
MNAME = {"rf": "Random forest", "mlp": "MLP"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "figure.dpi": 110})


def kendall_w(R):
    """R: raters x objects matrix of scores (higher = better). Average ranks, no tie correction."""
    R = np.asarray(R, float)
    m, n = R.shape
    if m < 2 or n < 2:
        return np.nan
    ranks = np.vstack([pd.Series(r).rank().values for r in R])
    S = ((ranks.sum(axis=0) - m * (n + 1) / 2.0) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def per_explainer(df):
    g = df.groupby("explainer")
    cols = {c: (c, "mean") for c in ["HP", "HP_dist", "HP_type", "CVR_dir", "CVR_dir_feat", "CVR_dir_marg", "CVR_slope", "inversion", "PWA", "PWA_feat", "HHR",
                                     "cf_present", "cf_valid", "cf_on", "cf_pass", "cf_changed", "cf_l1",
                                     "rule_pass", "CVR_all", "cos_whole", "testHP_cond", "testHP_marg",
                                     "H_accept", "PWA_0.5", "PWA_0.25", "PWA_0.125"] if c in df.columns}
    out = g.agg(**{k.replace(".", ""): v for k, v in cols.items()})
    out["GI"] = g["HHR"].apply(lambda s: float((s == 1).mean()))
    out["AR"] = g["HHR"].apply(lambda s: float((s == 0).mean()))
    return out


def reader_summary(readers):
    rows = {}
    for n, r in readers.items():
        hg = [v["HG"] for v in r.values()]
        gva = [v["gain_vs_all"] for v in r.values()]
        rows[n] = dict(HG_mean=float(np.mean(hg)), HG_min=float(np.min(hg)), HG_max=float(np.max(hg)),
                       gain_vs_all=float(np.mean(gva)), **{f"HG_{k}": v["HG"] for k, v in r.items()})
    return pd.DataFrame(rows).T


def load_main():
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "main_*_s*.csv"))):
        m = re.match(r"main_(\w+)_(rf|mlp)_s(\d+)\.csv", os.path.basename(f))
        ds, mk, sd = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(f)
        js = json.load(open(f[:-4] + ".json"))
        pe = per_explainer(df)
        rs = reader_summary(js["readers"])
        pe = pe.join(rs)
        pe["HCT"] = [float((r.HP * r.CVR_dir * r.PWA * max(r.HG_min, 0.0)) ** 0.25) if not np.isnan(r.HP) else np.nan
                     for r in pe.itertuples()]
        W = kendall_w(np.array([[js["readers"][n][rd]["HG"] for n in ORDER if n in js["readers"]]
                                for rd in ["logistic", "knn", "tree"]]))
        out[(ds, mk, sd)] = dict(df=df, meta=js["meta"], pe=pe, readers=js["readers"], W_readers=W)
    return out


def fmt(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def main():
    S = {}
    runs = load_main()
    seed0 = {k: v for k, v in runs.items() if k[2] == 0}
    S["settings"] = []
    S["groups"] = {}
    for (ds, mk, sd), v in sorted(seed0.items()):
        if mk == "rf":
            S["groups"][ds] = v["meta"].get("groups", [])
        mt = v["meta"]
        S["settings"].append(dict(dataset=ds, model=mk, test_acc=mt["test_acc"], whole_fidelity=mt["whole_fidelity"],
                                  hp_real_test=mt["hp_real_test"], hp_real_test_dist=mt.get("hp_real_test_dist"),
                                  n_discrete=mt.get("n_discrete"), n_total=mt.get("n_total"), d=mt["d"],
                                  n_cat=mt.get("n_cat"), pos_rate=mt.get("pos_rate"), n_E=mt["n_E"], n_R=mt["n_R"],
                                  W_readers=v["W_readers"], seconds=mt.get("seconds")))
    # ---- main profile: mean and sd over the seed-0 settings
    metrics = ["HP", "HP_dist", "HP_type", "HCT", "CVR_dir", "CVR_dir_feat", "CVR_dir_marg", "CVR_slope", "inversion", "PWA", "PWA_feat", "GI", "AR", "HHR", "HG_mean",
               "HG_min", "gain_vs_all", "HG_logistic", "HG_knn", "HG_tree", "cf_present", "cf_valid", "cf_on",
               "cf_pass", "cf_changed", "cf_l1", "rule_pass", "cos_whole", "testHP_cond", "testHP_marg",
               "H_accept", "PWA_05", "PWA_025", "PWA_0125"]
    stack = pd.concat([v["pe"].assign(dataset=k[0], model=k[1]) for k, v in seed0.items()])
    stack.index.name = "explainer"
    stack = stack.reset_index()
    prof = {}
    for n in ORDER:
        sub = stack[stack.explainer == n]
        if sub.empty:
            continue
        prof[n] = {}
        for mtr in metrics:
            if mtr in sub.columns:
                vals = sub[mtr].astype(float)
                prof[n][mtr] = dict(mean=fmt(vals.mean(), 4), sd=fmt(vals.std(ddof=1), 4),
                                    min=fmt(vals.min(), 4), max=fmt(vals.max(), 4), n=int(vals.notna().sum()))
    S["profile"] = prof
    # per dataset (mean over the two models) and per setting
    S["per_setting"] = {f"{r.dataset}|{r.model}|{r.explainer}": {m: fmt(getattr(r, m), 4) for m in metrics if m in stack.columns}
                        for r in stack.itertuples()}
    # real-data horizon reference
    S["hp_real_mean"] = fmt(np.mean([s["hp_real_test"] for s in S["settings"]]), 4)
    S["W_readers_mean"] = fmt(np.nanmean([s["W_readers"] for s in S["settings"]]), 4)
    # paired comparisons over instances pooled across settings (HermAI-H vs others)
    pooled = pd.concat([v["df"].assign(setting=f"{k[0]}|{k[1]}") for k, v in seed0.items()])
    S["pooled_n_instances"] = int(pooled[pooled.explainer == "HermAI-H"].shape[0])

    # ---- reliability over seeds (pima, compas)
    rel = {}
    for ds in ["pima", "compas"]:
        for mk in MODELS:
            ks = [k for k in runs if k[0] == ds and k[1] == mk]
            if len(ks) < 3:
                continue
            for mtr in ["HP", "CVR_dir", "PWA", "GI", "HG_mean"]:
                mat = []
                for k in sorted(ks):
                    pe = runs[k]["pe"]
                    mat.append([pe.loc[n, mtr] if n in pe.index else np.nan for n in ORDER])
                mat = np.array(mat, float)
                okc = ~np.isnan(mat).any(axis=0)
                rel.setdefault(mtr, {"sd": [], "W": []})
                rel[mtr]["sd"].append(float(np.nanmean(np.nanstd(mat, axis=0, ddof=1))))
                rel[mtr]["W"].append(kendall_w(mat[:, okc]))
    S["reliability"] = {m: dict(mean_sd=fmt(np.mean(v["sd"]), 4), mean_W=fmt(np.nanmean(v["W"]), 4),
                                min_W=fmt(np.nanmin(v["W"]), 4), n=len(v["W"])) for m, v in rel.items()}

    # ---- planted structure
    pf = os.path.join(RES, "planted_s0.csv")
    if os.path.exists(pf):
        dfp = pd.read_csv(pf)
        mp = json.load(open(os.path.join(RES, "planted_s0.json")))["meta"]
        t1 = pd.read_csv(os.path.join(RES, "planted_top1_s0.csv"))
        pl = {}
        for n in ORDER + ["TreeSHAP-forest"]:
            sub = dfp[dfp.explainer == n]
            if sub.empty:
                continue
            y = sub.in_region.astype(int).values
            tt = t1[t1.explainer == n]
            pl[n] = dict(auc_hhr=fmt(roc_auc_score(y, -sub.HHR.values), 4),
                         auc_pwa=fmt(roc_auc_score(y, -sub.PWA.values), 4),
                         hhr_in=fmt(sub[sub.in_region].HHR.mean(), 4), hhr_out=fmt(sub[~sub.in_region].HHR.mean(), 4),
                         pwa_in=fmt(sub[sub.in_region].PWA.mean(), 4), pwa_out=fmt(sub[~sub.in_region].PWA.mean(), 4),
                         top1_in=fmt(tt[tt.in_region].top1_correct.mean(), 4),
                         top1_out=fmt(tt[~tt.in_region].top1_correct.mean(), 4),
                         mech_in=fmt(tt[tt.in_region].top1_mechanism.mean(), 4) if "top1_mechanism" in tt else None,
                         agree_forest=fmt(tt.agree_forest.mean(), 4) if "agree_forest" in tt else None,
                         cvr_in=fmt(sub[sub.in_region].CVR_dir.mean(), 4) if "CVR_dir" in sub and sub.CVR_dir.notna().any() else None,
                         cvr_out=fmt(sub[~sub.in_region].CVR_dir.mean(), 4) if "CVR_dir" in sub and sub.CVR_dir.notna().any() else None)
        S["planted"] = dict(meta={k: fmt(v, 4) if isinstance(v, float) else v for k, v in mp.items() if not isinstance(v, (dict, list))},
                            explainers=pl)
    # ---- label permutation
    perm = {}
    for ds in ["pima", "breast"]:
        fp = os.path.join(RES, f"permuted_{ds}_s0.csv")
        if os.path.exists(fp) and (ds, "rf", 0) in runs:
            dfq = pd.read_csv(fp)
            jq = json.load(open(fp[:-4] + ".json"))
            peq = per_explainer(dfq).join(reader_summary(jq["readers"]))
            pen = runs[(ds, "rf", 0)]["pe"]
            perm[ds] = dict(fid_perm=fmt(jq["meta"]["whole_fidelity"], 4), fid_norm=fmt(runs[(ds, "rf", 0)]["meta"]["whole_fidelity"], 4),
                            acc_perm=fmt(jq["meta"]["test_acc"], 4), acc_norm=fmt(runs[(ds, "rf", 0)]["meta"]["test_acc"], 4),
                            ga_perm=jq["meta"]["GA_rate"], ga_norm=runs[(ds, "rf", 0)]["meta"]["GA_rate"],
                            explainers={n: {m: dict(perm=fmt(peq.loc[n, m], 4), norm=fmt(pen.loc[n, m], 4))
                                            for m in ["HP", "CVR_dir", "PWA", "GI", "AR", "HG_mean"]}
                                        for n in peq.index})
    S["permuted"] = perm
    # ---- adversarial scaffold
    fa = os.path.join(RES, "adversarial_top_s0.csv")
    if os.path.exists(fa):
        top = pd.read_csv(fa).set_index("explainer")
        dfa = pd.read_csv(os.path.join(RES, "adversarial_s0.csv"))
        ma = json.load(open(os.path.join(RES, "adversarial_s0.json")))["meta"]
        pea = per_explainer(dfa)
        names_a = ma.get("names") or []
        claim_stats = {}
        if "claim_detail" in dfa.columns:
            for n, g in dfa.groupby("explainer"):
                rec = []
                for cd in g.claim_detail.dropna():
                    for it in str(cd).split(";"):
                        if it:
                            j, sg, ok = it.split(":")
                            rec.append((int(j), int(ok)))
                rec = np.array(rec)
                ju, ja = ma.get("j_unrel", 7), ma.get("j_age", 0)
                claim_stats[n] = dict(decoy_claims=int((rec[:, 0] == ju).sum()), decoy_pass=fmt(rec[rec[:, 0] == ju, 1].mean(), 4),
                                      age_claims=int((rec[:, 0] == ja).sum()), age_pass=fmt(rec[rec[:, 0] == ja, 1].mean(), 4),
                                      other_pass=fmt(rec[(rec[:, 0] != ju) & (rec[:, 0] != ja), 1].mean(), 4))
        S["adversarial_claims"] = claim_stats
        S["adversarial"] = dict(det_real_test=fmt(ma["det_real_test"], 4), det_threshold=fmt(ma.get("det_threshold"), 4),
                                det_train_neg_caught=fmt(ma.get("det_train_neg_caught"), 4),
                                explainers={n: dict(age_top1=fmt(top.loc[n, "age_top1"], 4),
                                                    unrelated_top1=fmt(top.loc[n, "unrelated_top1"], 4),
                                                    unrelated_in_top3=fmt(top.loc[n, "unrelated_in_top3"], 4),
                                                    evidence_seen_as_real=fmt(top.loc[n, "evidence_seen_as_real"], 4),
                                                    HP=fmt(pea.loc[n, "HP"], 4), HP_dist=fmt(pea.loc[n, "HP_dist"], 4),
                                                    HP_type=fmt(pea.loc[n, "HP_type"], 4), CVR_dir=fmt(pea.loc[n, "CVR_dir"], 4))
                                            for n in top.index})
    # ---- consistency audit
    cons = [pd.read_csv(f) for f in glob.glob(os.path.join(RES, "consistency_*.csv"))]
    if cons:
        cdf = pd.concat(cons)
        S["consistency"] = {f"{r.dataset}|{r.model}|{r.explainer}": dict(
                                sa=fmt(r.run_sign_agreement, 4), jac=fmt(r.run_jaccard, 4),
                                sa_u=fmt(getattr(r, "run_sign_agreement_unit", np.nan), 4),
                                jac_u=fmt(getattr(r, "run_jaccard_unit", np.nan), 4))
                            for r in cdf.itertuples()}
        S["consistency_mean"] = {n: dict(sa=fmt(cdf[cdf.explainer == n].run_sign_agreement.mean(), 4),
                                         jac=fmt(cdf[cdf.explainer == n].run_jaccard.mean(), 4))
                                 for n in cdf.explainer.unique()}
    fe = os.path.join(RES, "example.json")
    if os.path.exists(fe):
        S["example"] = json.load(open(fe))
    json.dump(S, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    stack.to_csv(os.path.join(RES, "per_setting_profile.csv"), index=False)
    make_figures(S, seed0, stack)
    print("summary written;", len(seed0), "seed-0 settings")


# ----------------------------------------------------------------------------- figures
def hbar_panel(ax, values, errs, names, xlabel, ref=None, ref_label=None, xlim=(0, 1.08), show_names=True):
    y = np.arange(len(names))[::-1]
    values = [np.nan if v is None else v for v in values]
    errs = [np.nan if e is None else e for e in errs]
    ax.barh(y, values, height=0.62, color=[COLOR[n] for n in names], xerr=errs,
            error_kw=dict(ecolor=INK2, elinewidth=0.9, capsize=2.2), zorder=3)
    for yi, v, e in zip(y, values, errs):
        if np.isnan(v):
            ax.text(0.01, yi, "n/a", va="center", ha="left", fontsize=7, color=INK2)
            continue
        xe = v + (0 if np.isnan(e) else e)
        ax.text(max(xe, 0) + 0.02, yi, f"{v:.2f}", va="center", ha="left", fontsize=7, color=INK)
    if ref is not None:
        ax.axvline(ref, color=INK, linewidth=0.9, zorder=4)
        ax.text(ref, len(names) - 0.35, ref_label, ha="right", va="bottom", fontsize=6.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[n] for n in names] if show_names else [""] * len(names), fontsize=7.4)
    ax.set_xlabel(xlabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(names) + (0.35 if ref is not None else -0.2))
    ax.grid(axis="y", visible=False)


def make_figures(S, seed0, stack):
    prof = S["profile"]
    # Figure 1: framework schematic (code-drawn)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=8.2, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.50, body, ha="center", va="top", fontsize=7.1, color=INK, linespacing=1.3)
    box(0.1, 2.15, 1.95, 1.7, "Explanation", "attributions\nnarrative claims\ncounterfactual\nevidence points", "#f0efec")
    mids = [("Interpreter 1: the data", "Horizon Plausibility (HP): evidence points\ninside the conformal, scale-aware horizon"),
            ("Interpreter 2: the model", "Claim Verification Rate (CVR): every sentence\nbecomes an on-horizon intervention test"),
            ("The whole at four context scales", "Part-Whole Agreement (PWA), radius (HHR):\nthe local account vs. the surrogate whole"),
            ("Interpreter 3: machine readers", "Hermeneutic Gain (HG): readers learn the\nmodel from explanations vs. chance")]
    ys = [4.6, 3.1, 1.6, 0.1]
    for (t, b), y in zip(mids, ys):
        box(2.75, y, 4.3, 1.25, t, b, "#e6f0fb")
        ax.annotate("", xy=(2.75, y + 0.62), xytext=(2.05, 3.0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
        ax.annotate("", xy=(7.75, 3.0), xytext=(7.05, y + 0.62),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
    box(7.75, 2.0, 2.15, 2.0, "HCT profile", "HP, CVR, PWA,\nGI / AR, HG, HG$_{min}$\n\nweakest-link index\n(for ranking only)", "#fdf1e9")
    fig.savefig(os.path.join(FIG, "fig1_framework.png"), bbox_inches="tight")
    plt.close(fig)

    names = [n for n in ORDER if n in prof]
    # Figure 2: HP and CVR
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.16))
    hbar_panel(axs[0], [prof[n]["HP"]["mean"] for n in names], [prof[n]["HP"]["sd"] for n in names], names,
               "Horizon Plausibility (HP)", ref=S["hp_real_mean"], ref_label=f"real data {S['hp_real_mean']:.2f}", xlim=(0, 1.22))
    axs[0].set_xticks(np.arange(0, 1.01, 0.2))
    axs[0].set_title("(a) Evidence inside the data horizon", fontsize=8.6, loc="left", color=INK)
    hbar_panel(axs[1], [prof[n]["CVR_dir"]["mean"] for n in names], [prof[n]["CVR_dir"]["sd"] for n in names], names,
               "Claim Verification Rate (CVR)", xlim=(0, 1.0), show_names=False)
    axs[1].set_title("(b) Direction claims passing on-horizon tests", fontsize=8.6, loc="left", color=INK)
    for a in axs:
        a.set_ylim(-0.6, len(names) + 0.35)
    fig.savefig(os.path.join(FIG, "fig2_hp_cvr.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 3: coherence profile across context scales
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    sc = [1.0, 0.5, 0.25, 0.125]
    keys = ["PWA", "PWA_05", "PWA_025", "PWA_0125"]
    for n in names:
        ys = [prof[n][k]["mean"] for k in keys]
        ax.plot(range(4), ys, color=COLOR[n], lw=2, marker="o", ms=5, mec="white", mew=1.2, label=LABEL[n],
                solid_capstyle="round")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["whole data\n(s = 1)", "s = 1/2", "s = 1/4", "s = 1/8"])
    ax.set_ylabel("Part-Whole Agreement (PWA$_s$)")
    ax.set_xlabel("Context scale of the whole (share of training data nearest to the instance)")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_coherence_profile.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 4: readers -- curves for one setting + HG summary
    key = ("adult", "rf", 0) if ("adult", "rf", 0) in seed0 else sorted(seed0)[0]
    key = ("pima", "rf", 0) if ("pima", "rf", 0) in seed0 else key
    rd = seed0[key]["readers"]
    mlist = seed0[key]["meta"]["m_list"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.55))
    ax = axs[0]
    rr = "knn"
    ax.plot(mlist, rd["HermAI-H"][rr]["acc_all"], color=INK2, lw=2, marker="s", ms=4, label="all features (reference)")
    ax.plot(mlist, rd["HermAI-H"][rr]["acc_B"], color="#9a9893", lw=2, marker="o", ms=4, label="3 random features (B)")
    for n in ["HermAI-H", "LIME", "KernelSHAP"]:
        if n in rd:
            ax.plot(mlist, rd[n][rr]["acc_A"], color=COLOR[n], lw=2, marker="o", ms=4, mec="white", mew=1,
                    label=f"3 features from {LABEL[n].split(' (')[0]} (A)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(mlist)
    ax.set_xticklabels([str(m) for m in mlist])
    ax.set_xlabel("explained cases seen by the reader (m)")
    ax.set_ylabel("agreement with the model")
    ax.set_title(f"(a) 3-NN reader, {DNAME[key[0]]}, {MNAME[key[1]].lower()}", fontsize=8.6, loc="left", color=INK)
    ax.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    hg = [prof[n]["HG_mean"]["mean"] for n in names]
    hbar_panel(axs[1], hg, [prof[n]["HG_mean"]["sd"] for n in names], names,
               "Hermeneutic Gain (mean of readers)", xlim=(min(-0.25, min(hg) - 0.15), 1.0))
    axs[1].axvline(0, color=INK2, lw=0.8)
    axs[1].set_title("(b) Gain over chance feature choice", fontsize=8.6, loc="left", color=INK)
    fig.savefig(os.path.join(FIG, "fig4_readers.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 5: meta-validation
    if "planted" in S and "adversarial" in S:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=0.75))
        pl = S["planted"]["explainers"]
        pn = ["TreeSHAP-forest"] + [n for n in ORDER if n in pl and n not in ("SignFlip",)]
        y = np.arange(len(pn))[::-1]
        h = 0.36
        ax = axs[0]
        ax.barh(y + h / 2 + 0.01, [pl[n]["hhr_out"] for n in pn], h, color=[COLOR[n] for n in pn], alpha=0.4, zorder=3)
        ax.barh(y - h / 2 - 0.01, [pl[n]["hhr_in"] for n in pn], h, color=[COLOR[n] for n in pn], zorder=3)
        for yi, n in zip(y, pn):
            ax.text(1.03, yi, f"{pl[n]['auc_hhr']:.2f}", va="center", ha="left", fontsize=7, color=INK)
        ax.text(1.03, len(pn) - 0.45, "AUC", va="bottom", ha="left", fontsize=7, color=INK, weight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[n] for n in pn], fontsize=7.2)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("mean horizon radius (HHR)")
        ax.grid(axis="y", visible=False)
        ax.set_title("(a) Planted exception: outside (light), inside (dark)", fontsize=8.2, loc="left", color=INK)
        ad = S["adversarial"]["explainers"]
        an = [n for n in ORDER if n in ad]
        y = np.arange(len(an))[::-1]
        ax = axs[1]
        ax.barh(y + h / 2 + 0.01, [ad[n]["age_top1"] for n in an], h, color=[COLOR[n] for n in an], zorder=3,
                label="true feature (age)")
        ax.barh(y - h / 2 - 0.01, [ad[n]["unrelated_top1"] for n in an], h, color="white",
                edgecolor=[COLOR[n] for n in an], hatch="////", linewidth=0.9, zorder=3, label="decoy column")
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[n] for n in an], fontsize=7.2)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("share of explanations (top-1 feature)")
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
        ax.set_title("(b) Scaffold attack on COMPAS", fontsize=8.2, loc="left", color=INK)
        fig.savefig(os.path.join(FIG, "fig5_meta_validation.png"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()

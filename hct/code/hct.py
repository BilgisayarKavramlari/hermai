"""
Hermeneutic Circle Test (HCT) -- reference implementation used in the paper
"Measuring the Hermeneutic Circle in Explainable AI".

Components
  HP   Horizon Plausibility      share of an explainer's evidence points inside the data horizon
                                  (split-conformal k-NN novelty test, alpha = 0.05)
  CVR  Claim Verification Rate   share of an explanation's claims that pass an executable,
                                  on-horizon test against the model
  PWA  Part-Whole Agreement      sign agreement@3 between the local explanation (part) and the
                                  exact TreeSHAP account of a surrogate of the whole, gated by
                                  whether the whole reproduces the decision
  HHR  Hermeneutic Horizon Radius largest context scale (1, 1/2, 1/4, 1/8 of the data) at which
                                  part and whole agree
  HG   Hermeneutic Gain          normalised learning-curve gain of machine readers that use the
                                  explanations as a prior, against the same readers without them
All numbers in the paper are produced by this file and run_experiments.py.
"""
import io
import contextlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, LogisticRegression

import shap
from lime.lime_tabular import LimeTabularExplainer
# documented import path of hermai 0.1.8 -> independent-noise generator (tabular_generator.py)
from hermai.perturbations import TabularPerturbationGenerator as HermaiPublicGen
# covariance-aware generator shipped in the same release (tabular.py)
from hermai.perturbations.tabular import TabularPerturbationGenerator as HermaiCovGen
from hermai.explainers import LocalExplainer as HermaiLocalExplainer

ALPHA = 0.05      # horizon level
K_HOR = 10        # k of the k-NN novelty score
K_COND = 30       # neighbours used for conditional (on-horizon) replacement tests
EPS = 0.01        # minimum effect size for a direction claim
TOPK = 3          # claims / agreement depth
N_SAMPLES = 1000  # perturbation budget for sampling explainers
SCALES = [1.0, 0.5, 0.25, 0.125]
TAU_PWA = 2.0 / 3.0
RHO_GROUP = 0.9    # features with average |rho| >= 0.9 form one meaning unit (part)


# ----------------------------------------------------------------------------- data
def load_dataset(name, data_dir, seed=0):
    """Return X (DataFrame, numeric codes), y (0/1), list of categorical column names."""
    if name == "breast":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        X = d.data.astype(float).copy()
        X.columns = [c.replace(" ", "_") for c in X.columns]
        y = d.target.values.astype(int)          # 1 = benign
        cat = []
    elif name == "pima":
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigree", "Age", "Outcome"]
        df = pd.read_csv(f"{data_dir}/pima.csv", header=None, names=cols)
        X = df.drop(columns="Outcome").astype(float)
        y = df["Outcome"].values.astype(int)
        cat = []
    elif name == "compas":
        df = pd.read_csv(f"{data_dir}/compas.csv")
        df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30)
                & (df.is_recid != -1) & (df.c_charge_degree != "O") & (df.score_text != "N/A")]
        X = pd.DataFrame({
            "age": df.age, "priors_count": df.priors_count, "juv_fel_count": df.juv_fel_count,
            "juv_misd_count": df.juv_misd_count, "juv_other_count": df.juv_other_count,
            "charge_felony": (df.c_charge_degree == "F").astype(int),
            "sex_male": (df.sex == "Male").astype(int)}).astype(float).reset_index(drop=True)
        y = df.two_year_recid.values.astype(int)
        cat = ["charge_felony", "sex_male"]
    elif name == "adult":
        names = ["Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MaritalStatus",
                 "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
                 "HoursPerWeek", "Country", "Target"]
        df = pd.read_csv(f"{data_dir}/adult.data", header=None, names=names, skipinitialspace=True)
        df = df.drop(columns=["fnlwgt", "Education"])
        y_all = (df.pop("Target").str.strip() == ">50K").astype(int).values
        cat = ["Workclass", "MaritalStatus", "Occupation", "Relationship", "Race", "Sex", "Country"]
        for c in cat:   # integer codes, most frequent category = 0
            order = df[c].value_counts().index.tolist()
            df[c] = df[c].map({v: i for i, v in enumerate(order)})
        X_all = df.astype(float)
        # fixed stratified subsample of 6000 rows for computational tractability
        X, _, y, _ = train_test_split(X_all, y_all, train_size=6000, stratify=y_all, random_state=0)
        X = X.reset_index(drop=True)
    else:
        raise ValueError(name)
    return X.reset_index(drop=True), np.asarray(y), cat


def train_model(kind, X, y, seed):
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed)
    elif kind == "mlp":
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                          max_iter=800, early_stopping=True,
                                                          random_state=seed))
    else:
        raise ValueError(kind)
    m.fit(X, y)
    return m


class ModelWrap:
    """Lets hermai pass DataFrames to a model fitted on numpy arrays."""
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ----------------------------------------------------------------------------- horizon
class Horizon:
    """Split-conformal novelty test: p(z) = (1 + #{cal scores >= s(z)}) / (n_cal + 1)."""
    def __init__(self, Z_fit, Z_cal, k=K_HOR, alpha=ALPHA):
        self.k, self.alpha = k, alpha
        self.nn = NearestNeighbors(n_neighbors=k).fit(Z_fit)
        self.s_cal = np.sort(self.nn.kneighbors(Z_cal, n_neighbors=k)[0][:, -1])
        self.n = len(self.s_cal)

    def pvalue(self, Z):
        Z = np.atleast_2d(Z)
        s = self.nn.kneighbors(Z, n_neighbors=self.k)[0][:, -1]
        cnt = self.n - np.searchsorted(self.s_cal, s, side="left")
        return (1.0 + cnt) / (self.n + 1.0)

    def on(self, Z):
        return self.pvalue(Z) > self.alpha


# ----------------------------------------------------------------------------- context
class Context:
    """Everything that is fixed for one (dataset, model, seed) setting."""
    def __init__(self, X, y, cat, model_kind, seed, n_explain=80, n_reader_test=1000,
                 model=None, permute_labels=False, X_test=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.names = list(X.columns)
        self.d = len(self.names)
        self.cat_names = list(cat)
        self.cat_idx = [self.names.index(c) for c in cat]
        self.num_idx = [j for j in range(self.d) if j not in self.cat_idx]
        Xa = X.values.astype(float)
        Xtr, Xte, ytr, yte = train_test_split(Xa, y, test_size=0.3, stratify=y, random_state=seed)
        if permute_labels:
            ytr = self.rng.permutation(ytr)
        self.Xtr, self.Xte, self.ytr, self.yte = Xtr, Xte, ytr, yte
        self.Xfit, self.Xcal = train_test_split(Xtr, test_size=0.25, random_state=seed)
        self.mu = Xtr.mean(axis=0)
        self.sd = Xtr.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.n_explain, self.n_reader_test = n_explain, n_reader_test
        # measurement scale of each feature: integer-valued features and observed ranges
        self.discrete = np.array([np.allclose(Xtr[:, j], np.round(Xtr[:, j])) for j in range(self.d)])
        self.xmin, self.xmax = Xtr.min(axis=0), Xtr.max(axis=0)
        self.Zfit, self.Zcal, self.Zte = self.z(self.Xfit), self.z(self.Xcal), self.z(Xte)
        self.horizon = Horizon(self.Zfit, self.Zcal)
        self.hp_real_test = float(self.on_horizon(self.Zte).mean())
        self.hp_real_test_dist = float(self.horizon.on(self.Zte).mean())
        # neighbour indices without feature j (for conditional replacement tests)
        self.nn_minus = []
        for j in range(self.d):
            keep = [i for i in range(self.d) if i != j]
            self.nn_minus.append((keep, NearestNeighbors(n_neighbors=K_COND).fit(self.Zfit[:, keep])))
        self.nn_all = NearestNeighbors(n_neighbors=K_COND).fit(self.Zfit)
        # meaning units: near-duplicate features are merged into one part
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        Cm = np.nan_to_num(np.abs(np.corrcoef(Xtr, rowvar=False)))
        if self.d > 1:
            Dm = 1.0 - Cm
            np.fill_diagonal(Dm, 0.0)
            lab = fcluster(linkage(squareform(np.clip(Dm, 0, None), checks=False), method="average"),
                           t=1.0 - RHO_GROUP, criterion="distance")
        else:
            lab = np.array([1])
        self.groups = [list(np.where(lab == g)[0]) for g in np.unique(lab)]
        self.gid = np.zeros(self.d, int)
        for gi, g in enumerate(self.groups):
            self.gid[g] = gi
        self.nn_group = {}
        for gi, g in enumerate(self.groups):
            if len(g) > 1:
                keep = [i for i in range(self.d) if i not in g]
                self.nn_group[gi] = (keep, NearestNeighbors(n_neighbors=K_COND).fit(self.Zfit[:, keep]))
        self.set_model(model if model is not None else train_model(model_kind, Xtr, ytr, seed))

    def set_model(self, model):
        """(Re)compute every model-dependent quantity."""
        self.model = model
        self.mw = ModelWrap(model)
        self.f_fit = self.f(self.Xfit)
        self.f_te = self.f(self.Xte)
        self.test_acc = float(((self.f_te >= 0.5).astype(int) == self.yte).mean())
        # explained set E (stratified by predicted class) and reader evaluation set R
        rng = np.random.default_rng(self.seed + 1)
        cls = (self.f_te >= 0.5).astype(int)
        idx1, idx0 = np.where(cls == 1)[0], np.where(cls == 0)[0]
        n1 = min(len(idx1), self.n_explain // 2)
        n0 = min(len(idx0), self.n_explain - n1)
        n1 = min(len(idx1), self.n_explain - n0)
        E = np.concatenate([rng.choice(idx1, n1, replace=False), rng.choice(idx0, n0, replace=False)])
        self.E = rng.permutation(E)
        rest = np.setdiff1d(np.arange(len(self.Xte)), self.E)
        self.R = rng.choice(rest, min(self.n_reader_test, len(rest)), replace=False)
        # global whole: regression tree on the model's probabilities (exact TreeSHAP)
        self.whole = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=self.seed).fit(self.Zfit, self.f_fit)
        self.whole_fidelity = float(((self.whole.predict(self.Zte) >= 0.5) == (self.f_te >= 0.5)).mean())
        # global rule surrogate (as hermai's GeneralExplainer: depth-3 classification tree)
        yfit_cls = (self.f_fit >= 0.5).astype(int)
        self.rule_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(self.Xfit, yfit_cls)
        leaves = self.rule_tree.apply(self.Xfit)
        self.leaf_prec = {}
        for lf in np.unique(leaves):
            m = leaves == lf
            c = int(np.round(yfit_cls[m].mean()))
            self.leaf_prec[lf] = (c, float((yfit_cls[m] == c).mean()))

    # helpers
    def z(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd

    def x_from_z(self, Z):
        """Back-transform and snap to the measurement scale (integers, observed range)."""
        X = np.asarray(Z, float) * self.sd + self.mu
        if self.discrete.any():
            X[..., self.discrete] = np.round(X[..., self.discrete])
        return np.clip(X, self.xmin, self.xmax)

    def type_ok(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        ok = np.ones(len(X), bool)
        if self.discrete.any():
            D = X[:, self.discrete]
            ok &= (np.abs(D - np.round(D)) <= 1e-6 * np.maximum(1.0, np.abs(D))).all(axis=1)
        tol = 1e-9 + 1e-6 * np.abs(self.xmax - self.xmin)
        ok &= ((X >= self.xmin - tol) & (X <= self.xmax + tol)).all(axis=1)
        return ok

    def on_horizon(self, Z):
        """Horizon = conformal k-NN test AND the measurement scale of every feature."""
        Z = np.atleast_2d(np.asarray(Z, float))
        return self.horizon.on(Z) & self.type_ok(Z * self.sd + self.mu)

    def f(self, X):
        return self.model.predict_proba(np.atleast_2d(np.asarray(X, float)))[:, 1]

    def f2(self, X):
        return self.model.predict_proba(np.atleast_2d(np.asarray(X, float)))


# ----------------------------------------------------------------------------- explainers
def topk(a, k=TOPK):
    return [int(j) for j in np.argsort(-np.abs(a), kind="stable")[:k]]


class LimeExplainer:
    name = "LIME"

    def __init__(self, ctx):
        self.ctx = ctx
        self.e = LimeTabularExplainer(ctx.Xtr, feature_names=ctx.names, categorical_features=ctx.cat_idx,
                                      mode="classification", discretize_continuous=True,
                                      random_state=ctx.seed)

    def explain(self, x, i):
        ctx = self.ctx
        ex = self.e.explain_instance(x, ctx.f2, num_features=ctx.d, num_samples=N_SAMPLES, labels=(1,))
        a = np.zeros(ctx.d)
        for j, w in ex.as_map()[1]:
            a[j] = w
        claims = [(j, float(np.sign(a[j])), "contrib") for j in topk(a) if a[j] != 0]
        _, inv = self.e._LimeTabularExplainer__data_inverse(x, 200)
        return dict(attr=a, claims=claims, samples=inv[1:], cf=None, rule=False)


class KernelShapExplainer:
    name = "KernelSHAP"

    def __init__(self, ctx):
        self.ctx = ctx
        self.bg = shap.kmeans(ctx.Xfit, 10)
        self.e = shap.KernelExplainer(ctx.f, self.bg)
        self.rng = np.random.default_rng(ctx.seed + 7)

    def explain(self, x, i):
        ctx = self.ctx
        sv = np.asarray(self.e.shap_values(x.reshape(1, -1), nsamples=500, silent=True)).reshape(-1)
        claims = [(j, float(np.sign(sv[j])), "contrib") for j in topk(sv) if sv[j] != 0]
        # evidence points of KernelSHAP: coalitions completed with background rows
        B, w = self.bg.data, np.asarray(self.bg.weights, float)
        w = w / w.sum()
        S = self.rng.random((200, ctx.d)) < 0.5
        rows = B[self.rng.choice(len(B), 200, p=w)]
        samples = np.where(S, x[None, :], rows)
        return dict(attr=sv, claims=claims, samples=samples, cf=None, rule=False)


class HermaiV018Explainer:
    """hermai 0.1.8 LocalExplainer with either generator; narrative semantics = sign of slope."""

    def __init__(self, ctx, gen_cls, name):
        self.ctx, self.name = ctx, name
        Xdf = pd.DataFrame(ctx.Xtr, columns=ctx.names)
        with contextlib.redirect_stdout(io.StringIO()):
            g = gen_cls(categorical_features=list(ctx.cat_names))
            g.fit(Xdf)
        orig = g.perturb
        holder = self

        def perturb(instance, n_samples=1000):
            P = orig(instance, n_samples=n_samples)
            holder._last = P
            return P
        g.perturb = perturb
        self.le = HermaiLocalExplainer(ctx.mw, g)

    def explain(self, x, i):
        ctx = self.ctx
        np.random.seed(100003 * ctx.seed + i)        # hermai 0.1.8 draws from the global RNG
        s = pd.Series(x, index=ctx.names)
        with contextlib.redirect_stdout(io.StringIO()):
            ex = self.le.explain(s, n_samples=N_SAMPLES)
        w = ex.feature_importances.set_index("feature").loc[ctx.names, "importance"].values.astype(float)
        P = self._last.values.astype(float)
        sdp = P.std(axis=0)
        sdp[sdp == 0] = 1.0
        attr = w / sdp * (x - ctx.mu)                 # slope x deviation from the training mean
        claims = [(j, float(np.sign(w[j])), "slope") for j in topk(w) if w[j] != 0]
        cf = None if ex.counterfactual is None else ex.counterfactual.values.astype(float)
        return dict(attr=attr, claims=claims, samples=P, cf=cf, rule=True, slope=w)


class HermaiHExplainer:
    """Horizon-aware HermAI variant proposed in the paper (HermAI-H)."""
    name = "HermAI-H"

    def __init__(self, ctx):
        self.ctx = ctx
        self.rng = np.random.default_rng(ctx.seed + 11)
        num = ctx.num_idx
        if num:
            R = np.corrcoef(ctx.Zfit[:, num], rowvar=False)
            R = np.atleast_2d(np.nan_to_num(R))
            R = R + 1e-6 * np.eye(len(num))
            self.L = np.linalg.cholesky(R)
        self.last_tau = None

    def sample(self, x, n=N_SAMPLES):
        ctx, rng = self.ctx, self.rng
        z = ctx.z(x)
        nb = ctx.nn_all.kneighbors(z.reshape(1, -1), n_neighbors=K_COND)[1][0]
        tau, kept, total, drawn = 0.5, [], 0, 0
        for _ in range(10):
            Zs = np.repeat(z[None, :], n, axis=0)
            if ctx.num_idx:
                Zs[:, ctx.num_idx] += tau * rng.standard_normal((n, len(ctx.num_idx))) @ self.L.T
            for j in ctx.cat_idx:
                m = rng.random(n) < 0.15
                Zs[m, j] = ctx.Zfit[rng.choice(nb, m.sum()), j]
            Zs = ctx.z(ctx.x_from_z(Zs))             # snap categorical codes
            ok = ctx.on_horizon(Zs)
            kept.append(Zs[ok])
            total += ok.sum()
            drawn += n
            if total >= n:
                break
            if ok.mean() < 0.2:
                tau *= 0.6
        self.last_tau = tau
        self.last_accept = float(total / drawn)
        Zk = np.vstack(kept)[:n]
        if len(Zk) < 50:                               # fall back to real neighbours
            Zk = ctx.Zfit[ctx.nn_all.kneighbors(z.reshape(1, -1), n_neighbors=min(200, len(ctx.Zfit)))[1][0]]
        return Zk

    def sparse_cf(self, x, z, Zs, ps):
        ctx = self.ctx
        cls = ctx.f(x)[0] >= 0.5
        flip = (ps >= 0.5) != cls
        if not flip.any():
            return None
        cand = Zs[flip]
        zc = cand[np.argmin(np.linalg.norm(cand - z, axis=1))].copy()
        for j in np.argsort(np.abs(zc - z)):
            if zc[j] == z[j]:
                continue
            trial = zc.copy()
            trial[j] = z[j]
            if (ctx.f(ctx.x_from_z(trial))[0] >= 0.5) != cls and ctx.on_horizon(trial)[0]:
                zc = trial
        return ctx.x_from_z(zc)

    def explain(self, x, i):
        ctx = self.ctx
        z = ctx.z(x)
        Zs = self.sample(x)
        Xs = ctx.x_from_z(Zs)
        ps = ctx.f(Xs)
        dist = np.linalg.norm(Zs - z, axis=1)
        med = np.median(dist[dist > 0]) if (dist > 0).any() else 1.0
        wts = np.exp(-dist ** 2 / med ** 2)
        beta = Ridge(alpha=1.0).fit(Zs, ps, sample_weight=wts).coef_
        attr = beta * z                                  # contribution relative to the training mean
        claims = [(j, float(np.sign(attr[j])), "contrib") for j in topk(attr) if attr[j] != 0]
        cf = self.sparse_cf(x, z, Zs, ps)
        return dict(attr=attr, claims=claims, samples=Xs, cf=cf, rule=True,
                    accept=self.last_accept, tau=self.last_tau)


def random_explanation(ctx, rng):
    a = rng.standard_normal(ctx.d)
    return dict(attr=a, claims=[(j, float(np.sign(a[j])), "contrib") for j in topk(a)],
                samples=None, cf=None, rule=False)


def signflip_explanation(h):
    a = -h["attr"]
    return dict(attr=a, claims=[(j, -s, sem) for (j, s, sem) in h["claims"]],
                samples=h["samples"], cf=None, rule=False)


# ----------------------------------------------------------------------------- claim tests
def test_contrib(ctx, x, j, s, mode, rng, level="feature"):
    """On-horizon (conditional) or marginal replacement test of a contribution claim.
    level='group' replaces the whole meaning unit that contains feature j."""
    cols = [j]
    if level == "group" and len(ctx.groups[ctx.gid[j]]) > 1:
        cols = ctx.groups[ctx.gid[j]]
    if mode == "cond":
        keep, nn = ctx.nn_minus[j] if len(cols) == 1 else ctx.nn_group[ctx.gid[j]]
        idx = nn.kneighbors(ctx.z(x)[keep].reshape(1, -1), n_neighbors=K_COND)[1][0]
    else:
        idx = rng.choice(len(ctx.Xfit), K_COND, replace=True)
    Xr = np.repeat(x[None, :], K_COND, axis=0)
    Xr[:, cols] = ctx.Xfit[idx][:, cols]
    dlt = ctx.f(x)[0] - ctx.f(Xr)
    m = dlt.mean()
    se = dlt.std(ddof=1) / np.sqrt(K_COND)
    passed = bool(np.sign(m) == s and abs(m) >= max(EPS, 1.96 * se))
    return passed, float(ctx.on_horizon(ctx.z(Xr)).mean())


def test_slope(ctx, x, j, s):
    """Library-literal reading of a hermai 0.1.8 sentence: 'increasing x_j increases P(y=1)'."""
    if j in ctx.cat_idx:
        return None
    vals = []
    for h in (0.25, 0.5):
        up, dn = x.copy(), x.copy()
        up[j] += h * ctx.sd[j]
        dn[j] -= h * ctx.sd[j]
        vals.append(ctx.f(up)[0] - ctx.f(dn)[0])
    m = float(np.mean(vals))
    return bool(np.sign(m) == s and abs(m) >= EPS)


def test_cf(ctx, x, cf):
    valid = bool((ctx.f(cf)[0] >= 0.5) != (ctx.f(x)[0] >= 0.5))
    on = bool(ctx.on_horizon(ctx.z(cf))[0])
    changed = int((np.abs(cf - x) > 1e-9).sum())
    l1 = float(np.abs(ctx.z(cf) - ctx.z(x)).sum())
    return valid, on, changed, l1


def test_rule(ctx, x):
    lf = ctx.rule_tree.apply(x.reshape(1, -1))[0]
    c, prec = ctx.leaf_prec[lf]
    return bool(c == int(ctx.f(x)[0] >= 0.5) and prec >= 0.8)


# ----------------------------------------------------------------------------- part-whole
def group_sum(ctx, a):
    return np.array([a[g].sum() for g in ctx.groups])


def sign_agreement(a, g, k=TOPK):
    k = min(k, len(a))
    ta, tg = topk(a, k), topk(g, k)
    common = set(ta) & set(tg)
    return sum(1 for j in common if np.sign(a[j]) == np.sign(g[j]) and a[j] != 0) / k


def regional_wholes(ctx):
    """Exact TreeSHAP accounts of the global and regional wholes for every explained instance."""
    ZE = ctx.Zte[ctx.E]
    fE = ctx.f_te[ctx.E]
    G, GA = {}, {}
    tex = shap.TreeExplainer(ctx.whole)
    G[1.0] = np.asarray(tex.shap_values(ZE))
    GA[1.0] = ((ctx.whole.predict(ZE) >= 0.5) == (fE >= 0.5))
    nfit = len(ctx.Zfit)
    nnr = NearestNeighbors().fit(ctx.Zfit)
    for s in SCALES[1:]:
        k = max(40, int(s * nfit))
        Gs, GAs = np.zeros_like(G[1.0]), np.zeros(len(ZE), bool)
        idxs = nnr.kneighbors(ZE, n_neighbors=k)[1]
        for r in range(len(ZE)):
            nb = idxs[r]
            t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=ctx.seed).fit(ctx.Zfit[nb], ctx.f_fit[nb])
            Gs[r] = np.asarray(shap.TreeExplainer(t).shap_values(ZE[r:r + 1])).reshape(-1)
            GAs[r] = (t.predict(ZE[r:r + 1])[0] >= 0.5) == (fE[r] >= 0.5)
        G[s], GA[s] = Gs, GAs
    return G, GA


# ----------------------------------------------------------------------------- readers (HG)
READERS = ["logistic", "knn", "tree"]
R_FEAT = 3


def _reader_acc(reader, ZE, yE, ZR, yR, S, cols):
    if reader == "logistic":
        clf = LogisticRegression(C=1.0, max_iter=2000)
    elif reader == "knn":
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(ZE[S][:, cols], yE[S])
    return float((clf.predict(ZR[:, cols]) == yR).mean())


def reader_curves(ctx, attrs, m_list=(10, 20, 40, 80), reps=10, n_rand=5, seed=0):
    """Machine readers learn the model's decisions from m explained cases.
    Condition A: the reader attends to the R_FEAT features that the explanations of those m cases
    rank highest (pre-understanding). Baseline B: R_FEAT features drawn at random (same budget).
    Reference 'all': the reader sees every feature.
    HG = (AUC_A - AUC_B) / (1 - AUC_B), AUC over log2(m)."""
    rng = np.random.default_rng(seed + 99)
    ZE, ZR = ctx.Zte[ctx.E], ctx.Zte[ctx.R]
    yE = (ctx.f_te[ctx.E] >= 0.5).astype(int)
    yR = (ctx.f_te[ctx.R] >= 0.5).astype(int)
    nE = len(ctx.E)
    m_list = [m for m in m_list if m <= nE]
    subsets = {}
    for m in m_list:
        subsets[m] = []
        for _r in range(reps if m < nE else 1):
            for _ in range(200):
                S = rng.choice(nE, m, replace=False)
                if len(np.unique(yE[S])) == 2 and min(np.bincount(yE[S])) >= 2:
                    break
            subsets[m].append(S)
    rand_cols = {(m, s): [list(rng.choice(ctx.d, R_FEAT, replace=False)) for _ in range(n_rand)]
                 for m in m_list for s in range(len(subsets[m]))}
    x = np.log2(m_list)

    def auc(acc):
        return float(np.trapezoid(acc, x) / (x[-1] - x[0]))

    base, allf = {}, {}
    for rd in READERS:
        base[rd] = np.array([np.mean([np.mean([_reader_acc(rd, ZE, yE, ZR, yR, S, c) for c in rand_cols[(m, s)]])
                                      for s, S in enumerate(subsets[m])]) for m in m_list])
        allf[rd] = np.array([np.mean([_reader_acc(rd, ZE, yE, ZR, yR, S, list(range(ctx.d)))
                                      for S in subsets[m]]) for m in m_list])
    out = {}
    for name, A in attrs.items():
        res = {}
        for rd in READERS:
            accA = []
            for m in m_list:
                vals = []
                for S in subsets[m]:
                    cols = topk(np.abs(A[S]).mean(axis=0), R_FEAT)
                    vals.append(_reader_acc(rd, ZE, yE, ZR, yR, S, cols))
                accA.append(np.mean(vals))
            accA = np.array(accA)
            aA, aB, aAll = auc(accA), auc(base[rd]), auc(allf[rd])
            res[rd] = dict(acc_A=accA.tolist(), acc_B=base[rd].tolist(), acc_all=allf[rd].tolist(),
                           HG=float((aA - aB) / (1 - aB)) if aB < 1 else 0.0,
                           gain_vs_all=float((aA - aAll) / (1 - aAll)) if aAll < 1 else 0.0)
        out[name] = res
    return out, m_list


# ----------------------------------------------------------------------------- one setting
def build_explainers(ctx, names):
    ex = {}
    if "LIME" in names:
        ex["LIME"] = LimeExplainer(ctx)
    if "KernelSHAP" in names:
        ex["KernelSHAP"] = KernelShapExplainer(ctx)
    if "HermAI-0.1.8" in names:
        ex["HermAI-0.1.8"] = HermaiV018Explainer(ctx, HermaiPublicGen, "HermAI-0.1.8")
    if "HermAI-0.1.8-cov" in names:
        ex["HermAI-0.1.8-cov"] = HermaiV018Explainer(ctx, HermaiCovGen, "HermAI-0.1.8-cov")
    if "HermAI-H" in names:
        ex["HermAI-H"] = HermaiHExplainer(ctx)
    return ex


ALL_EXPLAINERS = ["LIME", "KernelSHAP", "HermAI-0.1.8", "HermAI-0.1.8-cov", "HermAI-H", "Random", "SignFlip"]


def run_setting(ctx, explainer_names=ALL_EXPLAINERS, with_readers=True, log=None):
    rng_test = np.random.default_rng(ctx.seed + 5)
    rng_rand = np.random.default_rng(ctx.seed + 6)
    exps = build_explainers(ctx, explainer_names)
    G, GA = regional_wholes(ctx)
    rows, attrs = [], {n: np.zeros((len(ctx.E), ctx.d)) for n in explainer_names}
    for r, ti in enumerate(ctx.E):
        x = ctx.Xte[ti]
        res = {}
        for n, e in exps.items():
            res[n] = e.explain(x, r)
        if "Random" in explainer_names:
            res["Random"] = random_explanation(ctx, rng_rand)
        if "SignFlip" in explainer_names and "HermAI-H" in res:
            res["SignFlip"] = signflip_explanation(res["HermAI-H"])
        for n, ex in res.items():
            a = ex["attr"]
            attrs[n][r] = a
            row = dict(explainer=n, inst=r, pred=float(ctx.f(x)[0]))
            if "accept" in ex:
                row["H_accept"], row["H_tau"] = ex["accept"], ex["tau"]
            # HP of the evidence points
            has = ex["samples"] is not None and len(ex["samples"])
            row["HP"] = float(ctx.on_horizon(ctx.z(ex["samples"])).mean()) if has else np.nan
            row["HP_dist"] = float(ctx.horizon.on(ctx.z(ex["samples"])).mean()) if has else np.nan
            row["HP_type"] = float(ctx.type_ok(ex["samples"]).mean()) if has else np.nan
            # direction claims
            cond, cfeat, marg, hp_c, hp_m, slope = [], [], [], [], [], []
            for (j, s, sem) in ex["claims"]:
                pg, hg_ = test_contrib(ctx, x, j, s, "cond", rng_test, level="group")
                pc, hc = test_contrib(ctx, x, j, s, "cond", rng_test)
                pm, hm = test_contrib(ctx, x, j, s, "marg", rng_test)
                cond.append(pg); cfeat.append(pc); marg.append(pm); hp_c.append(hg_); hp_m.append(hm)
                if sem == "slope":
                    t = test_slope(ctx, x, j, s)
                    slope.append(pc if t is None else t)
            row["n_claims"] = len(ex["claims"])
            row["claim_detail"] = ";".join(f"{j}:{int(sg)}:{int(pp)}" for (j, sg, _), pp in zip(ex["claims"], cond))
            row["CVR_dir"] = float(np.mean(cond)) if cond else np.nan
            row["CVR_dir_feat"] = float(np.mean(cfeat)) if cfeat else np.nan
            row["CVR_dir_marg"] = float(np.mean(marg)) if marg else np.nan
            row["testHP_cond"] = float(np.mean(hp_c)) if hp_c else np.nan
            row["testHP_marg"] = float(np.mean(hp_m)) if hp_m else np.nan
            row["CVR_slope"] = float(np.mean(slope)) if slope else np.nan
            if "slope" in ex:   # semantic inversion: slope sign vs contribution sign on narrated features
                js = [j for (j, s, sem) in ex["claims"]]
                row["inversion"] = float(np.mean([np.sign(ex["slope"][j]) != np.sign(ex["attr"][j]) for j in js])) if js else np.nan
            # counterfactual claim
            if ex["cf"] is not None:
                v, on, ch, l1 = test_cf(ctx, x, ex["cf"])
                row.update(cf_present=1, cf_valid=int(v), cf_on=int(on), cf_pass=int(v and on), cf_changed=ch, cf_l1=l1)
            else:
                row.update(cf_present=0)
            # rule claim
            if ex["rule"]:
                row["rule_pass"] = int(test_rule(ctx, x))
            # overall CVR on all claims that the explanation makes
            passes = list(cond)
            if ex["cf"] is not None:
                passes.append(row["cf_pass"])
            if ex["rule"]:
                passes.append(row["rule_pass"])
            row["CVR_all"] = float(np.mean(passes)) if passes else np.nan
            # part-whole agreement and horizon radius
            pw = {}
            ag = group_sum(ctx, a)
            for s in SCALES:
                pw[s] = sign_agreement(ag, group_sum(ctx, G[s][r])) * float(GA[s][r])
                row[f"PWA_{s}"] = pw[s]
            row["PWA"] = pw[1.0]
            row["PWA_feat"] = sign_agreement(a, G[1.0][r]) * float(GA[1.0][r])
            hhr = 0.0
            for s in SCALES:
                if pw[s] >= TAU_PWA - 1e-9:
                    hhr = s
                    break
            row["HHR"] = hhr
            g = G[1.0][r]
            row["cos_whole"] = float(np.dot(a, g) / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-12))
            rows.append(row)
        if log and (r + 1) % 10 == 0:
            log(f"  {r + 1}/{len(ctx.E)} instances")
    df = pd.DataFrame(rows)
    ctx._G, ctx._GA = G, GA
    readers, m_list = (reader_curves(ctx, attrs, seed=ctx.seed) if with_readers else (None, None))
    meta = dict(test_acc=ctx.test_acc, whole_fidelity=ctx.whole_fidelity, hp_real_test=ctx.hp_real_test,
                hp_real_test_dist=ctx.hp_real_test_dist, n_discrete=int(ctx.discrete.sum()),
                n_E=int(len(ctx.E)), n_R=int(len(ctx.R)), d=ctx.d, n_train=int(len(ctx.Xtr)),
                n_test=int(len(ctx.Xte)), m_list=m_list, n_groups=len(ctx.groups),
                groups=[[ctx.names[j] for j in g] for g in ctx.groups if len(g) > 1],
                GA_rate={str(s): float(GA[s].mean()) for s in SCALES})
    return df, readers, meta, attrs

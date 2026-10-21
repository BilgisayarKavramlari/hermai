# Hermeneutic Circle Test (HCT): reproduction code and results

This folder contains the code and the result files for the article *Measuring the Hermeneutic Circle in Explainable AI: A Human-Free
Test Based on Data Horizons, Executable Claims, Part–Whole Agreement and Machine Readers* (Ş. E. Şeker, under review).

HCT evaluates explanations of tabular classifiers without human raters. It reports

| Component | Question it answers |
|---|---|
| Horizon Plausibility (HP) | Do the evidence points of an explanation lie inside a conformal, scale-aware horizon of the training data? |
| Claim Verification Rate (CVR) | Do the sentences of an explanation survive executable, on-horizon intervention tests? |
| Part–Whole Agreement (PWA) and Hermeneutic Horizon Radius (HHR) | Does the account of a case agree with the account of a surrogate of the whole model, and at which context scale? |
| Hermeneutic Gain (HG) | Do explanations help machine readers learn the model, relative to a random choice of features? |

`code/hct.py` also contains **HermAI-H**, the horizon-aware HermAI explainer studied in the article. It is a research prototype and is
not yet part of the `hermai` package API.

## Files

| Path | Purpose |
|---|---|
| `code/hct.py` | Reference implementation of HP, CVR, PWA/HHR and HG, the HermAI-H explainer, and wrappers for LIME, KernelSHAP and hermai 0.1.8 |
| `code/run_experiments.py` | Main grid, planted local exception, label randomisation and scaffold attack |
| `code/audit_consistency.py` | Run-to-run consistency of the explainers, over features and over meaning units |
| `code/example_narrative.py` | Worked example on one Pima Diabetes case |
| `code/aggregate_and_plot.py` | Builds `results/summary.json` and the figures (written to `figures/`) |
| `code/baselines.py` | Established metrics on the same explanations that HCT scored: prediction gap on important features (PGI), deletion curves and faithfulness correlation, each with marginal or training-mean and conditional replacement, and the disagreement metrics of Krishna et al. |
| `code/roar.py` | Remove-and-retrain (ROAR) on the Pima and Breast Cancer settings |
| `code/adaptive_attack.py` | Scaffold attacks A0–A3 on COMPAS: the original attack, a horizon-aware, a test-aware and a memorising attacker |
| `code/revision_analysis.py` | Builds `results/revision_summary.json` and the revised Figure 1 and Figure 6 |
| `code/run_revision_jobs.sh` | Runs the `baselines.py` and `roar.py` jobs of the article |
| `code/get_data.sh` | Downloads the public datasets into `data/` |
| `code/jobs_all.txt` | The exact list of `run_experiments.py` runs used for the article |
| `results/` | Per-instance and per-setting result files; `summary.json` and `revision_summary.json` hold every number reported in the article |

## Environment

Python 3.13, scikit-learn 1.9.1, numpy 2.x, pandas 3.x, scipy, shap 0.52.0, lime 0.2.0.1, hermai 0.1.8 and matplotlib. lime 0.2.0.1
is pure Python; if its setup script fails under a recent setuptools, placing the `lime` package folder from its source distribution on
the Python path is enough (for `run_revision_jobs.sh`, set `LIME_SRC` to that folder).

## Reproduce

```bash
cd hct/code
sh get_data.sh                                  # Pima, COMPAS and Adult; Breast Cancer comes with scikit-learn
xargs -L 1 python run_experiments.py < jobs_all.txt
python audit_consistency.py pima rf 25
python audit_consistency.py breast rf 25
python example_narrative.py
python aggregate_and_plot.py                    # results/summary.json and figures/
# analyses added in the revision
sh run_revision_jobs.sh baselines               # baselines.py for {breast,pima,compas,adult} x {rf,mlp}
sh run_revision_jobs.sh roar                    # roar.py for {pima,breast} x {rf,mlp}
for a in A0 A1 A2 A3; do python adaptive_attack.py $a 0; done
python revision_analysis.py                     # results/revision_summary.json, Figures 1 and 6
```

The original runs take about 1.5 CPU-hours and the revision adds about 0.6 CPU-hours. `adaptive_attack.py A0` reproduces the
original scaffold experiment exactly (same claims and outcomes as `results/adversarial_s0.csv`). Run `revision_analysis.py` after
`aggregate_and_plot.py`, because it replaces the Figure 1 of the original submission with the revised two-panel figure.

The result files in `results/` are the ones used for the article, so `python aggregate_and_plot.py` followed by
`python revision_analysis.py` regenerates both summary files and all figures without rerunning any experiment.

## Data

All datasets are public: Breast Cancer Wisconsin (Diagnostic) as distributed with scikit-learn; the Pima Indians Diabetes data
(768-record version); the COMPAS two-year recidivism data released by ProPublica; and the UCI Adult data. They are downloaded by
`get_data.sh` and are not stored in this repository.

## License

MIT, as for the rest of this repository.

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
| `code/get_data.sh` | Downloads the public datasets into `data/` |
| `code/jobs_all.txt` | The exact list of runs used for the article |
| `results/` | Per-instance and per-setting result files; `summary.json` holds every number reported in the article |

## Environment

Python 3.13, scikit-learn 1.9.1, numpy 2.x, pandas 3.x, shap 0.52.0, lime 0.2.0.1, hermai 0.1.8 and matplotlib. lime 0.2.0.1 is pure
Python; if its setup script fails under a recent setuptools, placing the `lime` package folder from its source distribution on the
Python path is enough.

## Reproduce

```bash
cd hct/code
sh get_data.sh                                  # Pima, COMPAS and Adult; Breast Cancer comes with scikit-learn
xargs -L 1 python run_experiments.py < jobs_all.txt
python audit_consistency.py pima rf 25
python audit_consistency.py breast rf 25
python example_narrative.py
python aggregate_and_plot.py                    # results/summary.json and figures/
```

The full set of runs takes about 1.5 CPU-hours. The result files in `results/` are the ones used for the article, so
`python aggregate_and_plot.py` alone regenerates `summary.json` and all figures.

## Data

All datasets are public: Breast Cancer Wisconsin (Diagnostic) as distributed with scikit-learn; the Pima Indians Diabetes data
(768-record version); the COMPAS two-year recidivism data released by ProPublica; and the UCI Adult data. They are downloaded by
`get_data.sh` and are not stored in this repository.

## License

MIT, as for the rest of this repository.

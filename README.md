# Hermai: A Hermeneutic Explainable AI Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/hermai.svg)](https://pypi.org/project/hermai/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24596854.svg)](https://doi.org/10.5281/zenodo.24596854)

Hermai is a Python library for explainable AI (XAI) that focuses on providing contextual, narrative, and actionable explanations for machine learning models. Inspired by hermeneutics, it aims to create a dialogue between the user and the model's decision-making process.
<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/BilgisayarKavramlari/hermai@main/hermai_wide.png" alt="Hermai Logo" width="300"/>
</p>


## Key Features

- **Contextual Perturbations:** Generates realistic "what-if" scenarios that respect the correlations in your data.
- **Local Explanations:** Explains individual predictions by identifying key feature contributions and generating counterfactuals.
- **General Explanations:** Approximates the overall behavior of a complex model with a simple, interpretable surrogate model (like a decision tree).
- **Narrative Summaries:** Presents explanations in human-readable text.

## Installation

Install from PyPI:

```bash
pip install hermai
```

## Quick Usage

Here's a quick example of how to get local and general explanations for a RandomForest model trained on the Titanic dataset.

```python
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hermai.perturbations.tabular import TabularPerturbationGenerator
from hermai.explainers import LocalExplainer, GeneralExplainer

# Load and prepare data (assuming X_train, black_box_model are ready)
# ...

# --- Local Explanation ---
generator = TabularPerturbationGenerator(categorical_features=['pclass', 'sex', 'embarked'])
generator.fit(X_train)
local_explainer = LocalExplainer(black_box_model, generator)
instance = X_test.iloc[0]
local_explanation = local_explainer.explain(instance)
print(local_explanation.narrative())
local_explanation.plot()


# --- General Model Explanation ---
general_explainer = GeneralExplainer(black_box_model)
general_explanation = general_explainer.explain(X_train)
general_explanation.plot_feature_importance()
general_explanation.plot_surrogate_tree()
```

## How to read hermai's narrative

A sentence such as *"Feature 'Glucose' increases the probability"* describes the **local slope** of hermai's surrogate
model: a slightly higher value of the feature would raise the predicted probability. It does **not** say that the
case's value raises the probability compared with a typical case. The two readings can disagree, for example when the
case's value is below average and the slope is positive. The Hermeneutic Circle Test (below) and the companion
package [xai-claims](https://github.com/BilgisayarKavramlari/xai-claims) make this difference measurable.

## Testing what hermai says with xai-claims

[xai-claims](https://github.com/BilgisayarKavramlari/xai-claims) (`pip install xai-claims`) compiles explanation
outputs into explicit claims and tests every claim by intervening on the model, under explicit readings: `REF`
(compared with typical cases), `COND` (compared with comparable cases), `SLOPE` (local sensitivity) and `BIN`
(compared with another interval). It reads hermai's narrative directly:

```python
import xai_claims as xc

explanation = LocalExplainer(model, generator).explain(instance)
narrative = explanation.narrative()

claims = xc.from_hermai(narrative, list(X_train.columns), k=3)     # the first three narrative sentences
verifier = xc.Verifier(model, X_train, seed=0)
report = verifier.verify(instance, claims)
print(report[["feature", "claimed_sign", "SLOPE_verdict", "REF_verdict", "COND_verdict", "holds_native"]])

# the counterfactual: does it change the class, and does it lie within the data?
if explanation.counterfactual is not None:
    print(verifier.verify(instance, [xc.CounterfactualClaim(explanation.counterfactual[X_train.columns], "hermai")]))
```

`holds_native` checks the sentence in the reading hermai intends (`SLOPE`); the other columns show whether the same
sentence would also be true in the sense a reader may take it. hermai produces explanations; xai-claims is an
independent verifier that treats hermai like any other explanation library.

## Hermeneutic Circle Test (HCT)

The folder [`hct/`](hct/) contains the code and results of the Hermeneutic Circle Test, a human-free evaluation of
explanations based on data horizons (HP), executable claims (CVR), part–whole agreement (PWA, HHR) and machine
readers (HG), including the HermAI-H prototype. It reproduces the results of the MAKE article below; see
[`hct/README.md`](hct/README.md). The HCT code is research code and is not part of the installed package.

## Publications

If you use hermai, please cite the software (https://doi.org/10.5281/zenodo.24596854, all versions) and:

- Ş. E. Şeker and Z. Arslanhan (2026). Hermai: A Hermeneutic Framework for Context-Aware and Narrative Explainable AI.
  *Artificial Intelligence Theory and Applications*, 6(1), 56–74.
- Ş. E. Şeker (2026). Measuring the Hermeneutic Circle in Explainable AI: A Human-Free Test Based on Data Horizons,
  Executable Claims, Part–Whole Agreement, and Machine Readers. *Machine Learning and Knowledge Extraction*, 8(11), 381.
  https://doi.org/10.3390/make8110381

## Related software

- [xai-claims](https://github.com/BilgisayarKavramlari/xai-claims): verifies the claims of explanation libraries,
  including hermai, under explicit readings ([DOI 10.5281/zenodo.24593226](https://doi.org/10.5281/zenodo.24593226)).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

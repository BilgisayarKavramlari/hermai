# hermai/visualizations/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

def plot_local_feature_importance(explanation, ax=None, show=True):
    top_features = explanation.feature_importances.head(10)
    top_features = top_features[top_features['importance'].abs() > 1e-6]
    top_features = top_features.sort_values(by='importance', ascending=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = ['#d65f5f' if x < 0 else '#5f8ad6' for x in top_features['importance']]
    ax.barh(top_features['feature'], top_features['importance'], color=colors)
    ax.set_xlabel("Contribution to Prediction Probability")
    ax.set_title("Local Feature Importance")
    
    if show:
        plt.tight_layout()
        plt.show()

def plot_general_feature_importance(explanation, ax=None, show=True):
    """Plots a bar chart for general feature importances."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.barplot(x='importance', y='feature', data=explanation.feature_importances.head(15), palette='viridis',
                hue='feature', legend=False, dodge=False, ax=ax)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_ylabel("Feature")
    ax.set_title("General Feature Importance (from Surrogate Model)")

    if show:
        plt.tight_layout()
        plt.show()

def plot_surrogate_tree(explanation, ax=None, show=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        explanation.surrogate_model,
        feature_names=explanation.feature_importances['feature'].tolist(),
        class_names=[str(c) for c in explanation.surrogate_model.classes_],
        filled=True,
        rounded=True,
        fontsize=10
    )
    ax.set_title("Surrogate Decision Tree Approximating the Model")
    if show:
        plt.tight_layout()
        plt.show()
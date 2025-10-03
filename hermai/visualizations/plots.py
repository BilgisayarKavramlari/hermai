# hermai/visualizations/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

def plot_local_feature_importance(explanation):
    # ... (Bu fonksiyonda değişiklik yok) ...
    top_features = explanation.feature_importances.head(10)
    top_features = top_features[top_features['importance'].abs() > 1e-6]
    top_features = top_features.sort_values(by='importance', ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ['#d65f5f' if x < 0 else '#5f8ad6' for x in top_features['importance']]
    plt.barh(top_features['feature'], top_features['importance'], color=colors)
    plt.xlabel("Contribution to Prediction Probability")
    plt.title("Local Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_general_feature_importance(explanation): # <-- DEĞİŞTİ
    """Plots a bar chart for general feature importances."""
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=explanation.feature_importances.head(15), palette='viridis',
                hue='feature', legend=False, dodge=False)
    plt.xlabel("Feature Importance (Gini)")
    plt.ylabel("Feature")
    plt.title("General Feature Importance (from Surrogate Model)") # <-- DEĞİŞTİ
    plt.tight_layout()
    plt.show()
    
def plot_surrogate_tree(explanation):
    # ... (Bu fonksiyonda değişiklik yok) ...
    plt.figure(figsize=(20, 10))
    plot_tree(
        explanation.surrogate_model,
        feature_names=explanation.feature_importances['feature'].tolist(),
        class_names=[str(c) for c in explanation.surrogate_model.classes_],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Surrogate Decision Tree Approximating the Model")
    plt.show()
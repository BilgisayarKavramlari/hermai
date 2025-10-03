# examples/example_titanic_v2.py
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import the new, modular Hermai library
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hermai.perturbations.tabular import TabularPerturbationGenerator
from hermai.explainers import LocalExplainer, GeneralExplainer # <-- Daha temiz import

# --- 1. Data Loading and Preparation ---
print("🚢 1. Loading and preparing the Titanic dataset...")
df = sns.load_dataset('titanic').drop(['deck', 'embark_town', 'alive', 'who', 'adult_male', 'alone', 'class'], axis=1)
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.dropna(inplace=True)
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['embarked'] = LabelEncoder().fit_transform(df['embarked'])
X, y = df.drop('survived', axis=1), df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Black-Box Model Training ---
print("\n🤖 2. Training a RandomForestClassifier model...")
black_box_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
print(f"   - Model accuracy: {black_box_model.score(X_test, y_test):.2f}")

# --- 3. LOCAL EXPLANATION ---
print("\n\n" + "="*50)
print("🔬 3. GENERATING LOCAL EXPLANATION")
print("="*50)
generator = TabularPerturbationGenerator(categorical_features=['pclass', 'sex', 'embarked'])
generator.fit(X_train)
local_explainer = LocalExplainer(black_box_model, generator)
instance = X_test.iloc[1]
print("\n👤 Explaining prediction for instance:")
print(instance)
local_explanation = local_explainer.explain(instance)
print("\n--- Explanation Narrative ---")
print(local_explanation.narrative())
print("\n--- Feature Contribution Plot ---")
local_explanation.plot()

# --- 4. GENERAL MODEL EXPLANATION ---  <-- DEĞİŞTİ
print("\n\n" + "="*50)
print("🌍 4. GENERATING GENERAL MODEL EXPLANATION") # <-- DEĞİŞTİ
print("="*50)
# Setup the general explainer
general_explainer = GeneralExplainer(black_box_model) # <-- DEĞİŞTİ

# Explain the model's overall behavior using the training data
general_explanation = general_explainer.explain(X_train) # <-- DEĞİŞTİ

# Show general importance and the decision tree approximation
print("\n--- General Feature Importance Plot ---") # <-- DEĞİŞTİ
general_explanation.plot_feature_importance()
print("\n--- Approximated Decision Rules (Surrogate Tree) ---")
print("This tree shows the main decision paths the complex RandomForest model appears to follow.")
general_explanation.plot_surrogate_tree()
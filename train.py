"""
Diabetes Prediction Model — Training Script
============================================
Dataset: Pima Indians Diabetes Dataset (768 rows × 9 columns)
Target : Outcome (0 = No Diabetes, 1 = Diabetes)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import joblib
import os

# ─────────────────────────────────────────────
# STEP 1 — Load the data
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv("diabetes.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# ─────────────────────────────────────────────
# STEP 2 — Handle biologically impossible zeros
# ─────────────────────────────────────────────
# WHY: Certain columns (Glucose, BloodPressure, SkinThickness,
# Insulin, BMI) cannot be zero in a living person.
# These zeros are MISSING VALUES recorded as 0.
# If we leave them, the model learns from fake numbers.
# FIX: Replace 0s with NaN, then impute with the column MEDIAN.
# We use MEDIAN (not mean) because it is robust to outliers.
print("\n" + "=" * 60)
print("STEP 2: Handling biologically impossible zero values")
print("=" * 60)

cols_with_impossible_zeros = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
for col in cols_with_impossible_zeros:
    zeros = (df[col] == 0).sum()
    print(f"  {col}: {zeros} zeros → replacing with median")
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

print("After imputation — null count:", df.isnull().sum().sum())

# ─────────────────────────────────────────────
# STEP 3 — Split features and target
# ─────────────────────────────────────────────
# WHY: We separate what we're predicting (Outcome) from the
# features the model will use to make that prediction.
print("\n" + "=" * 60)
print("STEP 3: Splitting features (X) and target (y)")
print("=" * 60)

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COLUMN = "Outcome"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ─────────────────────────────────────────────
# STEP 4 — Train / Test split
# ─────────────────────────────────────────────
# WHY: We need unseen data to honestly evaluate the model.
# 80% train, 20% test is a common split.
# stratify=y ensures both splits have the same class ratio
# (important because our dataset is imbalanced: 65% No / 35% Yes).
# random_state=42 makes the split reproducible.
print("\n" + "=" * 60)
print("STEP 4: Train / Test split  (80% / 20%, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
# STEP 5 — Feature Scaling (Normalization)
# ─────────────────────────────────────────────
# WHY: Features have very different scales:
#   - Insulin can be up to 846
#   - DiabetesPedigreeFunction is 0.08–2.42
#   - Pregnancies is 0–17
# Without scaling, models that rely on distances or gradients
# are dominated by large-valued features.
#
# HOW — StandardScaler (Z-score normalisation):
#   scaled_value = (value − mean) / std_dev
# After scaling every feature has mean=0 and std=1.
#
# CRITICAL RULE: Fit the scaler ONLY on training data.
# Then use that same fitted scaler to transform test data AND
# future prediction inputs.
# If we fit on all data first, we "leak" test information into
# training (data leakage), giving falsely optimistic results.
#
# AT PREDICTION TIME: apply the exact same fitted scaler to the
# incoming values before feeding them to the model.
# The scaler is saved to disk alongside the model so it can be
# loaded in the API.
print("\n" + "=" * 60)
print("STEP 5: Feature scaling with StandardScaler")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on TRAIN
X_test_scaled  = scaler.transform(X_test)         # transform only on TEST

print("Scaler fitted on training data only (no data leakage).")
print("Sample means after scaling (should be ≈ 0):", X_train_scaled.mean(axis=0).round(2))
print("Sample stds  after scaling (should be ≈ 1):", X_train_scaled.std(axis=0).round(2))

# ─────────────────────────────────────────────
# STEP 6 — Choose and train the model
# ─────────────────────────────────────────────
# WHY Random Forest?
#   • Handles non-linear relationships between features
#   • Naturally resistant to overfitting (ensemble of trees)
#   • Gives feature importance for interpretability
#   • Works well on tabular medical data without heavy tuning
#
# Key hyperparameters:
#   n_estimators=200  — 200 decision trees; more trees = more stable,
#                       diminishing returns after ~200
#   max_depth=10      — limits tree depth to prevent overfitting
#   min_samples_split=5  — a node must have ≥5 samples to split
#   class_weight='balanced'  — compensates for class imbalance
#                             (500 non-diabetic vs 268 diabetic)
#   random_state=42   — reproducibility
print("\n" + "=" * 60)
print("STEP 6: Training Random Forest Classifier")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("Model trained successfully.")

# ─────────────────────────────────────────────
# STEP 7 — Evaluate the model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Model Evaluation")
print("=" * 60)

y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy : {acc:.4f}  ({acc*100:.1f}%)")
print(f"ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Cross-validation for reliability
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="roc_auc")
print(f"\n5-Fold Cross-Validation ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
print("\nFeature Importances:")
fi = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
for feat, imp in fi.items():
    print(f"  {feat:30s}: {imp:.4f}")

# ─────────────────────────────────────────────
# STEP 8 — Save model and scaler
# ─────────────────────────────────────────────
# WHY: We save BOTH the trained model AND the fitted scaler.
# At prediction time we load both, apply scaler.transform() to
# the raw input, then call model.predict() on the scaled input.
print("\n" + "=" * 60)
print("STEP 8: Saving model and scaler to ./model/")
print("=" * 60)

os.makedirs("model", exist_ok=True)
joblib.dump(model,  "model/diabetes_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(FEATURE_COLUMNS, "model/feature_columns.pkl")

print("Saved: model/diabetes_model.pkl")
print("Saved: model/scaler.pkl")
print("Saved: model/feature_columns.pkl")
print("\nTraining complete!")

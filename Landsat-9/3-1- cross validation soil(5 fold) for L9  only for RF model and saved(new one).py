# -*- coding: utf-8 -*-
"""
Created on Fri May 30 21:41:54 2025

@author: alimo
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import os

# Load Landsat 9 Data
file_path = 'GroundTruth_Landsat9.csv'
data = pd.read_csv(file_path)

# Select relevant features and targets for Landsat 9
selected_features_clay = ['B4_L9', 'B5_L9', 'NDVI_L9', 'SAVI_L9']
selected_features_sand = ['B5_L9', 'B7_L9', 'EVI_L9', 'NDVI_L9', 'SAVI_L9']

clay_targets = ['clay_0_5', 'clay_5_15']
sand_targets = ['sand_0_5', 'sand_5_15']

# Create output directory for feature importance plots
os.makedirs("feature_importance_plots", exist_ok=True)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

# Function to plot and save feature importances
def plot_feature_importance(importances, feature_names, title, filename):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.bar(range(len(importances)), np.array(importances)[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45)
    plt.tight_layout()
    plt.savefig(f"feature_importance_plots/{filename}.png", dpi=300)
    plt.close()

# Prepare data for Clay
for clay_target in clay_targets:
    data_clay = data.dropna(subset=selected_features_clay + [clay_target])
    X_clay = data_clay[selected_features_clay]
    y_clay = data_clay[clay_target] > data_clay[clay_target].median()

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_clay_balanced, y_clay_balanced = smote.fit_resample(X_clay, y_clay)

    print(f"\n--- Clay Classification Results ({clay_target}) ---")
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_list = []
        kappa_list = []
        cm_total = np.zeros((2, 2))

        for train_idx, test_idx in skf.split(X_clay_balanced, y_clay_balanced):
            X_train, X_test = X_clay_balanced.iloc[train_idx], X_clay_balanced.iloc[test_idx]
            y_train, y_test = y_clay_balanced.iloc[train_idx], y_clay_balanced.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_list.append(accuracy_score(y_test, y_pred))
            kappa_list.append(cohen_kappa_score(y_test, y_pred))
            cm_total += confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
        print(f"Kappa: {np.mean(kappa_list):.4f} ± {np.std(kappa_list):.4f}")
        print("Confusion Matrix:\n", cm_total)

        # Feature Importance for RF
        if model_name == 'Random Forest':
            importances = model.feature_importances_
            print(f"\nFeature Importances for {clay_target} ({model_name}):")
            for f_name, f_value in zip(X_train.columns, importances):
                print(f"{f_name}: {f_value:.4f}")
            plot_feature_importance(
                importances,
                X_train.columns,
                f"{clay_target} - {model_name} Feature Importance",
                f"{clay_target}_{model_name.replace(' ', '_')}_feature_importance"
            )

# Prepare data for Sand
for sand_target in sand_targets:
    data_sand = data.dropna(subset=selected_features_sand + [sand_target])
    X_sand = data_sand[selected_features_sand]
    y_sand = data_sand[sand_target] > data_sand[sand_target].median()

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_sand_balanced, y_sand_balanced = smote.fit_resample(X_sand, y_sand)

    print(f"\n--- Sand Classification Results ({sand_target}) ---")
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_list = []
        kappa_list = []
        cm_total = np.zeros((2, 2))

        for train_idx, test_idx in skf.split(X_sand_balanced, y_sand_balanced):
            X_train, X_test = X_sand_balanced.iloc[train_idx], X_sand_balanced.iloc[test_idx]
            y_train, y_test = y_sand_balanced.iloc[train_idx], y_sand_balanced.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_list.append(accuracy_score(y_test, y_pred))
            kappa_list.append(cohen_kappa_score(y_test, y_pred))
            cm_total += confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
        print(f"Kappa: {np.mean(kappa_list):.4f} ± {np.std(kappa_list):.4f}")
        print("Confusion Matrix:\n", cm_total)

        # Feature Importance for RF
        if model_name == 'Random Forest':
            importances = model.feature_importances_
            print(f"\nFeature Importances for {sand_target} ({model_name}):")
            for f_name, f_value in zip(X_train.columns, importances):
                print(f"{f_name}: {f_value:.4f}")
            plot_feature_importance(
                importances,
                X_train.columns,
                f"{sand_target} - {model_name} Feature Importance",
                f"{sand_target}_{model_name.replace(' ', '_')}_feature_importance"
            )

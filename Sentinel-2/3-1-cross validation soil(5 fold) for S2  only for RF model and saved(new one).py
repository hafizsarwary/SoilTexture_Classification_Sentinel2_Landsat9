# -*- coding: utf-8 -*-
"""
Created on Fri May 30 21:50:13 2025

@author: alimo
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# Load Sentinel-2 Data
file_path = 'GroundTruth_Sentinel2.csv'
data = pd.read_csv(file_path)

# Select relevant features and targets for Sentinel-2
selected_features_clay = ['B7', 'B6', 'B5', 'B8', 'IRECI_S2', 'NDVI_S2', 'SAVI_S2']
selected_features_sand = ['B7', 'B6', 'B5', 'B4', 'MCARI_S2', 'IRECI_S2', 'NDVI_S2', 'SAVI_S2']

clay_targets = ['clay_0_5', 'clay_5_15']
sand_targets = ['sand_0_5', 'sand_5_15']

# Create output directory for feature importance plots
os.makedirs("feature_importance_plots_s2", exist_ok=True)

# Plotting function
def plot_feature_importance(importances, feature_names, title, filename):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.bar(range(len(importances)), np.array(importances)[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45)
    plt.tight_layout()
    plt.savefig(f"feature_importance_plots_s2/{filename}.png", dpi=300)
    plt.close()

# Define the Random Forest model
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

# Train and Save Model for Clay
for clay_target in clay_targets:
    print(f"\n--- Training and Saving Model for Clay ({clay_target}) ---")
    data_clay = data.dropna(subset=selected_features_clay + [clay_target])
    X_clay = data_clay[selected_features_clay]
    y_clay = data_clay[clay_target] > data_clay[clay_target].median()

    smote = SMOTE(random_state=42)
    X_clay_balanced, y_clay_balanced = smote.fit_resample(X_clay, y_clay)

    for model_name, model in models.items():
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

        # Feature Importance only for Random Forest
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

        clay_results = {
            'model': model,
            'accuracy': accuracy_list,
            'kappa': kappa_list,
            'confusion_matrix': cm_total,
            'features': selected_features_clay,
            'target': clay_target
        }
        with open(f'clay_model_{clay_target}.pkl', 'wb') as clay_model_file:
            pickle.dump(clay_results, clay_model_file)
        print(f"Clay model and metrics saved as 'clay_model_{clay_target}.pkl'")

# Train and Save Model for Sand
for sand_target in sand_targets:
    print(f"\n--- Training and Saving Model for Sand ({sand_target}) ---")
    data_sand = data.dropna(subset=selected_features_sand + [sand_target])
    X_sand = data_sand[selected_features_sand]
    y_sand = data_sand[sand_target] > data_sand[sand_target].median()

    smote = SMOTE(random_state=42)
    X_sand_balanced, y_sand_balanced = smote.fit_resample(X_sand, y_sand)

    for model_name, model in models.items():
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

        # Feature Importance only for Random Forest
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

        sand_results = {
            'model': model,
            'accuracy': accuracy_list,
            'kappa': kappa_list,
            'confusion_matrix': cm_total,
            'features': selected_features_sand,
            'target': sand_target
        }
        with open(f'sand_model_{sand_target}.pkl', 'wb') as sand_model_file:
            pickle.dump(sand_results, sand_model_file)
        print(f"Sand model and metrics saved as 'sand_model_{sand_target}.pkl'")

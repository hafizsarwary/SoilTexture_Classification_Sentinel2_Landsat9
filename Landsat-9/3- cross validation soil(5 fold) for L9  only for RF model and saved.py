# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:06:39 2025

@author: alimo
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle

# Load Landsat 9 Data
file_path = 'GroundTruth_Landsat9.csv'
data = pd.read_csv(file_path)

# Select relevant features and targets for Landsat 9
selected_features_clay = ['B4_L9', 'B5_L9', 'NDVI_L9', 'SAVI_L9']
selected_features_sand = ['B5_L9', 'B7_L9', 'EVI_L9', 'NDVI_L9', 'SAVI_L9']

clay_targets = ['clay_0_5', 'clay_5_15']
sand_targets = ['sand_0_5', 'sand_5_15']

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

# Train and Save Model for Clay
for clay_target in clay_targets:
    print(f"\n--- Training and Saving Model for Clay ({clay_target}) ---")
    data_clay = data.dropna(subset=selected_features_clay + [clay_target])
    X_clay = data_clay[selected_features_clay]
    y_clay = data_clay[clay_target] > data_clay[clay_target].median()
    
    # Apply SMOTE to balance classes
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

        # Save all training details and model
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
    
    # Apply SMOTE to balance classes
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

        # Save all training details and model
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

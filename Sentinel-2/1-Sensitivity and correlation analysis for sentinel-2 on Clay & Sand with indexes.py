# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:11:34 2025

@author: alimo
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Sentinel-2 data from CSV
file_path = 'GroundTruth_Sentinel2.csv'
data = pd.read_csv(file_path)

# Select relevant columns for Sentinel-2 using recommended bands and indices
columns_of_interest = [
    'clay_0_5', 'clay_5_15', 'sand_0_5', 'sand_5_15',
    'B3', 'B4', 'B5', 'B6', 'B7', 'B8',  # Bands used for indices
    'EVI_S2', 'NDVI_S2', 'SAVI_S2',      # Common indices
    'MCARI_S2', 'IRECI_S2', 'MTCI_S2', 'S2REP_S2'  # Red-edge indices
]
data_selected = data[columns_of_interest].dropna()

# Function to perform regression analysis and calculate R², P-value, and plot with equation
def regression_and_stats(x_col, y_col, data):
    X = data[[x_col]].dropna()
    y = data[y_col][X.index]

    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Calculate Pearson correlation and P-value
    r, p_value = pearsonr(X[x_col], y)

    # Regression equation
    slope = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {slope:.2f}x + {intercept:.2f}"

    # Plot Regression
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X[x_col], y=y, label='Data')
    plt.plot(X[x_col], y_pred, color='red', label=f'Regression Line\n{equation}\nR² = {r2:.2f}, P = {p_value:.5f}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Regression and Correlation between {x_col} and {y_col}')
    plt.legend()
    plt.show()

    return r2, p_value

# Define predictors for Sentinel-2 (bands + indices)
predictors_sentinel2 = [
    'B3', 'B4', 'B5', 'B6', 'B7', 'B8',  # Bands
    'EVI_S2', 'NDVI_S2', 'SAVI_S2',      # Common indices
    'MCARI_S2', 'IRECI_S2', 'MTCI_S2', 'S2REP_S2'  # Red-edge indices
]

results_clay = []
results_sand = []

# Sentinel-2 analysis for Clay content
print("\n--- Sentinel-2 Analysis for Clay ---")
for predictor in predictors_sentinel2:
    for clay_target in ['clay_0_5', 'clay_5_15']:
        r2, p_value = regression_and_stats(predictor, clay_target, data_selected)
        results_clay.append({'Predictor': predictor, 'Target': clay_target, 'R²': r2, 'P-value': p_value})

# Sentinel-2 analysis for Sand content
print("\n--- Sentinel-2 Analysis for Sand ---")
for predictor in predictors_sentinel2:
    for sand_target in ['sand_0_5', 'sand_5_15']:
        r2, p_value = regression_and_stats(predictor, sand_target, data_selected)
        results_sand.append({'Predictor': predictor, 'Target': sand_target, 'R²': r2, 'P-value': p_value})

# Convert results to DataFrame
results_clay_df = pd.DataFrame(results_clay)
results_sand_df = pd.DataFrame(results_sand)

# Display results for Clay
print("\nSummary of R² and P-values for Clay (Sentinel-2):")
print(results_clay_df)

# Display results for Sand
print("\nSummary of R² and P-values for Sand (Sentinel-2):")
print(results_sand_df)

# Analyze correlation between Clay and Sand
print("\n--- Correlation Analysis between Clay and Sand ---")
for clay_col, sand_col in [('clay_0_5', 'sand_0_5'), ('clay_5_15', 'sand_5_15')]:
    correlation = data_selected[clay_col].corr(data_selected[sand_col])
    print(f"Correlation between {clay_col} and {sand_col}: {correlation:.2f}")

    # Scatter plot for Clay and Sand
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=clay_col, y=sand_col, data=data_selected)
    plt.title(f"Scatter Plot between {clay_col} and {sand_col} (Correlation: {correlation:.2f})")
    plt.xlabel("Clay Content")
    plt.ylabel("Sand Content")
    plt.show()

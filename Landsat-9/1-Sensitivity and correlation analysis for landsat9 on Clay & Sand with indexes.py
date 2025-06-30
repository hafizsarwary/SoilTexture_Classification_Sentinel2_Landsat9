# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:16:27 2025

@author: alimo
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Landsat 9 data from CSV
file_path = 'GroundTruth_Landsat9.csv'
data = pd.read_csv(file_path)

# Select relevant columns for Landsat 9 using recommended bands and indices
columns_of_interest = [
    'clay_0_5', 'clay_5_15', 'sand_0_5', 'sand_5_15',
    'B4_L9', 'B5_L9', 'B6_L9', 'B7_L9', 'EVI_L9', 'NDVI_L9', 'SAVI_L9'
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

# Define predictors for Landsat 9
predictors_landsat9 = ['B4_L9', 'B5_L9', 'B6_L9', 'B7_L9', 'EVI_L9', 'NDVI_L9', 'SAVI_L9']

results_clay = []
results_sand = []

# Landsat 9 analysis for Clay content
print("\n--- Landsat 9 Analysis for Clay ---")
for predictor in predictors_landsat9:
    for clay_target in ['clay_0_5', 'clay_5_15']:
        r2, p_value = regression_and_stats(predictor, clay_target, data_selected)
        results_clay.append({'Predictor': predictor, 'Target': clay_target, 'R²': r2, 'P-value': p_value})

# Landsat 9 analysis for Sand content
print("\n--- Landsat 9 Analysis for Sand ---")
for predictor in predictors_landsat9:
    for sand_target in ['sand_0_5', 'sand_5_15']:
        r2, p_value = regression_and_stats(predictor, sand_target, data_selected)
        results_sand.append({'Predictor': predictor, 'Target': sand_target, 'R²': r2, 'P-value': p_value})

# Convert results to DataFrame
results_clay_df = pd.DataFrame(results_clay)
results_sand_df = pd.DataFrame(results_sand)

# Display results for Clay
print("\nSummary of R² and P-values for Clay (Landsat 9):")
print(results_clay_df)

# Display results for Sand
print("\nSummary of R² and P-values for Sand (Landsat 9):")
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

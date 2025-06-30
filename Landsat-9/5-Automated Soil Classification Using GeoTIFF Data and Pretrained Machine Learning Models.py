# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:42:41 2025

@author: alimo
"""

import rasterio
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Input Landsat-9 GeoTIFF file
input_image_path = "Landsat9_Bands_and_Indices.tif"

# Pre-trained model files (pickle)
clay_model_0_5_path = "clay_model_clay_0_5.pkl"
clay_model_5_15_path = "clay_model_clay_5_15.pkl"
sand_model_0_5_path = "sand_model_sand_0_5.pkl"
sand_model_5_15_path = "sand_model_sand_5_15.pkl"

# Step 1: Read the GeoTIFF image
with rasterio.open(input_image_path) as dataset:
    bands = [dataset.read(i) for i in range(1, dataset.count + 1)]  # Read all bands and indices
    meta = dataset.meta.copy()  # Extract metadata of the image
    transform = dataset.transform  # Extract georeferencing transform
    crs = dataset.crs  # Extract coordinate reference system

# Convert bands to a 3D array
bands_array = np.stack(bands, axis=-1)
rows, cols, n_features = bands_array.shape  # Extract dimensions of the image

# Flatten the 3D array into a 2D array (pixels × features)
pixels_array = bands_array.reshape(-1, n_features)

# Define correct feature names for Landsat-9 based on actual order in TIF
band_names = ["SR_B1", "SR_B2", "SR_B3", "B4_L9", "B5_L9", "SR_B6", "B7_L9", "NDVI_L9", "SAVI_L9", "EVI_L9"]

# Convert array to DataFrame with correct band names
data_df = pd.DataFrame(pixels_array, columns=band_names)

# 🔍 **Step 2: Data Cleaning**
data_df = data_df.fillna(data_df.median())  # جایگزینی NaN با میانه‌ی هر باند
lower_bound = data_df.quantile(0.01)
upper_bound = data_df.quantile(0.99)
data_df = data_df.clip(lower=lower_bound, upper=upper_bound, axis=1)

# Ensure only the required columns are used for model input
expected_features = ["B4_L9", "B5_L9", "B7_L9", "NDVI_L9", "SAVI_L9", "EVI_L9"]
data_df = data_df[expected_features]

# Step 3: Load pre-trained models
def load_model(model_path):
    with open(model_path, "rb") as model_file:
        model_data = pickle.load(model_file)
    return model_data["model"], model_data["features"]

clay_model_0_5, clay_features_0_5 = load_model(clay_model_0_5_path)
clay_model_5_15, clay_features_5_15 = load_model(clay_model_5_15_path)
sand_model_0_5, sand_features_0_5 = load_model(sand_model_0_5_path)
sand_model_5_15, sand_features_5_15 = load_model(sand_model_5_15_path)

# Step 4: Predict Soil
def predict_soil(model, features, data):
    inputs = data[features]
    inputs = inputs.clip(lower=data_df[features].min(), upper=data_df[features].max(), axis=1)
    return model.predict(inputs)

clay_pred_0_5 = predict_soil(clay_model_0_5, clay_features_0_5, data_df)
clay_pred_5_15 = predict_soil(clay_model_5_15, clay_features_5_15, data_df)
sand_pred_0_5 = predict_soil(sand_model_0_5, sand_features_0_5, data_df)
sand_pred_5_15 = predict_soil(sand_model_5_15, sand_features_5_15, data_df)

# Step 5: Create Separate Soil Classification Maps
classified_0_5 = np.full((rows * cols), 0)  # Default class: 0 (unclassified)
classified_5_15 = np.full((rows * cols), 0)  # Default class: 0 (unclassified)

# Assign classes for 0-5 cm depth
classified_0_5[np.where(clay_pred_0_5 == 1)] = 1  # Clay (0-5 cm)
classified_0_5[np.where(sand_pred_0_5 == 1)] = 2  # Sand (0-5 cm)

# Assign classes for 5-15 cm depth
classified_5_15[np.where(clay_pred_5_15 == 1)] = 1  # Clay (5-15 cm)
classified_5_15[np.where(sand_pred_5_15 == 1)] = 2  # Sand (5-15 cm)

# Reshape classified images
classified_0_5 = classified_0_5.reshape(rows, cols)
classified_5_15 = classified_5_15.reshape(rows, cols)
# Step 5.1: Remove extra borders from the image
border_size = 12  # Adjust this value based on the border width
classified_0_5 = classified_0_5[border_size:-border_size, border_size:-border_size]
classified_5_15 = classified_5_15[border_size:-border_size, border_size:-border_size]

# Update metadata settings to correct the resized image dimensions
new_transform = rasterio.transform.Affine(
    transform.a, transform.b, transform.c + border_size * transform.a,
    transform.d, transform.e, transform.f + border_size * transform.e
)

# Update metadata to save the new images with corrected dimensions
meta.update({
    "height": classified_0_5.shape[0],  # Updated number of rows
    "width": classified_0_5.shape[1],   # Updated number of columns
    "transform": new_transform
})

# Step 6: Save the classified soil maps as GeoTIFF with full metadata
output_image_0_5_path = "classified_soil_map_0_5_Landsat9.tif"
output_image_5_15_path = "classified_soil_map_5_15_Landsat9.tif"

meta.update({"count": 1, "dtype": "float32", "driver": "GTiff", "transform": transform, "crs": crs})

with rasterio.open(output_image_0_5_path, "w", **meta) as dst:
    dst.write(classified_0_5, 1)  # Save classification map (0-5 cm)

with rasterio.open(output_image_5_15_path, "w", **meta) as dst:
    dst.write(classified_5_15, 1)  # Save classification map (5-15 cm)

print("Classified soil maps saved as:")
print(f"🔹 {output_image_0_5_path} (GeoTIFF, ready for QGIS)")
print(f"🔹 {output_image_5_15_path} (GeoTIFF, ready for QGIS)")

# Step 7: Visualize the classified soil maps (each in a separate figure)
cmap = ListedColormap(["black", "green", "yellow"])  # Black: No Data, Green: Clay, Yellow: Sand

# Plot 0-5 cm depth map in a separate figure
plt.figure(figsize=(7, 6))
plt.imshow(classified_0_5, cmap=cmap)
plt.colorbar(ticks=[1, 2], label="Soil Classes")
plt.title("Classified Soil Map (0-5 cm, Landsat-9)")
plt.show()

# Plot 5-15 cm depth map in a separate figure
plt.figure(figsize=(7, 6))
plt.imshow(classified_5_15, cmap=cmap)
plt.colorbar(ticks=[1, 2], label="Soil Classes")
plt.title("Classified Soil Map (5-15 cm, Landsat-9)")
plt.show()


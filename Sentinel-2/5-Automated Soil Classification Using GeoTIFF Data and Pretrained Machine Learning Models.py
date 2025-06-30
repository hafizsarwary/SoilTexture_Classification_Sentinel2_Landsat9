# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:41:18 2025

@author: alimo
"""

import rasterio
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 📌 Input Sentinel-2 Image File Path
input_image_path = "Sentinel2_Bands_and_Indices.tif"

# 📌 Paths to Pre-trained Models (Sentinel-2)
clay_model_0_5_path = "clay_model_clay_0_5.pkl"
clay_model_5_15_path = "clay_model_clay_5_15.pkl"
sand_model_0_5_path = "sand_model_sand_0_5.pkl"
sand_model_5_15_path = "sand_model_sand_5_15.pkl"

# ✅ Step 1: Read Sentinel-2 Image
with rasterio.open(input_image_path) as dataset:
    bands = [dataset.read(i) for i in range(1, dataset.count + 1)]  # Read all bands and indices
    meta = dataset.meta.copy()  # Extract metadata
    transform = dataset.transform  # Get georeferencing transform
    crs = dataset.crs  # Get coordinate reference system

# Convert to a 3D array
bands_array = np.stack(bands, axis=-1)
rows, cols, n_features = bands_array.shape  # Get image dimensions

# Convert 3D array to 2D for processing
pixels_array = bands_array.reshape(-1, n_features)

# 📌 Sentinel-2 Band Names in Correct Order
band_names = [
    "B3", "B4", "B5", "B6", "B7", "B8",  # Sentinel-2 raw bands
    "NDVI_S2", "SAVI_S2", "EVI_S2",  # Common indices
    "MCARI_S2", "IRECI_S2", "MTCI_S2", "S2REP_S2"  # Red-edge derived indices
]

# Convert array to DataFrame with correct band names
data_df = pd.DataFrame(pixels_array, columns=band_names)

# ✅ Step 2: Data Preprocessing
data_df = data_df.fillna(data_df.median())  # Replace NaN with the median value
lower_bound = data_df.quantile(0.01)
upper_bound = data_df.quantile(0.99)
data_df = data_df.clip(lower=lower_bound, upper=upper_bound, axis=1)

# 📌 Correct Feature Selection for the Model
expected_features = ["B3", "B4", "B5", "B6", "B7", "B8", "NDVI_S2", "SAVI_S2", "EVI_S2", "MCARI_S2", "IRECI_S2"]
data_df = data_df[expected_features]

# ✅ Step 3: Load Pre-trained Models
def load_model(model_path):
    with open(model_path, "rb") as model_file:
        model_data = pickle.load(model_file)
    return model_data["model"], model_data["features"]

clay_model_0_5, clay_features_0_5 = load_model(clay_model_0_5_path)
clay_model_5_15, clay_features_5_15 = load_model(clay_model_5_15_path)
sand_model_0_5, sand_features_0_5 = load_model(sand_model_0_5_path)
sand_model_5_15, sand_features_5_15 = load_model(sand_model_5_15_path)

# ✅ Step 4: Soil Prediction
def predict_soil(model, features, data):
    inputs = data[features]
    inputs = inputs.clip(lower=data_df[features].min(), upper=data_df[features].max(), axis=1)
    return model.predict(inputs)

clay_pred_0_5 = predict_soil(clay_model_0_5, clay_features_0_5, data_df)
clay_pred_5_15 = predict_soil(clay_model_5_15, clay_features_5_15, data_df)
sand_pred_0_5 = predict_soil(sand_model_0_5, sand_features_0_5, data_df)
sand_pred_5_15 = predict_soil(sand_model_5_15, sand_features_5_15, data_df)

# ✅ Step 5: Generate Soil Classification Maps
classified_0_5 = np.full((rows * cols), 0)  # Default class: 0 (unclassified)
classified_5_15 = np.full((rows * cols), 0)  # Default class: 0 (unclassified)

# Assign classes for 0-5 cm depth
classified_0_5[np.where(clay_pred_0_5 == 1)] = 1  # Clay (0-5 cm)
classified_0_5[np.where(sand_pred_0_5 == 1)] = 2  # Sand (0-5 cm)

# Assign classes for 5-15 cm depth
classified_5_15[np.where(clay_pred_5_15 == 1)] = 1  # Clay (5-15 cm)
classified_5_15[np.where(sand_pred_5_15 == 1)] = 2  # Sand (5-15 cm)

# Reshape to image dimensions
classified_0_5 = classified_0_5.reshape(rows, cols)
classified_5_15 = classified_5_15.reshape(rows, cols)

# ✅ Step 5.1: Remove extra borders from the image
# Adjust border_size based on Sentinel-2 image (you may need to change this value)
border_size = 34  # Adjust this value based on the border width of Sentinel-2 image

# Remove borders from the classified images
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

# ✅ Step 6: Save Classification Maps as GeoTIFF
output_image_0_5_path = "classified_soil_map_0_5_Sentinel2.tif"
output_image_5_15_path = "classified_soil_map_5_15_Sentinel2.tif"

meta.update({"count": 1, "dtype": "float32", "driver": "GTiff", "transform": transform, "crs": crs})

with rasterio.open(output_image_0_5_path, "w", **meta) as dst:
    dst.write(classified_0_5, 1)  # Save 0-5 cm classification map

with rasterio.open(output_image_5_15_path, "w", **meta) as dst:
    dst.write(classified_5_15, 1)  # Save 5-15 cm classification map

print("✅ Classified soil maps saved:")
print(f"🔹 {output_image_0_5_path} (GeoTIFF, ready for QGIS)")
print(f"🔹 {output_image_5_15_path} (GeoTIFF, ready for QGIS)")

# ✅ Step 7: Visualization
cmap = ListedColormap(["black", "green", "yellow"])  # Black: No Data, Green: Clay, Yellow: Sand

# Display 0-5 cm depth map
plt.figure(figsize=(7, 6))
plt.imshow(classified_0_5, cmap=cmap)
plt.colorbar(ticks=[1, 2], label="Soil Classes")
plt.title("Classified Soil Map (0-5 cm, Sentinel-2)")
plt.show()

# Display 5-15 cm depth map
plt.figure(figsize=(7, 6))
plt.imshow(classified_5_15, cmap=cmap)
plt.colorbar(ticks=[1, 2], label="Soil Classes")
plt.title("Classified Soil Map (5-15 cm, Sentinel-2)")
plt.show()

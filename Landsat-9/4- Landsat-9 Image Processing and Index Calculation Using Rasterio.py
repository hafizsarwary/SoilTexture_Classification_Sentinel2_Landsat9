# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:07:10 2025

@author: alimo
"""

import rasterio
import numpy as np

# Path to the Landsat-9 satellite image
input_image_path = "Landsat9_FullBands.tif"
output_image_path = "Landsat9_Bands_and_Indices.tif"

# Define band names (raw bands + indices)
band_names = [
    "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
    "NDVI_L9", "SAVI_L9", "EVI_L9"
]

# Read the satellite image
with rasterio.open(input_image_path) as dataset:
    bands = [dataset.read(i) for i in range(1, min(8, dataset.count + 1))]  # Read only required raw bands
    meta = dataset.meta  # Save metadata

# Convert bands to a 3D array
bands_array = np.stack(bands, axis=-1)

# Calculate indices with NaN prevention
NDVI_L9 = np.nan_to_num(((bands_array[:, :, 4] - bands_array[:, :, 3]) / (bands_array[:, :, 4] + bands_array[:, :, 3] + 1e-10)).astype(np.float32))
SAVI_L9 = np.nan_to_num(((bands_array[:, :, 4] - bands_array[:, :, 3]) * 1.5 / (bands_array[:, :, 4] + bands_array[:, :, 3] + 0.5 + 1e-10)).astype(np.float32))
EVI_L9 = np.nan_to_num(((bands_array[:, :, 4] - bands_array[:, :, 3]) * 2.5 / (bands_array[:, :, 4] + 6 * bands_array[:, :, 3] - 7.5 * bands_array[:, :, 1] + 1 + 1e-10)).astype(np.float32))

# Combine indices with raw bands
all_bands_and_indices = np.concatenate(
    [
        bands_array,
        NDVI_L9[..., None],
        SAVI_L9[..., None],
        EVI_L9[..., None],
    ],
    axis=-1,
)

# Ensure band_names has the correct number of elements
expected_band_count = all_bands_and_indices.shape[-1]
if len(band_names) != expected_band_count:
    print(f"Warning: band_names length ({len(band_names)}) does not match expected band count ({expected_band_count}).")
    band_names = band_names[:expected_band_count]  # Adjust to match expected count

# Update metadata for the new image
meta.update({
    "count": expected_band_count,  # Number of new bands (raw bands + indices)
    "dtype": "float32",
})

# Save the new image with indices and band names
with rasterio.open(output_image_path, "w", **meta) as dst:
    for i in range(expected_band_count):
        dst.write(all_bands_and_indices[:, :, i], i + 1)
        dst.set_band_description(i + 1, band_names[i])  # Assign band names

print("Image with bands and indices saved as:", output_image_path)

# Verify and display stored band names
print("\nNames of stored bands and indices:")
with rasterio.open(output_image_path) as dataset:
    stored_band_names = dataset.descriptions
    for i, name in enumerate(stored_band_names, start=1):
        print(f"Band {i} ({name}):")
        band_data = dataset.read(i)
        print(f"  Minimum value: {np.nanmin(band_data)}")
        print(f"  Maximum value: {np.nanmax(band_data)}")
        print(f"  Mean value: {np.nanmean(band_data)}")
        print("-" * 30)


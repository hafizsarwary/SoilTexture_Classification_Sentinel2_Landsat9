# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:43:15 2025

@author: alimo
"""

import rasterio
import numpy as np

# Path to the Sentinel-2 satellite image
input_image_path = "Sentinel2_Ordered_FullBands.tif"
output_image_path = "Sentinel2_Bands_and_Indices.tif"

# Define band names (raw bands + indices)
band_names = [
    "B3", "B4", "B5", "B6", "B7", "B8",  # Raw bands
    "NDVI_S2", "SAVI_S2", "EVI_S2",      # Common indices
    "MCARI_S2", "IRECI_S2", "MTCI_S2", "S2REP_S2"  # Red-edge indices
]

# Read the satellite image
with rasterio.open(input_image_path) as dataset:
    bands = [dataset.read(i) for i in range(1, min(7, dataset.count + 1))]  # Read only required raw bands
    meta = dataset.meta  # Save metadata

# Convert bands to a 3D array
bands_array = np.stack(bands, axis=-1)

# Extract required bands for calculations
B3 = bands_array[:, :, 0].astype(np.float32)
B4 = bands_array[:, :, 1].astype(np.float32)
B5 = bands_array[:, :, 2].astype(np.float32)
B6 = bands_array[:, :, 3].astype(np.float32)
B7 = bands_array[:, :, 4].astype(np.float32)
B8 = bands_array[:, :, 5].astype(np.float32)

# ✅ Calculate common indices
NDVI_S2 = np.nan_to_num(((B8 - B4) / (B8 + B4 + 1e-10)).astype(np.float32))
SAVI_S2 = np.nan_to_num(((B8 - B4) * 1.5 / (B8 + B4 + 0.5 + 1e-10)).astype(np.float32))
EVI_S2 = np.nan_to_num(((B8 - B4) * 2.5 / (B8 + 6 * B4 - 7.5 * B3 + 1 + 1e-10)).astype(np.float32))

# ✅ Calculate Red-Edge indices
MCARI_S2 = np.nan_to_num(((B6 - B5) - 0.2 * (B6 - B3)).astype(np.float32))
IRECI_S2 = np.nan_to_num(((B7 - B4) * B6 / (B5 + 1e-10)).astype(np.float32))
MTCI_S2 = np.nan_to_num(((B6 - B5) / (B5 - B4 + 1e-10)).astype(np.float32))
S2REP_S2 = np.nan_to_num((705 + 35 * ((B4 + B7) / 2 - B6) / (B7 - B6 + 1e-10)).astype(np.float32))

# ✅ Combine indices with raw bands
all_bands_and_indices = np.concatenate(
    [
        bands_array,
        NDVI_S2[..., None],
        SAVI_S2[..., None],
        EVI_S2[..., None],
        MCARI_S2[..., None],
        IRECI_S2[..., None],
        MTCI_S2[..., None],
        S2REP_S2[..., None]
    ],
    axis=-1,
)

# ✅ Ensure band_names has the correct number of elements
expected_band_count = all_bands_and_indices.shape[-1]
if len(band_names) != expected_band_count:
    print(f"Warning: band_names length ({len(band_names)}) does not match expected band count ({expected_band_count}).")
    band_names = band_names[:expected_band_count]  # Adjust to match expected count

# ✅ Update metadata for the new image
meta.update({
    "count": expected_band_count,  # Number of new bands (raw bands + indices)
    "dtype": "float32",
})

# ✅ Save the new image with indices and band names
with rasterio.open(output_image_path, "w", **meta) as dst:
    for i in range(expected_band_count):
        dst.write(all_bands_and_indices[:, :, i], i + 1)
        dst.set_band_description(i + 1, band_names[i])  # Assign band names

print("✅ Image with bands and indices saved as:", output_image_path)

# ✅ Verify and display stored band names
print("\n🔹 Names of stored bands and indices:")
with rasterio.open(output_image_path) as dataset:
    stored_band_names = dataset.descriptions
    for i, name in enumerate(stored_band_names, start=1):
        print(f"Band {i} ({name}):")
        band_data = dataset.read(i)
        print(f"  - Minimum value: {np.nanmin(band_data)}")
        print(f"  - Maximum value: {np.nanmax(band_data)}")
        print(f"  - Mean value: {np.nanmean(band_data)}")
        print("-" * 30)

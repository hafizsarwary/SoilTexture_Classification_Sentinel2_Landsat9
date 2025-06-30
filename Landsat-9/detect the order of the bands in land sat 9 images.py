# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:41:21 2025

@author: alimo
"""

import rasterio

# Path to the Landsat-9 image file
input_image_path = "Landsat9_Bands_Indices.tif"

# Open the raster file
with rasterio.open(input_image_path) as dataset:
    num_bands = dataset.count  # Number of bands
    print(f"Total Bands: {num_bands}\n")

    for i in range(1, num_bands + 1):
        print(f"Band {i}: {dataset.descriptions[i-1]}")  # Display band names

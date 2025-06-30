Soil Texture Classification Using Sentinel-2 and Landsat-9 Imagery with Machine Learning
This repository contains all codes, essential files, and model outputs used in our project for soil texture classification over Semnan Province, Iran.

📦 Project Structure
/Sentinel2
Contains codes, satellite imagery, extracted indices, ground truth data, trained models, and classification maps for the Sentinel-2 workflow.

/Landsat9
Contains similar materials for the Landsat-9 workflow.

/Feature_Importance_Plots
Contains visualizations of feature importance results from the models.

/Other
Additional files, including SoilGrids ground truth points if applicable.

🛠 Methods
Data Preprocessing: Radiometric & atmospheric corrections, spectral indices calculation
Feature Screening: Correlation and regression analysis
Machine Learning: Random Forest (RF) and Support Vector Machine (SVM)
Model Evaluation: Accuracy, Kappa, Confusion Matrix
Implemented in Python using libraries such as rasterio, numpy, pandas, scikit-learn, and others.

🗂 Files of Interest
.py files: Python scripts for processing, feature screening, model training, and evaluation
.tif files: Processed satellite imagery and final classification maps
.pkl files: Trained machine learning models
.csv files: Ground truth data extracted from SoilGrids

📋 Notes
All imagery and data are clipped to the study area.
Codes were executed on local Python environments and Google Earth Engine was used for data extraction.
This repository is intended for verification purposes to demonstrate the project was properly implemented as described

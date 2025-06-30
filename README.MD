Soil Texture Classification Using Sentinel-2 and Landsat-9 Imagery with Machine Learning

This repository contains the essential code, models, and relevant files for our soil texture classification project conducted over Semnan Province, Iran, as part of our Master's project at Politecnico di Milano.

📦 Project Structure

```
/Sentinel2              Sentinel-2 data, codes, models, outputs  
/Landsat9               Landsat-9 data, codes, models, outputs  
/Feature_Importance_Plots  Feature importance visualizations  
/Other                  Additional files, including SoilGrids points  
```

🗂 Contents

Sentinel2 Folder:

* Preprocessed satellite data (e.g., `Sentinel2_Ordered_FullBands.tif`)
* Calculated indices (e.g., `Sentinel2_Bands_and_Indices.tif`)
* Ground truth data (`GroundTruth_Sentinel2.csv`)
* Python scripts for:

  * Sensitivity and correlation analysis
  * Model training and evaluation (RF & SVM)
  * Image processing and classification
* Trained model files (`*.pkl`) and classified maps (`*.tif`)

Landsat9 Folder:

* Similar structure as Sentinel-2 for Landsat-9 workflow
* Includes band data, indices, ground truth, codes, models, and classification maps

Feature\_Importance\_Plots:

* Visual outputs showing feature importance for model interpretation

Other:

* `SoilGrids_Points.csv`: Contains the 3,000 ground truth soil texture points used for training and evaluation

🛠 Methods

* **Data Preprocessing**: Clipping, atmospheric correction, spectral index calculation
* **Feature Screening**: Correlation & regression analysis to select relevant features
* **Machine Learning**: Random Forest (RF) and Support Vector Machine (SVM)
* **Model Evaluation**: Overall Accuracy, Kappa Coefficient, Confusion Matrix
* **Tools Used**: Python (`scikit-learn`, `pandas`, `rasterio`, etc.) and Google Earth Engine for data extraction

📋 Notes

* The `.pkl` files are saved trained models for reproducibility
* `.tif` files include both raw and classified maps for visual validation
* The codes are organized for both Sentinel-2 and Landsat-9 workflows separately
* This repository is shared for project verification and academic purposes only

---

👨‍🏫 Project Details

* **Project Title**: Soil Texture Classification Using Multitemporal Landsat-9 and Sentinel-2 Data Combined with Machine Learning Techniques
* **Institution**: Politecnico di Milano - Master of Science in Geoinformatics Engineering
* **Supervision**: Prof. Mariagrazia Fugini & Prof. Giovanna Venuti
* **Authors**: Hafizullah Sarwary & Ali Moeinkhah

---

📝 Disclaimer

The dataset and codes are provided strictly for project evaluation and academic use. Redistribution or commercial use is not permitted.

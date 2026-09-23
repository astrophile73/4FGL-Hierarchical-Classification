# Fermi-LAT 4FGL-DR4 Hierarchical Source Classification

## Overview
This repository contains a complete machine learning pipeline designed to categorize unassociated and unknown gamma-ray sources from the Fermi Large Area Telescope (Fermi-LAT) 4FGL-DR4 catalog. Due to severe class imbalances and the physical distinctness of source types, the project utilizes a **Hierarchical Classification Architecture**. It first separates sources into broad astrophysical macro-classes, and then routes high-confidence candidates to specialized micro-classifiers for fine-grained identification.

## Pipeline Architecture
The workflow is divided into four distinct stages:

1. **Data & Feature Engineering:** Extracts spectral features (e.g., `PL_Index`, `LP_SigCurv`), variability indices, and Galactic coordinates from raw FITS files, unpacking 8-band flux arrays to compute inter-band Hardness Ratios.
2. **Preprocessing & Balancing:** Handles Astropy string masks and infinite values, scales the feature space, and applies SMOTE to balance the heavily skewed macro-classes (e.g., AGNs vs. Pulsars) for model training.
3. **Macro-Classification:** Trains a Random Forest classifier (5-Fold Cross-Validation) to categorize sources into 5 broad classes: AGN, Pulsar, Binary, Galactic, and Galaxy.
4. **Micro-Classification & Validation:** Routes high-confidence AGNs and Pulsars to specialized binary sub-classifiers (BLL vs. FSRQ, and MSP vs. PSR). Outputs final merged catalogs and scientific validation plots (Feature Importances and Aitoff Spatial Distribution).

## Key Results
* **Model Performance:** The macro-classifier achieved a 98.84% Balanced Accuracy and Macro F1-Score during 5-fold cross-validation.
* **Inference:** Successfully classified 2,563 unassociated targets, heavily identifying new Active Galactic Nuclei (AGNs) and resolving 1,221 uncertain blazars (BCUs) into concrete BL Lacs and FSRQs.
* **Physical Validity:** Feature importance extraction confirmed that spectral curvature (`LP_SigCurv`) and Galactic Latitude (`GLAT`) correctly drove the model's physical decision boundaries.

## Repository Structure
* `notebook.ipynb` - The complete codebase covering Stages 1 through 4.
* `4fgl_final_master_catalog.csv` - The final resolved classification catalog for unassociated sources.
* `4fgl_resolved_bcus.csv` - Specific classifications for Blazar Candidates of Uncertain type.
* `4fgl_dr4_class_summary.csv` - Baseline class distributions from the raw 4FGL-DR4 catalog.
* `4fgl_rf_model.joblib` / `scaler.joblib` / `label_encoder.joblib` - Trained model artifacts.
* `spatial_distribution_map.png` - Galactic coordinate plot of high-confidence predictions.
* `feature_importances.png` - Bar chart of the physical features driving the Macro-Classifier.
* `confusion_matrix.png` - Cross-validation performance visualization.

## Requirements and Usage
**Dependencies:**
`pip install pandas numpy scikit-learn imbalanced-learn astropy matplotlib seaborn joblib`

**Execution:**
To run the pipeline locally, download the original `gll_psc_v35.fit` catalog from the [Fermi Science Support Center](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/) and place it in the root directory before running the notebook cells sequentially.

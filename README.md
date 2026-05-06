**README**

---

### 1. Introduction  
Soil temperature is a critical variable influencing land-atmosphere interactions. 
However, long-term and high-resolution soil temperature data remain scarce across China. 
To address this issue, we trained a CatBoost model using 2,481 in situ observations and developed a Multi-Source Soil Temperature Fusion Dataset for China (CSTX).  
The dataset is publicly accessible at:  https://doi.org/10.11888/Terre.tpdc.302923

---

### 2. Required Python Packages  
The following Python packages are required to execute the code:

| Package       | Version | Documentation/Installation           |
|---------------|---------|--------------------------------------|
| CatBoost      | 1.2.7   | https://catboost.ai/docs/en/         |
| Scikit‑Learn  | 1.2.1   | https://scikit‑learn.org/stable/     |
| Optuna        | 3.3.0   | https://optuna.readthedocs.io/       |
| SHAP          | 0.46.0  | https://shap.readthedocs.io/         |

---

### 3. Usage Instructions  
- **Model training**: `train_ST_ML_model.ipynb`  
- **Hyperparameter tuning**: `optimize_ML_hyperparams.ipynb`  
- **Feature selection**: `feature_selection.py`  
- **Visualization**:  
  - SHAP analysis: `ShapAnalysis.ipynb`  
  - CSTX data sample visualization: `draw_ST_sample_show.ipynb`
---

### 4. Citation  
If you use this dataset or code in your research, please cite the corresponding data publication using the DOI:  
https://doi.org/10.11888/Terre.tpdc.302923

# Group Contribution Gaussian Process Hybrid Models for Thermophysical Property Prediction

This repository accompanies the research paper:

**“Enhanced Thermophysical Property Prediction with Uncertainty Quantification using Group Contribution-Gaussian Process Regression”**

---

## Repository Contents

This repository contains all data, scripts, and results related to the study. Below is a breakdown of the key folders and files:

### 1. Data Files
- `Data_for_Model_Building` folder containing all curated and preprocessed datasets (`*_fcl.csv`) for model building for all properties.
- `Hvap_data_test_fluorinated_molecules.csv`: Test set used to evaluate GCGP ΔHvap model performance on highly fluorinated molecules.

### 2. Final Results
- Located in the `Final_Results` folder.
- Includes:
  - Parity plots
  - Numerical model predictions
  - Model performance metrics
  - Model training outputs
  - Results from different random seeds for train/test splits
  - Results from different random seeds for train/test splits
  - `lml_values.csv`: A summary of collected LML values from kernel architecture experiments.

### 3. Kernel Sweep Experiments
- Found in `kernel_sweep_code_and_results` folder.
- Contains results of testing multiple kernel designs and model architectures as detailed in the paper.

### 4. Data Visualization, Data Quality, and Outlier Analysis
- `Data_Visualization_Pretraining` and `Data_Quality_and_Outlier_Checks` folders include:
  - Data analysis plots
  - Visualization figures
  - Outlier detection and data quality assessment

### 5. White Noise Kernel Tests
- `Tm_whitenoise_tests` folder contains results analyzing the impact of various white noise kernel settings on normal melting temperature (`Tm`) predictions.

### 6. Log-Marginal Likelihood (LML) Analysis
- `LML_plots` folder contains LML plots for various model and kernel combinations across all properties.
- - `lml_values.csv` in `Final_Results` folder contains data used in making the plots in `LML_plots` as well as additional data.
  

### 7. Python Scripts
- `Code.py` folder contains all Python code (`*.py`) used to train models, analyze data, and generate figures.
- Scripts are descriptively named for easy navigation and use.

---

## Citation
If you use this code or data, please cite the corresponding paper.

---

Feel free to open an issue or pull request if you have questions, suggestions, or contributions!

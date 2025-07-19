# Group Contribution Gaussian Process Models for Thermophysical Property Prediction

This repository accompanies the research paper:

**“Enhanced Thermophysical Property Prediction with Uncertainty Quantification using Group Contribution-Gaussian Process Regression”**

---

## Repository Contents

This repository contains all data, scripts, and results related to the study. Below is a breakdown of the key folders and files:

### 1. Data Files
- Curated and preprocessed datasets (`*_fcl.csv`) for all properties.
- `Hvap_data_test_fluorinated_molecules.csv`: Test set used to evaluate GCGP ΔHvap model performance on highly fluorinated molecules.

### 2. Final Results
- Located in the `Final_Results` folder.
- Includes:
  - Parity plots
  - Numerical model predictions
  - Model performance metrics
  - Model training outputs
  - Results from different random seeds for train/test splits

### 3. Kernel Sweep Experiments
- Found in `kernel_sweep_code_and_results`.
- Contains results of testing multiple kernel designs and model architectures as detailed in the paper.

### 4. Data Visualization and Outlier Analysis
- `Data_Vis_Figs` and `Data_Viz_and_Outlier_Figs` folders include:
  - Data analysis plots
  - Visualization figures
  - Outlier detection and data quality assessment

### 5. White Noise Kernel Tests
- `Tm_whitenoise_tests` contains results analyzing the impact of various white noise kernel settings on normal melting temperature (`Tm`) predictions.

### 6. Log-Marginal Likelihood (LML) Analysis
- `LML_plots`: LML plots for various model and kernel combinations across all properties.
- `lml_values.csv`: A summary of collected LML values from kernel architecture experiments.

### 7. Python Scripts
- All Python scripts (`*.py`) used to generate figures, train models, and analyze data are included.
- Scripts are descriptively named for easy navigation and use.

---

## Citation
If you use this code or data, please cite the corresponding paper.

---

Feel free to open an issue or pull request if you have questions, suggestions, or contributions!

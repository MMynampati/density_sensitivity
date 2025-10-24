# Density Sensitivity ML Pipeline

A machine learning pipeline for predicting density sensitivity in chemical reactions using molecular strucuture and Coulomb matrices.

## 🔬 Project Overview

This project implements a complete ML pipeline to predict whether chemical reactions are sensitive to changes in electron density.  
Density-sensitive reactions are those where energy errors are driven by inaccuracies in the electron density, while density-insensitive reactions are those where errors arise primarily from the approximate functional form.

The pipeline integrates physics-based molecular encoding with modern ML techniques:

- **Molecular Parsing** – Reads and parses molecular geometries from `.xyz` files.  
- **Coulomb Matrix Descriptors** – Represents each molecule as a rotation- and permutation-invariant matrix capturing interatomic electrostatic interactions.  
- **Reaction Matrices** – Constructs block-diagonal reaction matrices that account for stoichiometric coefficients of reactants and products.  
- **Spectral Feature Extraction** – Computes and sorts eigenvalues of each reaction matrix to obtain fixed-length, invariant feature vectors.  
- **Learning and Prediction** – Trains **Random Forest** and **XGBoost** models for **binary classification** (density sensitive vs. insensitive).


## 📁 Project Structure

```
density_sensitivity/
├── main.py                      # Main pipeline orchestration
├── generate_cm.py              # Coulomb matrix generation
├── diagonalize_matrices.py     # Matrix diagonalization
├── pad_and_metadata.py         # Feature standardization
├── preprocess.py               # Data preprocessing utilities (combining matrices) 
├── train_binary_classifier.py  # Binary classification training
├── train_random_forest.py      # Regression model training
├── requirements.txt            # Python dependencies
├── Descriptor1/                # data and Analysis for descriptor 1 
└── validation/                 # Validation scripts
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MMynampati/density_sensitivity
cd density_sensitivity

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Process molecular datasets
python main.py

# Train binary classifier
python train_binary_classifier.py

# Train regression model
python train_random_forest.py
```
  
## 📈 Model Performance

### Random Forest
- **Accuracy:** [To be updated]  
- **F1-Score:** [To be updated]  
- **Precision / Recall:** [To be updated]  
- **Feature Importance:** 

---

### XGBoost
- **Accuracy:** [To be updated]  
- **F1-Score:** [To be updated]  
- **Precision / Recall:** [To be updated]  
- **Feature Importance:**   

---

*The dataset contains a moderate class imbalance (~33% density-sensitive, ~67% density-insensitive reactions). Models were evaluated with metrics robust to imbalance, including F1-score and precision/recall.*



## 📝 Data 

- GMTKN55 database 

## 🙏 Acknowledgments

- Burke Group @ UCI

##  Resources
- https://hunterheidenreich.com/posts/molecular-descriptor-coulomb-matrix/#the-coulomb-matrix
- https://goerigk.chemistry.unimelb.edu.au/research/the-gmtkn55-database
- https://pubs.acs.org/doi/10.1021/acs.jctc.4c00689
 

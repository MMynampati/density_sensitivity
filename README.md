# Density Sensitivity ML Pipeline

A machine learning pipeline for predicting density sensitivity in chemical reactions using molecular strucuture and Coulomb matrices.

## 🎯 Project Overview

This project implements a complete ML pipeline to predict whether chemical reactions are sensitive to changes in electron density. The pipeline:

1. **Processes molecular structures** from XYZ files
2. **Generates Coulomb matrices** for molecular representation
3. **Combines reaction matrices** according to stoichiometry
4. **Extracts eigenvalues** as compressed features
5. **Trains ML models** for binary classification and regression using Random Forest and XGBoost

## 🔬 Scientific Background

- **Density Sensitivity**: Measures how much a chemical reaction's outcome changes when electron density is perturbed
- **Coulomb Matrices**: Mathematical representations encoding atomic interactions in molecules
- **Eigenvalues**: Compressed representations preserving essential chemical information

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
├── TODO.md                     # Project status and tasks
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

## 📊 Current Status

- ✅ expanded to all 55 molecular datasets
- ✅ Binary classification model trained and evaluated
  
## 📈 Model Performance

The binary classifier achieves:
- **F1-Score**: [To be updated]
- **Accuracy**: [To be updated]
- **Feature Importance**: Top features identified from eigenvalue analysis


## 📝 Data 

- GMTKN55 database 

## 🙏 Acknowledgments

- Burke Group @ UCI

##  Resources
- https://hunterheidenreich.com/posts/molecular-descriptor-coulomb-matrix/#the-coulomb-matrix
- https://goerigk.chemistry.unimelb.edu.au/research/the-gmtkn55-database
- https://pubs.acs.org/doi/10.1021/acs.jctc.4c00689
 

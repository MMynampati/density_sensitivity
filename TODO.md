# Density Sensitivity ML Pipeline - TODO Checklist

## Data Processing Pipeline

### 1. Dataset Structure & Access
- [ ] Write structure to go into each subset folder
- [ ] Fix hardcoded data paths in scripts
- [ ] Verify access to all 55 molecular datasets

### 2. Coulomb Matrix Generation  
- [x] Make Coulomb matrices for each molecule (ACONF subset complete)
- [ ] Extend Coulomb matrix generation to all remaining 54 subsets
- [ ] Validate matrix generation for all molecule types

### 3. Reaction Matrix Creation
- [x] Create combined reactant-product matrix (basic implementation)
- [ ] Optimize combined matrix creation for all datasets
- [ ] Handle edge cases in stoichiometry combinations

### 4. Matrix Compression
- [ ] Diagonalize matrix to compress to 1D
- [ ] Implement eigenvalue extraction
- [ ] Preserve chemical information during compression

### 5. Array Standardization
- [ ] Pad produced 1D array with 0's to size of largest molecule in all subsets
- [ ] Determine maximum molecule size across all 55 datasets
- [ ] Implement consistent padding strategy

### 6. Metadata Integration
- [ ] Add charge information to array (based on ref file)
- [ ] Add spin information to array (based on ref file) 
- [ ] Parse additional molecular properties from info files

### 7. Target Labeling
- [ ] Label each array based on density sensitivity value in SWARM file
- [ ] Map molecular systems to SWARM dataset entries
- [ ] Create target variable for supervised learning

### 8. Data Splitting
- [ ] Create train/val/test split for Random Forest model
- [ ] Ensure balanced distribution across splits
- [ ] Implement stratified splitting if needed

### 9. Random Forest Implementation
- [ ] Create Random Forest model architecture
- [ ] Set up hyperparameter tuning pipeline
- [ ] Implement cross-validation strategy

### 10. Model Training & Evaluation
- [ ] Train Random Forest on prepared dataset
- [ ] Test Random Forest performance
- [ ] Generate performance metrics and visualizations
- [ ] Analyze feature importance

## Current Status
- ✅ Basic Coulomb matrix generation (ACONF only)
- ✅ Reference file parsing
- ✅ Combined matrix creation (basic)
- ✅ Requirements.txt and project structure

## Next Priority
**Fix data paths and test end-to-end pipeline on ACONF subset before scaling to all datasets**

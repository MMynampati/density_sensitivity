# Density Sensitivity ML Pipeline - TODO Checklist

## Data Processing Pipeline

### 1. Dataset Structure & Access
- [x] Write structure to go into each subset folder
- [x] Fix hardcoded data paths in scripts
- [ ] Verify access to all 55 molecular datasets
- [ ] Verify ref files exist for all 55 folders. 

### 2. Coulomb Matrix Generation  
- [x] Make Coulomb matrices for each molecule (ACONF subset complete)
- [ ] Extend Coulomb matrix generation to all remaining 54 subsets
- [ ] Validate matrix generation for all molecule types

### 3. Reaction Matrix Creation
- [x] Create combined reactant-product matrix (basic implementation)
- [ ] Optimize combined matrix creation for all datasets
- [x] Handle edge cases in stoichiometry combinations

### 4. Matrix Compression
- [x] Diagonalize matrix to compress to 1D
- [x] Implement eigenvalue extraction
- [x] Preserve chemical information during compression

### 5. Array Standardization
- [x] Pad produced 1D array with 0's to size of largest molecule in all subsets
- [x] Determine maximum molecule size across all 55 datasets
- [x] Implement consistent padding strategy

### 6. Metadata Integration
- [x] Add charge information to array (based on ref file)
- [x] Add spin information to array (based on ref file) 
- [x] Parse additional molecular properties from info files

### 7. Target Labeling
- [x] Label each array based on density sensitivity value in SWARM file
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

### 11. Testing & Validation
- [x] Test pipeline on single dataset (ACONF)
- [ ] Cross-dataset compatibility testing
- [ ] Performance benchmarking across dataset sizes
- [ ] Memory usage optimization for large datasets

### 12. Data Analysis [NON-PRIO, NICE TO HAVE]
- [ ] Feature importance analysis across datasets
- [ ] Eigenvalue distribution analysis across all datasets  
- [ ] Correlation analysis between molecular properties and targets
- [ ] Dataset size and complexity statistics
- [ ] Investigate class imbalances for density sensitivity across all 55 subsets

## Current Status
- ✅ Complete end-to-end pipeline working on ACONF dataset
- ✅ Modular architecture with clean separation of concerns
- ✅ All core functions implemented and tested
- ✅ Multiple output formats (npy, csv, pkl)

## Next Priority
**Expand pipeline to process all 55 datasets, then implement Random Forest training**

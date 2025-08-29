#!/usr/bin/env python3
"""
Train Random Forest on the complete balanced dataset for density sensitivity prediction.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Tuple, Dict, Any

def load_balanced_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the balanced dataset created by main.py
    """
    print("Loading balanced dataset...")
    
    features = np.load("balanced_features.npy")
    targets = np.load("balanced_targets.npy")
    # features = np.load("complete_features.npy")
    # targets = np.load("complete_targets.npy")
    
    print(f"✅ Loaded balanced dataset:")
    print(f"   Features shape: {features.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Class distribution: {np.sum(targets)} sensitive, {len(targets) - np.sum(targets)} not sensitive")
    
    return features, targets

def check_class_balance(targets: np.ndarray) -> Dict[str, Any]:
    """
    Check class balance and provide recommendations.
    """
    num_positive = np.sum(targets)
    num_negative = len(targets) - num_positive
    total = len(targets)
    
    balance_info = {
        'total_samples': total,
        'positive_samples': num_positive,
        'negative_samples': num_negative,
        'positive_ratio': num_positive / total,
        'negative_ratio': num_negative / total,
        'is_balanced': abs(num_positive - num_negative) <= 0.2 * total,
        'is_severe_imbalance': min(num_positive, num_negative) / total < 0.1
    }
    
    print(f"\n📊 Class Balance Analysis:")
    print(f"   Total samples: {total}")
    print(f"   Positive (sensitive): {num_positive} ({balance_info['positive_ratio']:.1%})")
    print(f"   Negative (not sensitive): {num_negative} ({balance_info['negative_ratio']:.1%})")
    
    if balance_info['is_balanced']:
        print(f"   ✅ Classes are perfectly balanced!")
    else:
        print(f"   ⚠️  Class imbalance detected")
    
    return balance_info

def create_train_test_split(features: np.ndarray, targets: np.ndarray, 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Create train/test splits for binary classification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, 
        test_size=test_size, 
        random_state=random_state,
        stratify=targets,  # Maintain class proportions
        shuffle=True
    )
    
    print(f"✅ Data split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Training class distribution: {np.sum(y_train)} sensitive, {len(y_train) - np.sum(y_train)} not sensitive")
    print(f"   Test class distribution: {np.sum(y_test)} sensitive, {len(y_test) - np.sum(y_test)} not sensitive")
    
    return X_train, X_test, y_train, y_test

def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    print("\n🔧 Performing Hyperparameter Tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=3,  # 3-fold CV for faster tuning
        scoring='f1',  # Optimize for F1-score
        n_jobs=-1,
        verbose=1
    )
    
    print("🔍 Testing parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV F1-score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, use_tuning: bool = True) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with optional hyperparameter tuning.
    """
    if use_tuning:
        print("\n🌲 Training Random Forest with Hyperparameter Tuning...")
        rf_model = hyperparameter_tuning(X_train, y_train)
    else:
        print("\n🌲 Training Random Forest with Default Parameters...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
    
    print(f"✅ Random Forest trained with {rf_model.n_estimators} trees")
    
    return rf_model

def perform_cross_validation(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
    """
    Perform stratified k-fold cross-validation.
    """
    print(f"\n🔄 Performing {n_splits}-fold Stratified Cross-Validation...")
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lists to store metrics for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model on this fold
        rf_fold = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        rf_fold.fit(X_train_fold, y_train_fold)
        
        # Predict on validation set
        y_pred_fold = rf_fold.predict(X_val_fold)
        
        # Calculate metrics
        accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
        precisions.append(precision_score(y_val_fold, y_pred_fold, zero_division=0))
        recalls.append(recall_score(y_val_fold, y_pred_fold, zero_division=0))
        f1_scores.append(f1_score(y_val_fold, y_pred_fold, zero_division=0))
        
        print(f"   Fold {fold}: Accuracy={accuracies[-1]:.4f}, F1={f1_scores[-1]:.4f}")
    
    # Calculate mean and std of metrics
    cv_metrics = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores)
    }
    
    print(f"\n📊 Cross-Validation Results ({n_splits}-fold):")
    print(f"   Accuracy:  {cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}")
    print(f"   Precision: {cv_metrics['precision_mean']:.4f} ± {cv_metrics['precision_std']:.4f}")
    print(f"   Recall:    {cv_metrics['recall_mean']:.4f} ± {cv_metrics['recall_std']:.4f}")
    print(f"   F1-Score:  {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
    
    return cv_metrics

def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    """
    print("\n📊 Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"📈 Test Set Performance:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
    print(f"   False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
    
    return metrics

def analyze_feature_importance(model: RandomForestClassifier) -> pd.DataFrame:
    """
    Analyze feature importance.
    """
    print("\n🔍 Analyzing feature importance...")
    
    importances = model.feature_importances_
    
    # Create feature names
    n_eigenvals = len(importances) - 2  # Subtract charge and spin
    feature_names = [f'eigenval_{i+1}' for i in range(n_eigenvals)] + ['charge', 'spin_mult']
    
    # Create DataFrame
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("🔝 Top 10 most important features:")
    print(feature_df.head(10).to_string(index=False))
    
    return feature_df

def create_evaluation_plots(y_test: np.ndarray, y_pred: np.ndarray, 
                           feature_importance_df: pd.DataFrame):
    """
    Create evaluation plots.
    """
    print("\n📊 Creating evaluation plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Plot 2: Class distribution comparison
    labels = ['Not Sensitive', 'Sensitive']
    true_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]
    pred_counts = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax2.bar(x - width/2, true_counts, width, label='True', alpha=0.7)
    ax2.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_title('True vs Predicted Class Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature importance (top 10)
    top_features = feature_importance_df.head(10)
    ax3.barh(range(len(top_features)), top_features['importance'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 10 Feature Importances')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature importance by type
    eigenval_importance = feature_importance_df[feature_importance_df['feature'].str.startswith('eigenval')]['importance'].sum()
    metadata_importance = feature_importance_df[~feature_importance_df['feature'].str.startswith('eigenval')]['importance'].sum()
    
    ax4.pie([eigenval_importance, metadata_importance], 
            labels=['Eigenvalues', 'Metadata (Charge + Spin)'], 
            autopct='%1.1f%%', startangle=90)
    ax4.set_title('Feature Importance by Type')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = "random_forest_evaluation.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved evaluation plots: {plot_filename}")

def save_model_and_results(model: RandomForestClassifier, 
                          metrics: Dict[str, float],
                          feature_importance_df: pd.DataFrame,
                          best_params: Dict = None):
    """
    Save the trained model and results.
    """
    print("\n💾 Saving model and results...")
    
    # Save model
    model_filename = "random_forest_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved model: {model_filename}")
    
    # Save metrics
    metrics_filename = "model_metrics.json"
    import json
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics: {metrics_filename}")
    
    # Save feature importance
    importance_filename = "feature_importance.csv"
    feature_importance_df.to_csv(importance_filename, index=False)
    print(f"✅ Saved feature importance: {importance_filename}")
    
    # Save best parameters if available
    if best_params:
        params_filename = "best_hyperparameters.json"
        import json
        with open(params_filename, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"✅ Saved best hyperparameters: {params_filename}")

def main():
    """Main training pipeline."""
    
    print("🎯 Random Forest Training on Complete Balanced Dataset")
    print("=" * 60)
    
    try:
        # Step 1: Load balanced data
        features, targets = load_balanced_data()
        
        # Step 2: Check class balance
        balance_info = check_class_balance(targets)
        
        # Step 3: Perform cross-validation
        cv_metrics = perform_cross_validation(features, targets, n_splits=5)
        
        # Step 4: Create train/test split for final evaluation
        X_train, X_test, y_train, y_test = create_train_test_split(features, targets)
        
        # Step 5: Train final Random Forest with hyperparameter tuning
        rf_model = train_random_forest(X_train, y_train, use_tuning=True)
        
        # Step 6: Evaluate final model
        test_metrics = evaluate_model(rf_model, X_test, y_test)
        
        # Step 6: Feature importance analysis
        feature_importance_df = analyze_feature_importance(rf_model)
        
        # Step 7: Create evaluation plots
        y_pred = rf_model.predict(X_test)
        create_evaluation_plots(y_test, y_pred, feature_importance_df)
        
        # Step 8: Save everything
        best_params = rf_model.get_params() if hasattr(rf_model, 'get_params') else None
        save_model_and_results(rf_model, test_metrics, feature_importance_df, best_params)
        
        # Step 9: Save cross-validation results
        cv_filename = "cross_validation_metrics.json"
        import json
        with open(cv_filename, 'w') as f:
            json.dump(cv_metrics, f, indent=2)
        print(f"✅ Saved cross-validation results: {cv_filename}")
        
        print("\n" + "=" * 60)
        print("🎉 Random Forest training complete!")
        print(f"📊 Cross-Validation F1-Score: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
        print(f"📊 Cross-Validation Accuracy: {cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}")
        print(f"📊 Test Set F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"📊 Test Set Accuracy: {test_metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

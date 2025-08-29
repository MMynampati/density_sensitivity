#!/usr/bin/env python3
"""
Binary classification training pipeline for density sensitivity prediction using SWARM labels.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_swarm_binary_data(subset_name: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load SWARM-based binary classification data.
    
    Args:
        subset_name: Name of subset (e.g., "ACONF")
        
    Returns:
        Tuple of (features, binary_targets, metadata_df)
    """
    
    # Load the arrays
    features_path = f"{subset_name}_features.npy"
    binary_targets_path = f"{subset_name}_binary_targets.npy"
    metadata_path = f"{subset_name}_swarm_metadata.csv"
    
    if not all(os.path.exists(p) for p in [features_path, binary_targets_path, metadata_path]):
        raise FileNotFoundError(f"Missing SWARM data files for {subset_name}")
    
    features = np.load(features_path)
    binary_targets = np.load(binary_targets_path)
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"✅ Loaded {subset_name} SWARM binary data:")
    print(f"   Features shape: {features.shape}")
    print(f"   Binary targets shape: {binary_targets.shape}")
    print(f"   Class distribution: {np.sum(binary_targets)} sensitive, {len(binary_targets) - np.sum(binary_targets)} not sensitive")
    print(f"   S value range: {metadata_df['swarm_s_value'].min():.4f} to {metadata_df['swarm_s_value'].max():.4f}")
    
    return features, binary_targets, metadata_df


def check_class_balance(targets: np.ndarray) -> Dict[str, Any]:
    """
    Check class balance and provide recommendations.
    
    Args:
        targets: Binary target array
        
    Returns:
        Dictionary with class balance information
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
    
    if balance_info['is_severe_imbalance']:
        print(f"   ⚠️  SEVERE CLASS IMBALANCE detected!")
        print(f"   📝 Recommendation: Consider collecting more data or using techniques like SMOTE")
    elif not balance_info['is_balanced']:
        print(f"   ⚠️  Moderate class imbalance detected")
        print(f"   📝 Recommendation: Consider class weights or stratified sampling")
    else:
        print(f"   ✅ Classes are reasonably balanced")
    
    return balance_info


def create_train_test_split_classification(features: np.ndarray, targets: np.ndarray, 
                                         test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Create train/test splits for binary classification.
    
    Args:
        features: Feature matrix
        targets: Binary target values
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    
    # Check if we have enough samples for splitting
    if len(targets) <= 5:
        print("⚠️  Very small dataset - using all data for training, no test set")
        return features, np.array([]), targets, np.array([])
    
    # For very imbalanced datasets, try stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, 
            test_size=test_size, 
            random_state=random_state,
            stratify=targets,  # Maintain class proportions
            shuffle=True
        )
        print(f"✅ Stratified split successful")
    except ValueError:
        # Fall back to regular split if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        print(f"⚠️  Stratified split failed, using regular split")
    
    print(f"✅ Data split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Training class distribution: {np.sum(y_train)} sensitive, {len(y_train) - np.sum(y_train)} not sensitive")
    if len(y_test) > 0:
        print(f"   Test class distribution: {np.sum(y_test)} sensitive, {len(y_test) - np.sum(y_test)} not sensitive")
    
    return X_train, X_test, y_train, y_test


def train_baseline_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                            class_weight: str = 'balanced') -> RandomForestClassifier:
    """
    Train a baseline Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training targets
        class_weight: How to handle class imbalance ('balanced', 'balanced_subsample', or None)
        
    Returns:
        Trained RandomForestClassifier
    """
    
    print(f"\n🌲 Training baseline Random Forest Classifier...")
    print(f"   Using class_weight='{class_weight}' to handle imbalance")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight  # Handle class imbalance
    )
    
    rf.fit(X_train, y_train)
    
    print(f"✅ Baseline classifier trained with {rf.n_estimators} trees")
    
    return rf


def evaluate_binary_classifier(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate binary classifier performance.
    
    Args:
        model: Trained Random Forest classifier
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    print("\n📊 Evaluating binary classifier...")
    
    if len(y_test) == 0:
        print("⚠️  No test data available - skipping evaluation")
        return {}
    
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


def create_classification_plots(y_test: np.ndarray, y_pred: np.ndarray, 
                              feature_importance_df: pd.DataFrame,
                              subset_name: str):
    """
    Create classification evaluation plots.
    
    Args:
        y_test: True test values
        y_pred: Predicted test values
        feature_importance_df: Feature importance DataFrame
        subset_name: Name of dataset subset
    """
    
    print("\n📊 Creating classification plots...")
    
    if len(y_test) == 0:
        print("⚠️  No test data - creating feature importance plot only")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Feature importance only
        top_features = feature_importance_df.head(10)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 10 Feature Importances ({subset_name})')
        ax.grid(True, alpha=0.3)
        
    else:
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
        
        # Plot 4: Prediction distribution (if we have probabilities)
        ax4.hist([0, 1], bins=2, alpha=0.7, label='Predictions')
        ax4.set_xlabel('Predicted Class')
        ax4.set_ylabel('Count')
        ax4.set_title('Prediction Distribution')
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Not Sensitive', 'Sensitive'])
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{subset_name}_binary_classifier_evaluation.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved classification plots: {plot_filename}")


def analyze_feature_importance_classification(model: RandomForestClassifier, feature_names: list = None) -> pd.DataFrame:
    """
    Analyze feature importance for classification.
    
    Args:
        model: Trained Random Forest classifier
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    
    print("\n🔍 Analyzing feature importance for classification...")
    
    importances = model.feature_importances_
    
    if feature_names is None:
        # Create default feature names
        n_eigenvals = len(importances) - 2  # Subtract charge and spin
        feature_names = [f'eigenval_{i+1}' for i in range(n_eigenvals)] + ['charge', 'spin_mult']
    
    # Create DataFrame
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("🔝 Top 10 most important features for classification:")
    print(feature_df.head(10).to_string(index=False))
    
    return feature_df


def main():
    """Main binary classification training pipeline."""
    
    print("🎯 Binary Classification Training Pipeline")
    print("=" * 60)
    
    # Configuration
    subset_name = "ACONF"  # Start with ACONF data
    
    try:
        # Step 1: Load SWARM-based binary data
        features, binary_targets, metadata_df = load_swarm_binary_data(subset_name)
        
        # Step 2: Check class balance
        balance_info = check_class_balance(binary_targets)
        
        # Step 3: Create train/test split
        X_train, X_test, y_train, y_test = create_train_test_split_classification(features, binary_targets)
        
        # Step 4: Train baseline classifier
        baseline_classifier = train_baseline_classifier(X_train, y_train, class_weight='balanced')
        
        # Step 5: Evaluate classifier
        if len(y_test) > 0:
            classification_metrics = evaluate_binary_classifier(baseline_classifier, X_test, y_test)
        else:
            classification_metrics = {}
            print("⚠️  No test set - trained on all available data")
        
        # Step 6: Feature importance analysis
        feature_importance_df = analyze_feature_importance_classification(baseline_classifier)
        
        # Step 7: Create evaluation plots
        if len(y_test) > 0:
            y_pred = baseline_classifier.predict(X_test)
        else:
            y_pred = np.array([])
        create_classification_plots(y_test, y_pred, feature_importance_df, subset_name)
        
        # Step 8: Save everything
        model_filename = f"{subset_name}_binary_classifier.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(baseline_classifier, f)
        print(f"✅ Saved binary classifier: {model_filename}")
        
        print("\n" + "=" * 60)
        print("🎉 Binary classification training complete!")
        
        if balance_info['is_severe_imbalance']:
            print("⚠️  Note: Severe class imbalance detected - results may not be reliable")
            print("📝 Consider expanding to more datasets with positive examples")
        
        if classification_metrics:
            print(f"📊 Final F1-Score: {classification_metrics['f1_score']:.4f}")
            print(f"📊 Final Accuracy: {classification_metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Error in binary classification pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

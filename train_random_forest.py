#!/usr/bin/env python3
"""
Random Forest training pipeline for density sensitivity prediction.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


def load_ml_data(subset_name: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load ML-ready data for a subset.
    
    Args:
        subset_name: Name of subset (e.g., "ACONF")
        
    Returns:
        Tuple of (features, targets, metadata_df)
    """
    
    # Load the arrays
    features_path = f"{subset_name}_features.npy"
    targets_path = f"{subset_name}_targets.npy"
    metadata_path = f"{subset_name}_metadata.csv"
    
    if not all(os.path.exists(p) for p in [features_path, targets_path, metadata_path]):
        raise FileNotFoundError(f"Missing data files for {subset_name}")
    
    features = np.load(features_path)
    targets = np.load(targets_path)
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"✅ Loaded {subset_name} data:")
    print(f"   Features shape: {features.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Target range: {targets.min():.3f} to {targets.max():.3f}")
    
    return features, targets, metadata_df


def create_train_test_split(features: np.ndarray, targets: np.ndarray, 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Create train/test splits with optional validation split.
    
    Args:
        features: Feature matrix
        targets: Target values
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"✅ Data split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Training target range: {y_train.min():.3f} to {y_train.max():.3f}")
    print(f"   Test target range: {y_test.min():.3f} to {y_test.max():.3f}")
    
    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Train a baseline Random Forest model with default parameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Trained RandomForestRegressor
    """
    
    print("\n🌲 Training baseline Random Forest...")
    
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    rf.fit(X_train, y_train)
    
    print(f"✅ Baseline model trained with {rf.n_estimators} trees")
    
    return rf


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Best RandomForestRegressor model
    """
    
    print("\n🔧 Hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {-grid_search.best_score_:.4f} (MSE)")
    
    return grid_search.best_estimator_


def evaluate_model(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    print("\n📊 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"📈 Test Set Performance:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    return metrics


def analyze_feature_importance(model: RandomForestRegressor, feature_names: list = None) -> pd.DataFrame:
    """
    Analyze and visualize feature importance.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    
    print("\n🔍 Analyzing feature importance...")
    
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
    
    print("🔝 Top 10 most important features:")
    print(feature_df.head(10).to_string(index=False))
    
    return feature_df


def create_evaluation_plots(y_test: np.ndarray, y_pred: np.ndarray, 
                           feature_importance_df: pd.DataFrame, 
                           subset_name: str):
    """
    Create evaluation plots.
    
    Args:
        y_test: True test values
        y_pred: Predicted test values
        feature_importance_df: Feature importance DataFrame
        subset_name: Name of dataset subset
    """
    
    print("\n📊 Creating evaluation plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Prediction vs True values
    ax1.scatter(y_test, y_pred, alpha=0.7)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'Predictions vs True Values ({subset_name})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature importance (top 10)
    top_features = feature_importance_df.head(10)
    ax3.barh(range(len(top_features)), top_features['importance'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 10 Feature Importances')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Target distribution
    ax4.hist(y_test, bins=10, alpha=0.7, label='True', density=True)
    ax4.hist(y_pred, bins=10, alpha=0.7, label='Predicted', density=True)
    ax4.set_xlabel('Density Sensitivity Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Target Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{subset_name}_random_forest_evaluation.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved evaluation plots: {plot_filename}")


def save_model_and_results(model: RandomForestRegressor, 
                          metrics: Dict[str, float],
                          feature_importance_df: pd.DataFrame,
                          subset_name: str):
    """
    Save the trained model and results.
    
    Args:
        model: Trained Random Forest model
        metrics: Evaluation metrics
        feature_importance_df: Feature importance DataFrame
        subset_name: Name of dataset subset
    """
    
    print("\n💾 Saving model and results...")
    
    # Save model
    model_filename = f"{subset_name}_random_forest_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved model: {model_filename}")
    
    # Save metrics
    metrics_filename = f"{subset_name}_model_metrics.pkl"
    with open(metrics_filename, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✅ Saved metrics: {metrics_filename}")
    
    # Save feature importance
    importance_filename = f"{subset_name}_feature_importance.csv"
    feature_importance_df.to_csv(importance_filename, index=False)
    print(f"✅ Saved feature importance: {importance_filename}")


def main():
    """Main Random Forest training pipeline."""
    
    print("🤖 Random Forest Training Pipeline")
    print("=" * 50)
    
    # Configuration
    subset_name = "ACONF"  # Start with ACONF data
    
    try:
        # Step 1: Load data
        features, targets, metadata_df = load_ml_data(subset_name)
        
        # Step 2: Create train/test split
        X_train, X_test, y_train, y_test = create_train_test_split(features, targets)
        
        # Step 3: Train baseline model
        baseline_model = train_baseline_model(X_train, y_train)
        
        # Step 4: Evaluate baseline
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
        
        # Step 5: Hyperparameter tuning (optional - comment out for speed)
        print("\n🤔 Do hyperparameter tuning? (This may take a while...)")
        print("   Skipping for now - using baseline model")
        # tuned_model = hyperparameter_tuning(X_train, y_train)
        # tuned_metrics = evaluate_model(tuned_model, X_test, y_test)
        
        # Use baseline model for now
        final_model = baseline_model
        final_metrics = baseline_metrics
        
        # Step 6: Feature importance analysis
        feature_importance_df = analyze_feature_importance(final_model)
        
        # Step 7: Create evaluation plots
        y_pred = final_model.predict(X_test)
        create_evaluation_plots(y_test, y_pred, feature_importance_df, subset_name)
        
        # Step 8: Save everything
        save_model_and_results(final_model, final_metrics, feature_importance_df, subset_name)
        
        print("\n" + "=" * 50)
        print("🎉 Random Forest training complete!")
        print(f"📊 Final R² score: {final_metrics['r2']:.4f}")
        print(f"📊 Final RMSE: {final_metrics['rmse']:.4f}")
        
    except Exception as e:
        print(f"❌ Error in Random Forest pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

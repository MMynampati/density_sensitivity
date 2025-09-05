#!/usr/bin/env python3
"""
Evaluate model performance by individual set.
For each of the 55 sets, calculate accuracy and confusion matrix on reactions from that set only.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple
import os
from datetime import datetime

def load_model_and_data():
    """
    Load the trained model and the complete dataset with set information.
    """
    print("Loading model and data...")
    
    # Load trained model
    model_path = "random_forest_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load complete dataset
    features = np.load("complete_features.npy")
    targets = np.load("complete_targets.npy")
    
    # Load reaction vectors CSV to get set information
    reaction_df = pd.read_csv("reaction_vectors.csv")
    
    print(f"✅ Loaded model and data:")
    print(f"   Model: {type(model).__name__}")
    print(f"   Features shape: {features.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Reaction data shape: {reaction_df.shape}")
    
    return model, features, targets, reaction_df

def evaluate_by_set(model, features: np.ndarray, targets: np.ndarray, reaction_df: pd.DataFrame) -> Dict:
    """
    Evaluate model performance for each individual set.
    
    Returns:
        Dictionary with results for each set
    """
    print("\n🔍 Evaluating model performance by set...")
    
    # Get unique sets
    unique_sets = sorted(reaction_df['setname'].unique())
    print(f"Found {len(unique_sets)} unique sets: {unique_sets[:5]}...")
    
    set_results = {}
    
    for setname in unique_sets:
        print(f"\n📊 Evaluating set: {setname}")
        
        # Get indices for this set
        set_mask = reaction_df['setname'] == setname
        set_indices = reaction_df[set_mask].index.tolist()
        
        if len(set_indices) == 0:
            print(f"   ⚠️  No data found for {setname}")
            continue
        
        # Get features and targets for this set
        set_features = features[set_indices]
        set_targets = targets[set_indices]
        
        print(f"   Samples in {setname}: {len(set_targets)}")
        print(f"   Class distribution: {np.sum(set_targets)} sensitive, {len(set_targets) - np.sum(set_targets)} not sensitive")
        
        # Make predictions
        set_predictions = model.predict(set_features)
        
        # Calculate metrics
        accuracy = accuracy_score(set_targets, set_predictions)
        precision = precision_score(set_targets, set_predictions, zero_division=0)
        recall = recall_score(set_targets, set_predictions, zero_division=0)
        f1 = f1_score(set_targets, set_predictions, zero_division=0)
        
        # Confusion matrix - ensure it's 2x2 even for single-class sets
        cm = confusion_matrix(set_targets, set_predictions, labels=[0, 1])
        
        # Store results
        set_results[setname] = {
            'num_samples': len(set_targets),
            'num_sensitive': int(np.sum(set_targets)),
            'num_not_sensitive': int(len(set_targets) - np.sum(set_targets)),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': set_predictions,
            'true_labels': set_targets
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    return set_results

def create_summary_table(set_results: Dict) -> pd.DataFrame:
    """
    Create a summary table of results for all sets.
    """
    print("\n📋 Creating summary table...")
    
    summary_data = []
    for setname, results in set_results.items():
        summary_data.append({
            'Set': setname,
            'Samples': results['num_samples'],
            'Sensitive': results['num_sensitive'],
            'Not_Sensitive': results['num_not_sensitive'],
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1_score'],
            'TN': results['confusion_matrix'][0,0],
            'FP': results['confusion_matrix'][0,1],
            'FN': results['confusion_matrix'][1,0],
            'TP': results['confusion_matrix'][1,1]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    
    return summary_df

def create_visualizations(set_results: Dict, summary_df: pd.DataFrame, results_dir: str):
    """
    Create visualizations for per-set performance.
    """
    print("\n📊 Creating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Accuracy by set (top 20)
    ax1 = plt.subplot(2, 3, 1)
    top_20 = summary_df.head(20)
    bars = ax1.bar(range(len(top_20)), top_20['Accuracy'])
    ax1.set_xlabel('Set')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Top 20 Sets by Accuracy')
    ax1.set_xticks(range(len(top_20)))
    ax1.set_xticklabels(top_20['Set'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Color bars by accuracy
    for i, bar in enumerate(bars):
        if top_20.iloc[i]['Accuracy'] >= 0.8:
            bar.set_color('green')
        elif top_20.iloc[i]['Accuracy'] >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 2: F1-Score by set (top 20)
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(range(len(top_20)), top_20['F1_Score'])
    ax2.set_xlabel('Set')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Top 20 Sets by F1-Score')
    ax2.set_xticks(range(len(top_20)))
    ax2.set_xticklabels(top_20['Set'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample size vs Accuracy
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(summary_df['Samples'], summary_df['Accuracy'], 
                         c=summary_df['F1_Score'], cmap='viridis', alpha=0.7)
    ax3.set_xlabel('Number of Samples')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Sample Size vs Accuracy (colored by F1-Score)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='F1-Score')
    
    # Plot 4: Class balance vs Performance
    ax4 = plt.subplot(2, 3, 4)
    # Calculate class balance ratio (sensitive / total)
    balance_ratio = summary_df['Sensitive'] / summary_df['Samples']
    ax4.scatter(balance_ratio, summary_df['Accuracy'], 
               c=summary_df['F1_Score'], cmap='viridis', alpha=0.7)
    ax4.set_xlabel('Class Balance Ratio (Sensitive/Total)')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Class Balance vs Accuracy')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='F1-Score')
    
    # Plot 5: Confusion matrix for best performing set
    ax5 = plt.subplot(2, 3, 5)
    best_set = summary_df.iloc[0]['Set']
    best_cm = set_results[best_set]['confusion_matrix']
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
    ax5.set_title(f'Confusion Matrix - Best Set: {best_set}')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    
    # Plot 6: Confusion matrix for worst performing set
    ax6 = plt.subplot(2, 3, 6)
    worst_set = summary_df.iloc[-1]['Set']
    worst_cm = set_results[worst_set]['confusion_matrix']
    sns.heatmap(worst_cm, annot=True, fmt='d', cmap='Reds', ax=ax6)
    ax6.set_title(f'Confusion Matrix - Worst Set: {worst_set}')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Actual')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(results_dir, "per_set_evaluation.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {plot_filename}")

def create_detailed_confusion_matrices(set_results: Dict, summary_df: pd.DataFrame, results_dir: str):
    """
    Create detailed confusion matrices for all sets.
    """
    print("\n📊 Creating detailed confusion matrices...")
    
    # Get top 10 and bottom 10 sets
    top_10 = summary_df.head(10)
    bottom_10 = summary_df.tail(10)
    
    # Create figure for top 10
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(top_10.iterrows()):
        setname = row['Set']
        cm = set_results[setname]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{setname}\nAcc: {row["Accuracy"]:.3f}')
        axes[i].set_xlabel('Pred')
        axes[i].set_ylabel('True')
    
    plt.suptitle('Top 10 Sets - Confusion Matrices', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "top_10_sets_confusion_matrices.png"), dpi=150, bbox_inches='tight')
    print("✅ Saved top 10 confusion matrices")
    
    # Create figure for bottom 10
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(bottom_10.iterrows()):
        setname = row['Set']
        cm = set_results[setname]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[i])
        axes[i].set_title(f'{setname}\nAcc: {row["Accuracy"]:.3f}')
        axes[i].set_xlabel('Pred')
        axes[i].set_ylabel('True')
    
    plt.suptitle('Bottom 10 Sets - Confusion Matrices', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bottom_10_sets_confusion_matrices.png"), dpi=150, bbox_inches='tight')
    print("✅ Saved bottom 10 confusion matrices")

def save_results(set_results: Dict, summary_df: pd.DataFrame, results_dir: str):
    """
    Save all results to files.
    """
    print("\n💾 Saving results...")
    
    # Save summary table
    summary_path = os.path.join(results_dir, "per_set_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary table: {summary_path}")
    
    # Save detailed results
    detailed_results = {}
    for setname, results in set_results.items():
        detailed_results[setname] = {
            'num_samples': results['num_samples'],
            'num_sensitive': results['num_sensitive'],
            'num_not_sensitive': results['num_not_sensitive'],
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'predictions': results['predictions'].tolist(),
            'true_labels': results['true_labels'].tolist()
        }
    
    import json
    detailed_path = os.path.join(results_dir, "per_set_detailed_results.json")
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✅ Saved detailed results: {detailed_path}")
    
    # Save confusion matrices as separate CSV files
    for setname, results in set_results.items():
        cm_df = pd.DataFrame(results['confusion_matrix'], 
                           index=['Not_Sensitive', 'Sensitive'],
                           columns=['Not_Sensitive', 'Sensitive'])
        cm_path = os.path.join(results_dir, f"confusion_matrix_{setname}.csv")
        cm_df.to_csv(cm_path)
    
    print(f"✅ Saved individual confusion matrices for {len(set_results)} sets")

def main():
    """
    Main evaluation pipeline.
    """
    print("🎯 Per-Set Model Evaluation")
    print("=" * 60)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"per_set_evaluation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Created results directory: {results_dir}")
    
    try:
        # Step 1: Load model and data
        model, features, targets, reaction_df = load_model_and_data()
        
        # Step 2: Evaluate by set
        set_results = evaluate_by_set(model, features, targets, reaction_df)
        
        # Step 3: Create summary table
        summary_df = create_summary_table(set_results)
        
        # Step 4: Print summary statistics
        print(f"\n📊 Overall Summary:")
        print(f"   Total sets evaluated: {len(set_results)}")
        print(f"   Average accuracy: {summary_df['Accuracy'].mean():.4f} ± {summary_df['Accuracy'].std():.4f}")
        print(f"   Average F1-score: {summary_df['F1_Score'].mean():.4f} ± {summary_df['F1_Score'].std():.4f}")
        print(f"   Best performing set: {summary_df.iloc[0]['Set']} (Acc: {summary_df.iloc[0]['Accuracy']:.4f})")
        print(f"   Worst performing set: {summary_df.iloc[-1]['Set']} (Acc: {summary_df.iloc[-1]['Accuracy']:.4f})")
        
        # Step 5: Create visualizations
        create_visualizations(set_results, summary_df, results_dir)
        create_detailed_confusion_matrices(set_results, summary_df, results_dir)
        
        # Step 6: Save results
        save_results(set_results, summary_df, results_dir)
        
        # Step 7: Display top and bottom performers
        print(f"\n🏆 Top 10 Performing Sets:")
        print(summary_df.head(10)[['Set', 'Samples', 'Accuracy', 'F1_Score']].to_string(index=False))
        
        print(f"\n📉 Bottom 10 Performing Sets:")
        print(summary_df.tail(10)[['Set', 'Samples', 'Accuracy', 'F1_Score']].to_string(index=False))
        
        print("\n" + "=" * 60)
        print("🎉 Per-set evaluation complete!")
        
    except Exception as e:
        print(f"❌ Error in evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

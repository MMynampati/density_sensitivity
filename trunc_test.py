#!/usr/bin/env python3
"""
Test eigenvalue truncation impact on model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns

def analyze_molecule_sizes(features):
    """Analyze the distribution of molecule sizes."""
    print("🔬 Analyzing molecule size distribution...")
    
    # Extract eigenvalues (excluding charge/spin)
    eigenvals = features[:, :-2]
    
    # Count non-zero eigenvalues per molecule
    non_zero_counts = np.sum(eigenvals != 0, axis=1)
    
    print(f"   Mean molecule size: {np.mean(non_zero_counts):.2f}")
    print(f"   Median molecule size: {np.median(non_zero_counts):.2f}")
    print(f"   Min/Max molecule size: {np.min(non_zero_counts)}/{np.max(non_zero_counts)}")
    print(f"   90th percentile: {np.percentile(non_zero_counts, 90):.0f}")
    print(f"   95th percentile: {np.percentile(non_zero_counts, 95):.0f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_counts, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(non_zero_counts), color='red', linestyle='--', label=f'Mean: {np.mean(non_zero_counts):.1f}')
    plt.axvline(np.median(non_zero_counts), color='blue', linestyle='--', label=f'Median: {np.median(non_zero_counts):.1f}')
    plt.xlabel('Number of Non-Zero Eigenvalues (Molecule Size)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Molecule Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('molecule_size_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return non_zero_counts

def test_truncation_levels(features, targets, truncation_levels=[10, 15, 20, 25, 30, 50]):
    """Test different truncation levels and their impact on performance."""
    print(f"\n🎯 Testing truncation levels: {truncation_levels}")
    
    # Baseline with all features
    print("   Testing baseline (all 144 eigenvalues + charge/spin)...")
    rf_baseline = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    baseline_scores = cross_val_score(rf_baseline, features, targets, cv=cv, scoring='f1')
    baseline_f1 = baseline_scores.mean()
    
    results = {'baseline': baseline_f1}
    print(f"      Baseline F1: {baseline_f1:.4f} ± {baseline_scores.std():.4f}")
    
    # Extract components
    eigenvals = features[:, :-2]
    metadata = features[:, -2:]  # charge, spin
    non_zero_counts = np.sum(eigenvals != 0, axis=1)
    
    # Test each truncation level
    for k in truncation_levels:
        print(f"   Testing truncation to {k} eigenvalues...")
        
        # Truncate eigenvalues
        eigenvals_truncated = eigenvals[:, :k]
        
        # Add molecule size as feature
        molecule_sizes = non_zero_counts.reshape(-1, 1)
        
        # Combine: truncated eigenvals + molecule size + charge/spin
        X_truncated = np.column_stack([eigenvals_truncated, molecule_sizes, metadata])
        
        # Test performance
        rf_truncated = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        truncated_scores = cross_val_score(rf_truncated, X_truncated, targets, cv=cv, scoring='f1')
        truncated_f1 = truncated_scores.mean()
        
        results[f'trunc_{k}'] = truncated_f1
        improvement = truncated_f1 - baseline_f1
        print(f"      Truncated to {k} F1: {truncated_f1:.4f} ± {truncated_scores.std():.4f} (Δ: {improvement:+.4f})")
    
    return results

def create_improved_features(features, optimal_k=20):
    """Create optimized feature set based on analysis."""
    print(f"\n🔧 Creating optimized features (k={optimal_k})...")
    
    # Extract components
    eigenvals = features[:, :-2]
    metadata = features[:, -2:]  # charge, spin
    
    # Calculate molecule sizes
    non_zero_counts = np.sum(eigenvals != 0, axis=1)
    
    # Truncate eigenvalues
    eigenvals_truncated = eigenvals[:, :optimal_k]
    
    # Add engineered features
    engineered_features = []
    
    # 1. Truncated eigenvalues
    engineered_features.append(eigenvals_truncated)
    
    # 2. Molecule size
    engineered_features.append(non_zero_counts.reshape(-1, 1))
    
    # 3. Eigenvalue statistics (for non-zero values only)
    eigenval_stats = []
    for i, row in enumerate(eigenvals):
        non_zero_vals = row[row != 0]
        if len(non_zero_vals) > 0:
            stats = [
                np.mean(non_zero_vals),
                np.std(non_zero_vals) if len(non_zero_vals) > 1 else 0,
                np.sum(non_zero_vals > 0),  # Count of positive eigenvals
                np.sum(non_zero_vals < 0),  # Count of negative eigenvals
                np.max(non_zero_vals) if len(non_zero_vals) > 0 else 0,
                np.min(non_zero_vals) if len(non_zero_vals) > 0 else 0
            ]
        else:
            stats = [0, 0, 0, 0, 0, 0]
        eigenval_stats.append(stats)
    
    eigenval_stats = np.array(eigenval_stats)
    engineered_features.append(eigenval_stats)
    
    # 4. Original metadata
    engineered_features.append(metadata)
    
    # Combine all features
    X_optimized = np.column_stack(engineered_features)
    
    feature_names = (
        [f'eigenval_{i+1}' for i in range(optimal_k)] +
        ['molecule_size'] +
        ['mean_eigenval', 'std_eigenval', 'pos_count', 'neg_count', 'max_eigenval', 'min_eigenval'] +
        ['charge', 'spin']
    )
    
    print(f"✅ Optimized features created:")
    print(f"   Original shape: {features.shape}")
    print(f"   Optimized shape: {X_optimized.shape}")
    print(f"   Feature reduction: {features.shape[1]} → {X_optimized.shape[1]} ({features.shape[1] - X_optimized.shape[1]} fewer)")
    
    return X_optimized, feature_names

def plot_truncation_results(results):
    """Plot truncation test results."""
    plt.figure(figsize=(12, 6))
    
    # Extract truncation levels and scores
    truncation_levels = []
    scores = []
    colors = []
    
    baseline_score = results['baseline']
    
    for key, score in results.items():
        if key == 'baseline':
            truncation_levels.append(144)  # Full feature set
            colors.append('red')
        else:
            k = int(key.split('_')[1])
            truncation_levels.append(k)
            colors.append('blue' if score > baseline_score else 'orange')
        scores.append(score)
    
    # Sort by truncation level
    sorted_data = sorted(zip(truncation_levels, scores, colors))
    truncation_levels, scores, colors = zip(*sorted_data)
    
    # Plot
    plt.bar(range(len(truncation_levels)), scores, color=colors, alpha=0.7)
    plt.axhline(y=baseline_score, color='red', linestyle='--', alpha=0.8, label=f'Baseline (144 features): {baseline_score:.4f}')
    
    # Labels
    plt.xlabel('Number of Eigenvalue Features')
    plt.ylabel('F1 Score')
    plt.title('Impact of Feature Truncation on Model Performance')
    plt.xticks(range(len(truncation_levels)), 
               [f'{k}\n(+size+meta)' if k != 144 else '144\n(baseline)' for k in truncation_levels])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate best performance
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_k = truncation_levels[best_idx]
    plt.annotate(f'Best: {best_score:.4f}', 
                xy=(best_idx, best_score), 
                xytext=(best_idx, best_score + 0.01),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('truncation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"🏆 Best performing truncation: {best_k} features with F1 = {best_score:.4f}")
    improvement = best_score - baseline_score
    print(f"   Improvement over baseline: {improvement:+.4f} ({improvement/baseline_score*100:+.1f}%)")

# Usage example
def run_truncation_analysis(features, targets):
    """Run complete truncation analysis."""
    print("🚀 Running Eigenvalue Truncation Analysis")
    print("=" * 60)
    
    # Step 1: Analyze molecule sizes
    non_zero_counts = analyze_molecule_sizes(features)
    
    # Step 2: Test truncation levels
    results = test_truncation_levels(features, targets)
    
    # Step 3: Plot results
    plot_truncation_results(results)
    
    # Step 4: Create optimized features with best k
    best_k = max(results.items(), key=lambda x: x[1] if x[0] != 'baseline' else 0)[0]
    if best_k != 'baseline':
        optimal_k = int(best_k.split('_')[1])
        X_optimized, feature_names = create_improved_features(features, optimal_k)
        
        print(f"\n✅ Recommended optimization:")
        print(f"   Use {optimal_k} eigenvalue features + molecule size + metadata")
        print(f"   Expected improvement: {results[best_k] - results['baseline']:+.4f}")
        
        return X_optimized, feature_names
    
    return features, None

def main():
    """Main function to run the analysis with your data."""
    print("🎯 Loading data and running truncation analysis...")
    
    # Load your balanced dataset
    try:
        features = np.load("balanced_features.npy")
        targets = np.load("balanced_targets.npy")
        
        print(f"✅ Loaded data:")
        print(f"   Features shape: {features.shape}")
        print(f"   Targets shape: {targets.shape}")
        
        # Run the complete analysis
        X_optimized, feature_names = run_truncation_analysis(features, targets)
        
        if X_optimized is not None and feature_names is not None:
            # Save optimized features
            np.save("optimized_features.npy", X_optimized)
            
            # Save feature names
            with open("optimized_feature_names.txt", "w") as f:
                for i, name in enumerate(feature_names):
                    f.write(f"{i}: {name}\n")
            
            print(f"\n💾 Saved optimized features:")
            print(f"   optimized_features.npy (shape: {X_optimized.shape})")
            print(f"   optimized_feature_names.txt")
            
            # Quick comparison test with original training script approach
            print(f"\n🔍 Quick performance comparison:")
            
            # Test original features
            rf_original = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            original_scores = cross_val_score(rf_original, features, targets, cv=cv, scoring='f1')
            
            # Test optimized features
            rf_optimized = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            optimized_scores = cross_val_score(rf_optimized, X_optimized, targets, cv=cv, scoring='f1')
            
            print(f"   Original (144 features):  F1 = {original_scores.mean():.4f} ± {original_scores.std():.4f}")
            print(f"   Optimized ({X_optimized.shape[1]} features): F1 = {optimized_scores.mean():.4f} ± {optimized_scores.std():.4f}")
            print(f"   Improvement: {optimized_scores.mean() - original_scores.mean():+.4f}")
            
        else:
            print("⚠️  No improvement found with truncation.")
    
    except FileNotFoundError:
        print("❌ Could not find balanced_features.npy and balanced_targets.npy")
        print("   Make sure these files exist in the current directory.")
        print("   Or modify the file paths in the main() function.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
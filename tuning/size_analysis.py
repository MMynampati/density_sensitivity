#!/usr/bin/env python3
"""
Size-stratified molecular analysis (refactored).
- safer CV checks (n_splits vs. class counts)
- clearer metrics (f1_macro + binary f1 when possible)
- handles missing optimized_features gracefully
- avoids unused imports
- saves plots and summary JSON
"""

import json
import os
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score

RANDOM_STATE = 42
MIN_SAMPLES_PER_SUBSET = 12  # requires at least this many samples to run a quick CV
MIN_SAMPLES_PER_CLASS = 3    # minimal per-class count to allow CV folds

def _safe_cv_scores(estimator, X, y, scoring='f1_macro', n_splits=5, random_state=RANDOM_STATE):
    """
    Returns cross-validated scores safely: chooses n_splits <= min_class_count if possible.
    Returns None if folding not possible due to tiny class counts.
    """
    if X.shape[0] < MIN_SAMPLES_PER_SUBSET:
        return None

    unique, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()

    # Ensure we have at least 2 unique classes
    if len(unique) < 2 or min_class_count < MIN_SAMPLES_PER_CLASS:
        return None

    # Max possible splits given the smallest class
    max_splits = min(n_splits, min_class_count)
    if max_splits < 2:
        return None

    cv = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return scores

def analyze_performance_by_size(features, targets, size_bins=[10, 20, 30, 50, 100]):
    """Analyze model performance across different molecule sizes."""
    print("🔬 Analyzing performance by molecule size...")

    eigenvals = features[:, :-2]
    non_zero_counts = np.sum(eigenvals != 0, axis=1)

    results = OrderedDict()
    size_ranges = []

    # Build bin edges: we will interpret bins as:
    # ≤size_bins[0], size_bins[0]+1..size_bins[1], ..., >size_bins[-1]
    for i in range(len(size_bins)):
        if i == 0:
            range_name = f"≤{size_bins[i]}"
            mask = non_zero_counts <= size_bins[i]
        elif i == len(size_bins) - 1:
            range_name = f">{size_bins[i-1]}"
            mask = non_zero_counts > size_bins[i-1]
        else:
            range_name = f"{size_bins[i-1]+1}-{size_bins[i]}"
            mask = (non_zero_counts > size_bins[i-1]) & (non_zero_counts <= size_bins[i])

        size_ranges.append((range_name, mask))
        n_mask = int(np.sum(mask))

        if n_mask < MIN_SAMPLES_PER_SUBSET:
            print(f"   {range_name:12s}: n={n_mask:4d} (SKIP - too few samples)")
            continue

        X_subset = features[mask]
        y_subset = targets[mask]

        # quick performance test with safety
        rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        scores_macro = _safe_cv_scores(rf, X_subset, y_subset, scoring='f1_macro', n_splits=3)
        # also attempt binary-average F1 if binary labels and possible
        scores_binary = None
        if scores_macro is not None:
            if len(np.unique(y_subset)) == 2:
                scores_binary = _safe_cv_scores(rf, X_subset, y_subset, scoring='f1', n_splits=3)

            results[range_name] = {
                'count': n_mask,
                'f1_macro_mean': float(np.mean(scores_macro)),
                'f1_macro_std': float(np.std(scores_macro)),
                'f1_binary_mean': float(np.mean(scores_binary)) if scores_binary is not None else None,
                'f1_binary_std': float(np.std(scores_binary)) if scores_binary is not None else None,
                'positive_ratio': float(np.mean(y_subset)),
                'size_range': (int(np.min(non_zero_counts[mask])), int(np.max(non_zero_counts[mask])))
            }

            macro_msg = f"F1_macro={results[range_name]['f1_macro_mean']:.4f}±{results[range_name]['f1_macro_std']:.4f}"
            if scores_binary is not None:
                macro_msg += f", F1_binary={results[range_name]['f1_binary_mean']:.4f}±{results[range_name]['f1_binary_std']:.4f}"
            print(f"   {range_name:12s}: {n_mask:4d} molecules, {macro_msg}")

    return results, size_ranges, non_zero_counts

def test_size_filtering_strategies(features, targets):
    """Test different strategies for filtering by molecule size."""
    print("\n🎯 Testing size filtering strategies...")

    eigenvals = features[:, :-2]
    non_zero_counts = np.sum(eigenvals != 0, axis=1)

    strategies = [
        ("baseline", lambda x: np.ones_like(x, dtype=bool)),
        ("≤10", lambda x: x <= 10),
        ("≤20", lambda x: x <= 20),
        ("≤30", lambda x: x <= 30),
        ("≤40", lambda x: x <= 40),
        ("≤50", lambda x: x <= 50),
        (">30", lambda x: x > 30),
        ("5-30", lambda x: (x >= 5) & (x <= 30)),
        ("5-40", lambda x: (x >= 5) & (x <= 40)),
        ("10-50", lambda x: (x >= 10) & (x <= 50))
    ]

    results = OrderedDict()
    optimized_exists = os.path.exists("optimized_features.npy")
    X_opt = None
    if optimized_exists:
        try:
            X_opt = np.load("optimized_features.npy")
            if X_opt.shape[0] != features.shape[0]:
                print("   WARNING: optimized_features.npy length differs from balanced_features.npy; ignoring optimized features.")
                X_opt = None
        except Exception as e:
            print(f"   WARNING: could not load optimized_features.npy: {e}")
            X_opt = None

    for strategy_name, filter_func in strategies:
        mask = filter_func(non_zero_counts)
        n_mask = int(np.sum(mask))
        if n_mask < 50:
            # Too few samples for reliable 5-fold CV
            print(f"   {strategy_name:12s}: n={n_mask:4d} (SKIP - <50 samples)")
            continue

        X_filtered = features[mask]
        y_filtered = targets[mask]

        if len(np.unique(y_filtered)) < 2:
            print(f"   {strategy_name:12s}: n={n_mask:4d} (SKIP - single class after filter)")
            continue

        print(f"   Testing {strategy_name:12s}: {n_mask:4d}/{len(features):4d} molecules ({n_mask/len(features)*100:.1f}%)")

        # Test with original features
        rf_original = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        scores_original = _safe_cv_scores(rf_original, X_filtered, y_filtered, scoring='f1_macro', n_splits=5)
        if scores_original is None:
            print(f"      SKIP {strategy_name} - cannot run CV safely (insufficient per-class samples).")
            continue

        entry = {
            'count': n_mask,
            'percentage': float(n_mask / len(features) * 100.0),
            'f1_original': float(np.mean(scores_original)),
            'f1_original_std': float(np.std(scores_original)),
            'class_balance': float(np.mean(y_filtered))
        }

        # Test with optimized features if available
        if X_opt is not None:
            X_opt_filtered = X_opt[mask]
            try:
                rf_opt = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
                scores_opt = _safe_cv_scores(rf_opt, X_opt_filtered, y_filtered, scoring='f1_macro', n_splits=5)
                if scores_opt is not None:
                    entry.update({
                        'f1_optimized': float(np.mean(scores_opt)),
                        'f1_optimized_std': float(np.std(scores_opt))
                    })
                else:
                    print("      Optimized features: SKIP (insufficient per-class samples for CV).")
            except Exception as e:
                print(f"      Warning: optimized features evaluation failed: {e}")
        else:
            # no optimized features file
            pass

        # store
        results[strategy_name] = entry

        # print summary lines
        print(f"      Original features:  F1_macro={entry['f1_original']:.4f}±{entry['f1_original_std']:.4f}")
        if 'f1_optimized' in entry:
            print(f"      Optimized features: F1_macro={entry['f1_optimized']:.4f}±{entry['f1_optimized_std']:.4f}")

    return results

def analyze_large_molecule_characteristics(features, targets, size_threshold=50):
    """Analyze what makes large molecules different."""
    print(f"\n🔍 Analyzing large molecules (>{size_threshold} atoms)...")

    eigenvals = features[:, :-2]
    metadata = features[:, -2:]
    non_zero_counts = np.sum(eigenvals != 0, axis=1)

    large_mask = non_zero_counts > size_threshold
    small_mask = non_zero_counts <= size_threshold

    n_large = int(np.sum(large_mask))
    n_total = features.shape[0]
    print(f"   Large molecules: {n_large} ({n_large/n_total*100:.1f}%)")
    print(f"   Small molecules: {int(np.sum(small_mask))} ({np.sum(small_mask)/n_total*100:.1f}%)")

    if n_large == 0:
        print("   No large molecules found, skipping further analysis.")
        return

    # Class distribution
    large_positive_rate = float(np.mean(targets[large_mask]))
    small_positive_rate = float(np.mean(targets[small_mask]))
    print(f"   Large molecules - positive rate: {large_positive_rate:.3f}")
    print(f"   Small molecules - positive rate: {small_positive_rate:.3f}")

    # Charge/spin analysis (assumes metadata columns: [charge, spin])
    large_charges = metadata[large_mask, 0]
    small_charges = metadata[small_mask, 0]
    large_spins = metadata[large_mask, 1]
    small_spins = metadata[small_mask, 1]

    print(f"   Large molecules - avg charge: {np.mean(large_charges):.3f}, avg spin: {np.mean(large_spins):.3f}")
    print(f"   Small molecules - avg charge: {np.mean(small_charges):.3f}, avg spin: {np.mean(small_spins):.3f}")

    # Eigenvalue magnitude statistics (mean of absolute non-zero eigenvalues per-molecule)
    def mean_abs_nonzero(rows):
        out_means = []
        for row in rows:
            nonzero = row[row != 0]
            if nonzero.size:
                out_means.append(np.mean(np.abs(nonzero)))
        return out_means

    large_nonzero_means = mean_abs_nonzero(eigenvals[large_mask])
    small_nonzero_means = mean_abs_nonzero(eigenvals[small_mask])

    if large_nonzero_means and small_nonzero_means:
        print(f"   Large molecules - avg |eigenvalue|: {np.mean(large_nonzero_means):.6f}")
        print(f"   Small molecules - avg |eigenvalue|: {np.mean(small_nonzero_means):.6f}")

def create_comparison_plots(size_results, filtering_results, outpath='size_analysis_results.png'):
    """Create visualization plots. Be defensive about missing keys (e.g., baseline)."""
    if not size_results:
        print("No size_results to plot.")
        return
    if not filtering_results:
        print("No filtering_results to plot.")
        return

    plt.figure(figsize=(15, 10))

    # Plot 1: Performance by size (use f1_macro)
    plt.subplot(2, 2, 1)
    size_names = list(size_results.keys())
    size_f1s = [size_results[name]['f1_macro_mean'] for name in size_names]
    size_counts = [size_results[name]['count'] for name in size_names]

    bars = plt.bar(size_names, size_f1s, alpha=0.7)
    plt.xlabel('Molecule Size Range')
    plt.ylabel('F1 (macro)')
    plt.title('Performance by Molecule Size')
    plt.xticks(rotation=45, ha='right')

    for bar, count in zip(bars, size_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'n={count}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Filtering strategies comparison
    plt.subplot(2, 2, 2)
    filter_names = list(filtering_results.keys())
    filter_f1s = []
    for name in filter_names:
        if 'f1_optimized' in filtering_results[name]:
            filter_f1s.append(filtering_results[name]['f1_optimized'])
        else:
            filter_f1s.append(filtering_results[name]['f1_original'])

    filter_counts = [filtering_results[name]['count'] for name in filter_names]
    colors = ['red' if name == 'baseline' else 'blue' for name in filter_names]
    bars = plt.bar(filter_names, filter_f1s, color=colors, alpha=0.7)
    plt.xlabel('Filtering Strategy')
    plt.ylabel('F1 (macro)')
    plt.title('Performance by Filtering Strategy')
    plt.xticks(rotation=45, ha='right')

    for bar, count in zip(bars, filter_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'n={count}', ha='center', va='bottom', fontsize=8)

    # Plot 3: Sample count vs performance
    plt.subplot(2, 2, 3)
    plt.scatter(filter_counts, filter_f1s, alpha=0.7)
    for i, name in enumerate(filter_names):
        plt.annotate(name, (filter_counts[i], filter_f1s[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Number of Molecules')
    plt.ylabel('F1 (macro)')
    plt.title('Sample Size vs Performance')

    # Plot 4: Data retention vs performance improvement (relative to baseline if present)
    plt.subplot(2, 2, 4)
    baseline_f1 = None
    if 'baseline' in filtering_results:
        baseline_f1 = filtering_results['baseline'].get('f1_optimized', filtering_results['baseline']['f1_original'])

    # Only include non-baseline strategies
    retentions = []
    improvements = []
    strategy_names = []
    for name, f1 in zip(filter_names, filter_f1s):
        if name == 'baseline' or baseline_f1 is None:
            continue
        retentions.append(filtering_results[name]['percentage'])
        improvements.append(f1 - baseline_f1)
        strategy_names.append(name)

    if retentions:
        plt.scatter(retentions, improvements, alpha=0.7)
        for i, name in enumerate(strategy_names):
            plt.annotate(name, (retentions[i], improvements[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Data Retention (%)')
        plt.ylabel('F1 (macro) Improvement vs baseline')
        plt.title('Data Retention vs Performance Gain')
    else:
        plt.text(0.1, 0.5, 'No baseline or no competing strategies to plot', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("🧬 Size-Stratified Molecular Analysis")
    print("=" * 60)
    try:
        features = np.load("balanced_features.npy")
        targets = np.load("balanced_targets.npy")
        print(f"✅ Loaded data: {features.shape[0]} molecules")

        size_results, size_ranges, non_zero_counts = analyze_performance_by_size(features, targets)
        filtering_results = test_size_filtering_strategies(features, targets)
        analyze_large_molecule_characteristics(features, targets)
        create_comparison_plots(size_results, filtering_results)

        # Save summary JSON
        summary = {
            'size_results': size_results,
            'filtering_results': filtering_results
        }
        with open('size_analysis_summary.json', 'w') as fh:
            json.dump(summary, fh, indent=2)
        print("Saved summary to size_analysis_summary.json")

        # Recommendations (simple)
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATIONS\n")
        if size_results:
            for size_range, res in size_results.items():
                f1 = res['f1_macro_mean']
                count = res['count']
                if f1 > 0.8:
                    perf = "EXCELLENT"
                elif f1 > 0.7:
                    perf = "GOOD"
                elif f1 > 0.5:
                    perf = "MODERATE"
                else:
                    perf = "POOR"
                print(f"  {size_range:12s}: F1_macro={f1:.4f} ({count} molecules) -> {perf}")

            f1_scores = [r['f1_macro_mean'] for r in size_results.values()]
            if max(f1_scores) - min(f1_scores) > 0.3:
                print("\n  ⚠️  Large performance variation across size ranges detected. Consider separate models per range.")

        if filtering_results:
            best_strategy = max(filtering_results.items(),
                                key=lambda x: x[1].get('f1_optimized', x[1]['f1_original']))
            best_f1 = best_strategy[1].get('f1_optimized', best_strategy[1]['f1_original'])
            print(f"\n  🏆 Best filtering strategy: {best_strategy[0]} (F1_macro={best_f1:.4f}, retention={best_strategy[1]['percentage']:.1f}%)")

        print("\n  Suggested next steps:")
        print("   - If you saw class imbalance issues, try class_weight or resampling (SMOTE/undersample).")
        print("   - If optimized features help, pipeline & tune per-size-range.")
        print("   - Persist per-fold results for statistical testing if you need significance tests.")
    except FileNotFoundError as e:
        print("❌ Could not find required data files (balanced_features.npy / balanced_targets.npy).")
        print(e)
    except Exception as e:
        print("❌ Unexpected error:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

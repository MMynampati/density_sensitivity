#!/usr/bin/env python3
"""
Analyze sets that have over 10% of the minority class.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_minority_class():
    """
    Find sets with over 10% minority class representation.
    """
    print("🔍 Analyzing sets with over 10% minority class...")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"minority_class_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Created results directory: {results_dir}")
    
    # Load the summary data from the results folder
    summary_file = "per_set_evaluation_results/per_set_evaluation_summary.csv"
    if not os.path.exists(summary_file):
        print(f"❌ Summary file not found: {summary_file}")
        print("Please run evaluate_by_set.py first to generate the summary data.")
        return
    
    df = pd.read_csv(summary_file)
    
    # Calculate minority class percentage for each set
    minority_percentages = []
    minority_class_info = []
    
    for _, row in df.iterrows():
        set_name = row['Set']
        total_samples = row['Samples']
        sensitive = row['Sensitive']
        not_sensitive = row['Not_Sensitive']
        
        # Determine minority class
        if sensitive == 0 or not_sensitive == 0:
            # Single class - no minority class
            minority_percentage = 0
            minority_class = "None (single class)"
        else:
            # Mixed classes - find minority
            if sensitive < not_sensitive:
                minority_count = sensitive
                minority_class = "Sensitive"
            else:
                minority_count = not_sensitive
                minority_class = "Not Sensitive"
            
            minority_percentage = (minority_count / total_samples) * 100
        
        minority_percentages.append(minority_percentage)
        minority_class_info.append(minority_class)
    
    # Add columns to dataframe
    df['Minority_Percentage'] = minority_percentages
    df['Minority_Class'] = minority_class_info
    
    # Filter sets with over 10% minority class
    balanced_sets = df[df['Minority_Percentage'] > 10].copy()
    balanced_sets = balanced_sets.sort_values('Minority_Percentage', ascending=False)
    
    print(f"\n📊 Results:")
    print(f"   Total sets: {len(df)}")
    print(f"   Sets with >10% minority class: {len(balanced_sets)}")
    print(f"   Sets with single class only: {len(df[df['Minority_Percentage'] == 0])}")
    
    if len(balanced_sets) > 0:
        print(f"\n🏆 Sets with over 10% minority class:")
        print("=" * 80)
        print(f"{'Set':<12} {'Samples':<8} {'Sensitive':<10} {'Not_Sensitive':<13} {'Minority%':<10} {'Minority_Class':<15} {'Accuracy':<8} {'F1_Score':<8}")
        print("-" * 80)
        
        for _, row in balanced_sets.iterrows():
            print(f"{row['Set']:<12} {row['Samples']:<8} {row['Sensitive']:<10} {row['Not_Sensitive']:<13} "
                  f"{row['Minority_Percentage']:<10.1f} {row['Minority_Class']:<15} {row['Accuracy']:<8.3f} {row['F1_Score']:<8.3f}")
        
        # Save results
        balanced_path = os.path.join(results_dir, "balanced_sets_over_10_percent.csv")
        balanced_sets.to_csv(balanced_path, index=False)
        print(f"\n✅ Saved results to: {balanced_path}")
        
        # Summary statistics for balanced sets
        print(f"\n📈 Summary for balanced sets (>10% minority class):")
        print(f"   Average accuracy: {balanced_sets['Accuracy'].mean():.3f} ± {balanced_sets['Accuracy'].std():.3f}")
        print(f"   Average F1-score: {balanced_sets['F1_Score'].mean():.3f} ± {balanced_sets['F1_Score'].std():.3f}")
        print(f"   Average minority percentage: {balanced_sets['Minority_Percentage'].mean():.1f}%")
        
        # Best performing balanced sets
        print(f"\n🥇 Top 5 balanced sets by accuracy:")
        top_balanced = balanced_sets.nlargest(5, 'Accuracy')[['Set', 'Samples', 'Minority_Percentage', 'Accuracy', 'F1_Score']]
        print(top_balanced.to_string(index=False))
        
        # Best performing balanced sets by F1-score
        print(f"\n🥇 Top 5 balanced sets by F1-score:")
        top_f1_balanced = balanced_sets.nlargest(5, 'F1_Score')[['Set', 'Samples', 'Minority_Percentage', 'Accuracy', 'F1_Score']]
        print(top_f1_balanced.to_string(index=False))
        
    else:
        print("\n⚠️  No sets found with over 10% minority class representation.")
    
    # Also show sets with exactly 10% or close to it
    print(f"\n📊 Sets with 5-15% minority class (near-balanced):")
    near_balanced = df[(df['Minority_Percentage'] >= 5) & (df['Minority_Percentage'] <= 15) & (df['Minority_Percentage'] > 0)].copy()
    near_balanced = near_balanced.sort_values('Minority_Percentage', ascending=False)
    
    if len(near_balanced) > 0:
        print(f"{'Set':<12} {'Samples':<8} {'Minority%':<10} {'Minority_Class':<15} {'Accuracy':<8} {'F1_Score':<8}")
        print("-" * 70)
        for _, row in near_balanced.iterrows():
            print(f"{row['Set']:<12} {row['Samples']:<8} {row['Minority_Percentage']:<10.1f} {row['Minority_Class']:<15} {row['Accuracy']:<8.3f} {row['F1_Score']:<8.3f}")
    else:
        print("   No sets found in the 5-15% range.")

if __name__ == "__main__":
    analyze_minority_class()

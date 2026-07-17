"""
Main pipeline - run everything from start to finish

This is the main file I run.
"""

import os
import sys
import time

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_data import generate_customers
from clean_data import clean_customers, prepare_for_clustering
from clustering import segment_customers, visualize_segments, find_best_k
from ab_test import assign_variants, simulate_test, analyze_test, make_recommendations
from visualize import plot_distributions, plot_segment_radar, plot_ab_results, plot_dashboard

def run_pipeline(n_customers=5000, k_clusters=4):
    """
    Run the entire pipeline
    
    Parameters:
    - n_customers: Number of customers to generate
    - k_clusters: Number of segments (I usually use 4)
    """
    
    print("="*80)
    print(" E-COMMERCE CUSTOMER ANALYTICS PIPELINE")
    print("="*80)
    print(f"\n Starting pipeline with {n_customers} customers...")
    
    # Keep track of timing
    start_time = time.time()
    
    # ====================================================================
    # STEP 1: Generate synthetic data
    # ====================================================================
    print("\n[1/5] Generating customer data...")
    raw_df = generate_customers(n=n_customers)
    
    # ====================================================================
    # STEP 2: Clean the data
    # ====================================================================
    print("\n[2/5] Cleaning data...")
    clean_df = clean_customers(raw_df)
    X_scaled, scaler = prepare_for_clustering(clean_df)
    
    # ====================================================================
    # STEP 3: Customer segmentation
    # ====================================================================
    print("\n[3/5] Running segmentation...")
    segmented_df, kmeans = segment_customers(clean_df, X_scaled, k=k_clusters)
    visualize_segments(segmented_df, X_scaled)
    
    # ====================================================================
    # STEP 4: A/B testing
    # ====================================================================
    print("\n[4/5] Running A/B test...")
    assigned_df = assign_variants(segmented_df)
    test_df = simulate_test(assigned_df)
    ab_results = analyze_test(test_df)
    recommendations = make_recommendations(ab_results)
    
    # ====================================================================
    # STEP 5: Visualizations
    # ====================================================================
    print("\n[5/5] Generating visualizations...")
    plot_distributions(clean_df)
    plot_segment_radar(segmented_df)
    plot_ab_results(ab_results)
    plot_dashboard(segmented_df, ab_results)
    
    # ====================================================================
    # Wrap up
    # ====================================================================
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(" PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n Total time: {elapsed_time:.2f} seconds")
    print(f" Customers: {n_customers}")
    print(f" Segments: {k_clusters}")
    print("\n Outputs saved to:")
    print("   - data/outputs/figures/  (all charts)")
    print("   - data/outputs/results/  (CSV files)")
    print("   - data/processed/        (cleaned data)")
    
    return {
        'raw_data': raw_df,
        'cleaned_data': clean_df,
        'segmented_data': segmented_df,
        'ab_results': ab_results,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    # Run the pipeline
    results = run_pipeline(n_customers=5000, k_clusters=4)
"""
Customer segmentation using K-Means
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import print_header, save_csv, get_figures_path, get_processed_path

def segment_customers(df, X_scaled, k=4):
    """Do the actual clustering"""
    
    print_header(f"Clustering with K={k}")
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['segment'] = kmeans.fit_predict(X_scaled)
    
    # Name the segments based on behavior
    features = ['total_spend', 'days_inactive', 'logins_30d']
    segment_means = df.groupby('segment')[features].mean()
    
    print("\nSegment profiles:")
    print(segment_means.round(2))
    
    def name_segment(profile):
        spend = profile['total_spend']
        inactive = profile['days_inactive']
        logins = profile['logins_30d']
        
        if spend > 1000 and inactive < 20:
            return 'High Value'
        elif spend > 300 and inactive < 40:
            return 'Loyal'
        elif inactive > 60:
            return 'At-Risk'
        else:
            return 'Occasional'
    
    seg_names = {}
    for seg in df['segment'].unique():
        profile = segment_means.loc[seg]
        seg_names[seg] = name_segment(profile)
    
    df['segment_name'] = df['segment'].map(seg_names)
    
    print("\nSegment distribution:")
    print(df['segment_name'].value_counts())
    
    # Save segmented data
    processed_path = get_processed_path()
    filepath = os.path.join(processed_path, 'customers_segmented.csv')
    df.to_csv(filepath, index=False)
    print(f" Saved: {filepath}")
    
    return df, kmeans

def visualize_segments(df, X_scaled):
    """Plot the segments using PCA"""
    
    print_header("Visualizing Segments")
    
    figures_path = get_figures_path()
    
    # PCA for 2D projection
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['pca_x'] = pca_result[:, 0]
    df['pca_y'] = pca_result[:, 1]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    colors = {
        'High Value': '#2E86AB',
        'Loyal': '#A23B72',
        'At-Risk': '#F18F01',
        'Occasional': '#C73E1D'
    }
    
    for seg_name in df['segment_name'].unique():
        subset = df[df['segment_name'] == seg_name]
        plt.scatter(subset['pca_x'], subset['pca_y'], 
                   label=seg_name, 
                   color=colors.get(seg_name, '#888888'),
                   alpha=0.6, s=40, edgecolor='white', linewidth=0.5)
    
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('Customer Segments', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(figures_path, 'segments_2d.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {filepath}")
    
    # Heatmap of segment characteristics
    features = ['total_spend', 'days_inactive', 'logins_30d', 'avg_order_value']
    heatmap_data = df.groupby('segment_name')[features].mean()
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Value'})
    plt.title('Segment Characteristics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(figures_path, 'segment_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {filepath}")

def find_best_k(X_scaled, min_k=2, max_k=7):
    """Find optimal number of clusters"""
    
    print_header("Finding Optimal Clusters")
    
    figures_path = get_figures_path()
    
    inertias = []
    sil_scores = []
    
    for k in range(min_k, max_k + 1):
        print(f"Testing K={k}...", end=' ')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        sil_scores.append(sil_score)
        print(f"Silhouette: {sil_score:.3f}")
    
    best_idx = np.argmax(sil_scores)
    best_k = range(min_k, max_k + 1)[best_idx]
    
    print(f"\n Best K: {best_k} (Silhouette: {sil_scores[best_idx]:.3f})")
    
    # Plot elbow and silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(range(min_k, max_k + 1), inertias, 'bo-')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].axvline(x=best_k, color='red', linestyle='--')
    
    axes[1].plot(range(min_k, max_k + 1), sil_scores, 'ro-')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].axvline(x=best_k, color='red', linestyle='--')
    
    plt.tight_layout()
    
    filepath = os.path.join(figures_path, 'optimal_k.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {filepath}")
    
    return best_k, inertias, sil_scores

if __name__ == "__main__":
    from generate_data import generate_customers
    from clean_data import clean_customers, prepare_for_clustering
    
    raw = generate_customers(2000)
    clean = clean_customers(raw)
    X_scaled, _ = prepare_for_clustering(clean)
    
    segmented_df, kmeans = segment_customers(clean, X_scaled, k=4)
    visualize_segments(segmented_df, X_scaled)
    
    print("\n Clustering complete!")
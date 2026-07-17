"""
Create all the visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from helpers import print_header, get_figures_path

def plot_distributions(df):
    """Plot distributions of all features"""
    
    print_header("Creating Distribution Plots")
    
    figures_path = get_figures_path()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Customer Data Distributions', fontsize=16, fontweight='bold')
    
    # Total spend
    ax = axes[0, 0]
    ax.hist(df['total_spend'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.axvline(df['total_spend'].mean(), color='red', linestyle='--', 
               label=f'Mean: ${df["total_spend"].mean():.0f}')
    ax.axvline(df['total_spend'].median(), color='green', linestyle='--', 
               label=f'Median: ${df["total_spend"].median():.0f}')
    ax.set_xlabel('Total Spend ($)')
    ax.set_ylabel('Count')
    ax.set_title('Total Spend')
    ax.legend()
    ax.set_xlim(0, 2000)
    
    # Days inactive
    ax = axes[0, 1]
    ax.hist(df['days_inactive'], bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
    ax.axvline(df['days_inactive'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["days_inactive"].mean():.1f}')
    ax.set_xlabel('Days Inactive')
    ax.set_ylabel('Count')
    ax.set_title('Recency')
    ax.legend()
    
    # Logins
    ax = axes[0, 2]
    ax.hist(df['logins_30d'], bins=20, color='#F18F01', alpha=0.7, edgecolor='black')
    ax.axvline(df['logins_30d'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["logins_30d"].mean():.1f}')
    ax.set_xlabel('Logins (30 days)')
    ax.set_ylabel('Count')
    ax.set_title('Login Frequency')
    ax.legend()
    
    # Boxplots
    ax = axes[1, 0]
    df.boxplot(column='total_spend', ax=ax)
    ax.set_title('Spend Outliers')
    
    ax = axes[1, 1]
    df.boxplot(column='days_inactive', ax=ax)
    ax.set_title('Recency Outliers')
    
    ax = axes[1, 2]
    df.boxplot(column='logins_30d', ax=ax)
    ax.set_title('Login Outliers')
    
    plt.tight_layout()
    
    filepath = os.path.join(figures_path, 'distributions.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {filepath}")

def plot_segment_radar(df):
    """Radar chart for segment comparison"""
    
    print_header("Creating Radar Chart")
    
    figures_path = get_figures_path()
    
    features = ['total_spend', 'days_inactive', 'logins_30d', 'avg_order_value']
    segment_means = df.groupby('segment_name')[features].mean()
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(segment_means)
    scaled_df = pd.DataFrame(scaled, columns=features, index=segment_means.index)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, seg in enumerate(scaled_df.index):
        values = scaled_df.loc[seg].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=seg, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1)
    ax.set_title('Customer Segment Profiles', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    filepath = os.path.join(figures_path, 'segment_radar.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {filepath}")

def plot_ab_results(results_df):
    """A/B test results visualization"""
    
    print_header("Plotting A/B Test Results")
    
    figures_path = get_figures_path()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('A/B Test Results by Segment', fontsize=14, fontweight='bold')
    
    # Left: Conversion rates
    ax = axes[0]
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['control_conv'], width, 
           label='Control', color='#A23B72', alpha=0.7)
    ax.bar(x + width/2, results_df['treatment_conv'], width, 
           label='Treatment', color='#2E86AB', alpha=0.7)
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Conversion Rate')
    ax.set_title('Conversion Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['segment'], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Lift
    ax = axes[1]
    colors = ['green' if x > 0 else 'red' for x in results_df['lift_pct']]
    ax.bar(results_df['segment'], results_df['lift_pct'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Lift (%)')
    ax.set_title('Treatment Lift')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, row in results_df.iterrows():
        if row['significant']:
            label = f'p={row["p_adjusted"]:.3f}*'
            ax.text(i, row['lift_pct'] + 2, label, ha='center', va='bottom', fontsize=9, color='green')
    
    plt.tight_layout()
    
    filepath = os.path.join(figures_path, 'ab_results.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {filepath}")

def plot_dashboard(df, results_df):
    """Create a business dashboard"""
    
    print_header("Creating Dashboard")
    
    figures_path = get_figures_path()
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Customer Analytics Dashboard', fontsize=18, fontweight='bold')
    
    grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Segment distribution
    ax = fig.add_subplot(grid[0, 0])
    segment_counts = df['segment_name'].value_counts()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    ax.pie(segment_counts.values, labels=segment_counts.index, 
           autopct='%1.1f%%', colors=colors[:len(segment_counts)], 
           textprops={'fontsize': 11, 'weight': 'bold'})
    ax.set_title('Customer Segments', fontsize=13, fontweight='bold')
    
    # 2. Key metrics table
    ax = fig.add_subplot(grid[0, 1])
    ax.axis('off')
    metrics = df.groupby('segment_name')[['total_spend', 'logins_30d', 'days_inactive']].mean().round(1)
    metrics.columns = ['Avg Spend', 'Avg Logins', 'Avg Days Inactive']
    
    table = ax.table(cellText=metrics.values,
                     rowLabels=metrics.index,
                     colLabels=metrics.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#2E86AB']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Segment Metrics', fontsize=13, fontweight='bold')
    
    # 3. Recommendations
    ax = fig.add_subplot(grid[1, 0])
    ax.axis('off')
    
    rec_text = ["RECOMMENDATIONS:\n"]
    for _, row in results_df.iterrows():
        if row['significant'] and row['lift_pct'] > 0:
            rec_text.append(f"DEPLOY: {row['segment']} (lift: {row['lift_pct']:.1f}%)")
        elif row['significant'] and row['lift_pct'] < 0:
            rec_text.append(f"AVOID: {row['segment']} (lift: {row['lift_pct']:.1f}%)")
        else:
            rec_text.append(f"RE-EVALUATE: {row['segment']} (lift: {row['lift_pct']:.1f}%)")
    
    ax.text(0.1, 0.5, '\n'.join(rec_text), fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', alpha=0.8))
    ax.set_title('Business Recommendations', fontsize=13, fontweight='bold')
    
    # 4. Key insights
    ax = fig.add_subplot(grid[1, 1])
    ax.axis('off')
    
    insights = ["KEY INSIGHTS:\n"]
    for _, row in results_df.iterrows():
        if row['significant'] and row['lift_pct'] > 20:
            insights.append(f"* '{row['segment']}' customers show +{row['lift_pct']:.0f}% lift")
            insights.append(f"  -> Send re-engagement offers immediately")
            insights.append("")
    
    if len(insights) <= 1:
        insights = ["KEY INSIGHTS:\n", "No significant high-lift results found.", "Consider running test longer."]
    
    ax.text(0.1, 0.5, '\n'.join(insights), fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8', alpha=0.8))
    ax.set_title('Key Insights', fontsize=13, fontweight='bold')
    
    fig.set_layout_engine('constrained')
    
    filepath = os.path.join(figures_path, 'dashboard.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved: {filepath}")

if __name__ == "__main__":
    from generate_data import generate_customers
    from clean_data import clean_customers, prepare_for_clustering
    from clustering import segment_customers
    from ab_test import assign_variants, simulate_test, analyze_test
    
    raw = generate_customers(2000)
    clean = clean_customers(raw)
    X_scaled, _ = prepare_for_clustering(clean)
    segmented_df, _ = segment_customers(clean, X_scaled, k=4)
    
    assigned = assign_variants(segmented_df)
    test_df = simulate_test(assigned)
    results = analyze_test(test_df)
    
    plot_distributions(clean)
    plot_segment_radar(segmented_df)
    plot_ab_results(results)
    plot_dashboard(segmented_df, results)
    
    print("\n All visualizations complete!")
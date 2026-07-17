"""
A/B Testing with segment-specific effects
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from helpers import print_header, save_csv, get_results_path

def assign_variants(df, split=0.5):
    """Randomly assign customers to control/treatment"""
    
    print_header("Assigning A/B Test Variants")
    
    np.random.seed(123)
    
    df['variant'] = np.random.choice(['Control', 'Treatment'], 
                                     size=len(df), 
                                     p=[1-split, split])
    
    print(f"Control: {len(df[df['variant'] == 'Control'])}")
    print(f"Treatment: {len(df[df['variant'] == 'Treatment'])}")
    
    return df

def simulate_test(df):
    """Simulate A/B test results with segment-specific effects"""
    
    print_header("Simulating A/B Test")
    
    # Base conversion rates per segment
    base_conv = {
        'High Value': 0.30,
        'Loyal': 0.22,
        'At-Risk': 0.08,
        'Occasional': 0.12
    }
    
    # Treatment effects per segment
    treatment_effects = {
        'High Value': -0.02,  # Coupon hurts high value
        'Loyal': 0.04,        # Small positive
        'At-Risk': 0.10,      # BIG positive - brings them back
        'Occasional': 0.01    # Almost no effect
    }
    
    def get_conversion(row):
        seg = row['segment_name']
        base = base_conv[seg]
        
        if row['variant'] == 'Treatment':
            prob = base + treatment_effects[seg]
        else:
            prob = base
        
        # Add random noise
        prob = np.clip(prob + np.random.normal(0, 0.015), 0.01, 0.99)
        
        return np.random.binomial(1, prob)
    
    df['converted'] = df.apply(get_conversion, axis=1)
    
    # Check results
    overall_conv = df['converted'].mean()
    control_conv = df[df['variant'] == 'Control']['converted'].mean()
    treatment_conv = df[df['variant'] == 'Treatment']['converted'].mean()
    
    print(f"Overall conversion: {overall_conv:.2%}")
    print(f"Control: {control_conv:.2%}")
    print(f"Treatment: {treatment_conv:.2%}")
    print(f"Overall lift: {((treatment_conv - control_conv) / control_conv * 100):.1f}%")
    
    return df

def analyze_test(df):
    """Analyze the A/B test results"""
    
    print_header("Analyzing A/B Test Results")
    
    results = []
    segments = df['segment_name'].unique()
    
    for seg in segments:
        seg_data = df[df['segment_name'] == seg]
        
        control = seg_data[seg_data['variant'] == 'Control']
        treatment = seg_data[seg_data['variant'] == 'Treatment']
        
        n_control = len(control)
        n_treatment = len(treatment)
        
        conv_control = control['converted'].mean()
        conv_treatment = treatment['converted'].mean()
        lift = ((conv_treatment - conv_control) / conv_control) * 100
        
        # T-test
        t_stat, p_val = stats.ttest_ind(control['converted'], treatment['converted'])
        
        # Bonferroni correction
        p_adj = min(p_val * len(segments), 1.0)
        
        results.append({
            'segment': seg,
            'n_control': n_control,
            'n_treatment': n_treatment,
            'control_conv': conv_control,
            'treatment_conv': conv_treatment,
            'lift_pct': lift,
            'p_value': p_val,
            'p_adjusted': p_adj,
            'significant': p_adj < 0.05
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nResults by segment:")
    print(results_df[['segment', 'lift_pct', 'p_adjusted', 'significant']].to_string(index=False))
    
    return results_df

def make_recommendations(results_df):
    """Generate business recommendations"""
    
    print_header("Recommendations")
    
    for _, row in results_df.iterrows():
        seg = row['segment']
        lift = row['lift_pct']
        
        if row['significant'] and lift > 0:
            print(f"\n At-Risk: DEPLOY")
            print(f"   -> Lift: {lift:.1f}%")
            print(f"   -> Roll out treatment to {seg} customers")
        elif row['significant'] and lift < 0:
            print(f"\n {seg}: AVOID")
            print(f"   -> Lift: {lift:.1f}%")
            print(f"   -> Do NOT use treatment for {seg} customers")
        else:
            print(f"\n {seg}: INCONCLUSIVE")
            print(f"   -> Lift: {lift:.1f}%")
            print(f"   -> Run test longer or try different treatment")
    
    # Save results
    results_path = get_results_path()
    filepath = os.path.join(results_path, 'ab_test_results.csv')
    results_df.to_csv(filepath, index=False)
    print(f"\n Saved: {filepath}")
    
    return results_df

if __name__ == "__main__":
    from generate_data import generate_customers
    from clean_data import clean_customers, prepare_for_clustering
    from clustering import segment_customers
    
    raw = generate_customers(2000)
    clean = clean_customers(raw)
    X_scaled, _ = prepare_for_clustering(clean)
    segmented_df, _ = segment_customers(clean, X_scaled, k=4)
    
    assigned = assign_variants(segmented_df)
    test_results = simulate_test(assigned)
    analysis = analyze_test(test_results)
    recommendations = make_recommendations(analysis)
    
    print("\n A/B test complete!")
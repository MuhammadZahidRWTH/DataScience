"""
Generate fake customer data
"""

import pandas as pd
import numpy as np
import os
from helpers import set_seed, print_header, save_csv, get_raw_path

def generate_customers(n=5000):
    """
    Create customer dataset with realistic distributions
    
    I'm creating 4 distinct segments intentionally:
    - High Value: High spend, recent, frequent logins
    - Loyal: Medium-high spend, somewhat recent, moderate logins
    - At-Risk: Previously high spend, now dormant
    - Occasional: Low spend, infrequent
    """
    
    set_seed(42)
    
    print_header("Generating Customer Data")
    print(f"Creating {n} customers...")
    
    # I'll create each segment separately and combine
    # This gives me clear, distinct segments
    
    n_high = int(n * 0.10)      # 10% High Value
    n_loyal = int(n * 0.20)     # 20% Loyal
    n_atrisk = int(n * 0.15)    # 15% At-Risk
    n_occasional = n - n_high - n_loyal - n_atrisk  # 55% Occasional
    
    print(f"Segment allocation:")
    print(f"   - High Value: {n_high} (10%)")
    print(f"   - Loyal: {n_loyal} (20%)")
    print(f"   - At-Risk: {n_atrisk} (15%)")
    print(f"   - Occasional: {n_occasional} (55%)")
    
    # 1. High Value customers
    high_spend = np.random.uniform(1500, 4000, n_high)
    high_inactive = np.random.exponential(3, n_high).clip(1, 14).astype(int)
    high_logins = np.random.poisson(20, n_high).clip(10, 50).astype(int)
    high_avg_order = high_spend / np.random.poisson(5, n_high).clip(1, 20)
    
    # 2. Loyal customers
    loyal_spend = np.random.uniform(300, 1200, n_loyal)
    loyal_inactive = np.random.exponential(10, n_loyal).clip(1, 30).astype(int)
    loyal_logins = np.random.poisson(10, n_loyal).clip(5, 30).astype(int)
    loyal_avg_order = loyal_spend / np.random.poisson(3, n_loyal).clip(1, 10)
    
    # 3. At-Risk customers (used to spend, now dormant)
    atrisk_spend = np.random.uniform(400, 1500, n_atrisk)
    atrisk_inactive = np.random.uniform(60, 150, n_atrisk).astype(int)
    atrisk_logins = np.random.poisson(2, n_atrisk).clip(0, 8).astype(int)
    atrisk_avg_order = atrisk_spend / np.random.poisson(3, n_atrisk).clip(1, 8)
    
    # 4. Occasional customers (low engagement)
    occ_spend = np.random.uniform(10, 200, n_occasional)
    occ_inactive = np.random.exponential(20, n_occasional).clip(5, 90).astype(int)
    occ_logins = np.random.poisson(3, n_occasional).clip(0, 12).astype(int)
    occ_avg_order = occ_spend / np.random.poisson(2, n_occasional).clip(1, 5)
    
    # Combine everything
    customer_ids = range(1, n + 1)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'total_spend': np.concatenate([high_spend, loyal_spend, atrisk_spend, occ_spend]).round(2),
        'days_inactive': np.concatenate([high_inactive, loyal_inactive, atrisk_inactive, occ_inactive]),
        'avg_order_value': np.concatenate([high_avg_order, loyal_avg_order, atrisk_avg_order, occ_avg_order]).round(2),
        'logins_30d': np.concatenate([high_logins, loyal_logins, atrisk_logins, occ_logins]).astype(int)
    })
    
    # Shuffle so segments are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add some missing values
    missing_idx = np.random.choice(df.index, size=80, replace=False)
    for idx in missing_idx:
        col = np.random.choice(['avg_order_value', 'logins_30d'])
        df.loc[idx, col] = np.nan
    
    print(f"\nGenerated {len(df)} customers")
    print(f"   - Spend range: ${df['total_spend'].min():.0f} to ${df['total_spend'].max():.0f}")
    print(f"   - Days inactive: {df['days_inactive'].min()} to {df['days_inactive'].max()}")
    print(f"   - Missing values: {df.isnull().sum().sum()} total")
    
    # Save to raw folder
    raw_path = get_raw_path()
    filepath = os.path.join(raw_path, 'customers_raw.csv')
    df.to_csv(filepath, index=False)
    print(f" Saved: {filepath}")
    
    return df

if __name__ == "__main__":
    df = generate_customers(1000)
    print("\nFirst 5 rows:")
    print(df.head())
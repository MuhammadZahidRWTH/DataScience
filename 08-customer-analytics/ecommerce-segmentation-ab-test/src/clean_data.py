"""
Clean the data - this is where I spend most of my time
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from helpers import print_header, save_csv, get_processed_path

def clean_customers(df):
    """Clean the customer data"""
    
    print_header("Cleaning Customer Data")
    
    df_clean = df.copy()
    
    # 1. Check missing values
    print("\n1. Checking missing values...")
    missing_before = df_clean.isnull().sum()
    print(f"   Missing: {missing_before[missing_before > 0]}")
    
    # Fill numeric missing with median
    for col in ['avg_order_value', 'logins_30d']:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"   - Filled {col} with median: {median_val:.2f}")
    
    # 2. Handle outliers using IQR method
    print("\n2. Handling outliers...")
    
    for col in ['total_spend', 'avg_order_value']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 3 * IQR
        
        outliers = (df_clean[col] > upper).sum()
        if outliers > 0:
            df_clean[col] = df_clean[col].clip(upper=upper)
            print(f"   - Capped {outliers} outliers in {col} at {upper:.2f}")
    
    # 3. Create RFM-inspired features
    print("\n3. Creating features...")
    
    # Recency score: lower days = higher score
    try:
        df_clean['recency_score'] = pd.qcut(df_clean['days_inactive'], 
                                            q=4, 
                                            labels=['4', '3', '2', '1']).astype(int)
    except ValueError:
        print("   - Using rank-based method for recency_score")
        df_clean['recency_score'] = pd.qcut(df_clean['days_inactive'].rank(method='first'), 
                                            q=4, 
                                            labels=['4', '3', '2', '1']).astype(int)
    
    # Frequency score: more logins = higher score
    try:
        df_clean['frequency_score'] = pd.qcut(df_clean['logins_30d'], 
                                              q=4, 
                                              labels=['1', '2', '3', '4']).astype(int)
    except ValueError:
        print("   - Using rank-based method for frequency_score")
        df_clean['frequency_score'] = pd.qcut(df_clean['logins_30d'].rank(method='first'), 
                                              q=4, 
                                              labels=['1', '2', '3', '4']).astype(int)
    
    # Monetary score: higher spend = higher score
    try:
        df_clean['monetary_score'] = pd.qcut(df_clean['total_spend'], 
                                             q=4, 
                                             labels=['1', '2', '3', '4']).astype(int)
    except ValueError:
        print("   - Using rank-based method for monetary_score")
        df_clean['monetary_score'] = pd.qcut(df_clean['total_spend'].rank(method='first'), 
                                             q=4, 
                                             labels=['1', '2', '3', '4']).astype(int)
    
    df_clean['rfm_score'] = (df_clean['recency_score'] + 
                             df_clean['frequency_score'] + 
                             df_clean['monetary_score'])
    
    print(f"   - Added recency, frequency, monetary scores")
    print(f"   - RFM score ranges from 3 to 12")
    
    # Save cleaned data
    processed_path = get_processed_path()
    filepath = os.path.join(processed_path, 'customers_cleaned.csv')
    df_clean.to_csv(filepath, index=False)
    print(f" Saved: {filepath}")
    
    return df_clean

def prepare_for_clustering(df, features=None):
    """Scale features for clustering"""
    
    if features is None:
        features = ['total_spend', 'days_inactive', 'logins_30d', 'avg_order_value']
    
    print(f"\nPreparing {len(features)} features for clustering...")
    
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   - Scaled shape: {X_scaled.shape}")
    
    return X_scaled, scaler

if __name__ == "__main__":
    from generate_data import generate_customers
    
    raw = generate_customers(1000)
    clean = clean_customers(raw)
    X_scaled, scaler = prepare_for_clustering(clean)
    print("\n Cleaning complete!")
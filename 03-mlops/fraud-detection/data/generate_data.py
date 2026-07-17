import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os
import sys

# generates synthetic fraud transaction data
# using make_classification as the base since we don't have real bank data
# tweaked weights to get roughly 2% fraud rate which matches industry averages
# ref: https://www.nilsonreport.com/fraud-statistics

def generate_fraud_data(n_samples=10000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.98, 0.02],
        flip_y=0.01,
        random_state=random_state,
    )

    # named these based on common features in fraud literature
    # not all of them are perfectly realistic but good enough for the pipeline demo
    feature_names = [
        "transaction_amount", "merchant_category", "hour_of_day",
        "day_of_week", "distance_from_home", "velocity_24h",
        "velocity_7d", "card_present", "online_transaction",
        "international", "num_transactions_24h", "avg_amount_7d",
        "std_amount_7d", "account_age_days", "credit_limit",
        "utilization_rate", "failed_attempts_24h", "device_change",
        "ip_change", "new_merchant",
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["is_fraud"] = y
    return df


def generate_drift_data(reference_df, n_samples=2000):
    # simulates distribution shift — transaction amounts and velocity
    # go up, which is a common pattern when fraud rings start operating
    # this is obviously simplified but enough to trigger evidently's drift detection
    np.random.seed(42)
    df = reference_df.drop("is_fraud", axis=1).sample(n_samples, replace=True, random_state=42).copy()
    df["transaction_amount"] += np.random.normal(5, 2, n_samples)
    df["velocity_24h"] += np.random.normal(2, 1, n_samples)
    df["is_fraud"] = reference_df["is_fraud"].sample(n_samples, replace=True, random_state=42).values
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_fraud_data()
    df.to_csv("data/transactions.csv", index=False)
    print(f"generated {len(df)} transactions — {df['is_fraud'].sum()} fraud cases ({df['is_fraud'].mean():.1%})")

    drift_df = generate_drift_data(df)
    drift_df.to_csv("data/transactions_drift.csv", index=False)
    print(f"generated {len(drift_df)} drifted transactions for monitoring tests")
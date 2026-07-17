import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from pathlib import Path

from dataset import load_and_split, fit_scaler, EnergyDataset, FEATURES
from model import TCN


def load_model(model_path='models/best_tcn.pt', n_features=9, device='cpu'):
    model = TCN(n_features=n_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def compute_reconstruction_errors(model, loader, device):
    errors, anomaly_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # Mean absolute error per sample
            err = (pred - y).abs().mean(dim=1).cpu().numpy()
            errors.append(err)
            # Get anomaly label for the forecast window
            # We use the label of the last step in the input window
            anomaly_labels.append(np.zeros(len(err)))  # placeholder
    return np.concatenate(errors), np.concatenate(anomaly_labels)


def detect_anomalies(threshold_sigma=3.0):
    device = torch.device('cpu')
    train_df, val_df, test_df = load_and_split()
    scaler = fit_scaler(train_df)

    model = load_model(device=device)

    # Compute errors on validation set to set threshold
    val_ds = EnergyDataset(val_df, scaler)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    val_errors, _ = compute_reconstruction_errors(model, val_loader, device)

    # Threshold: mean + N * std of validation errors
    threshold = val_errors.mean() + threshold_sigma * val_errors.std()
    print(f'Anomaly threshold: {threshold:.4f} (mean={val_errors.mean():.4f}, std={val_errors.std():.4f})')

    # Apply on test set
    test_ds = EnergyDataset(test_df, scaler)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    test_errors, _ = compute_reconstruction_errors(model, test_loader, device)

    # Flag anomalies
    predicted_anomalies = (test_errors > threshold).astype(int)
    n_flagged = predicted_anomalies.sum()
    pct = n_flagged / len(predicted_anomalies) * 100

    print(f'Test samples:      {len(test_errors):,}')
    print(f'Flagged anomalies: {n_flagged} ({pct:.1f}%)')
    print(f'Error range:       {test_errors.min():.4f} to {test_errors.max():.4f}')

    # Get actual anomaly labels from test set
    # Align with dataset windows
    test_anomaly_labels = test_df['is_anomaly'].values
    input_window = 168
    forecast_horizon = 24
    n_samples = len(test_df) - input_window - forecast_horizon + 1
    aligned_labels = test_anomaly_labels[input_window: input_window + n_samples]

    if len(aligned_labels) == len(predicted_anomalies):
        f1 = f1_score(aligned_labels, predicted_anomalies, zero_division=0)
        print(f'F1 Score vs labeled anomalies: {f1:.3f}')
        print(classification_report(aligned_labels, predicted_anomalies,
                                    target_names=['Normal', 'Anomaly'],
                                    zero_division=0))
    else:
        print(f'Label alignment mismatch: {len(aligned_labels)} vs {len(predicted_anomalies)}')

    results = {
        'threshold': round(float(threshold), 6),
        'n_flagged': int(n_flagged),
        'pct_flagged': round(float(pct), 2),
        'val_error_mean': round(float(val_errors.mean()), 6),
        'val_error_std': round(float(val_errors.std()), 6),
    }
    Path('data').mkdir(exist_ok=True)
    with open('data/anomaly_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved: data/anomaly_results.json')
    return results


if __name__ == '__main__':
    detect_anomalies()

import json
import numpy as np
import pandas as pd
import torch
import shap
from pathlib import Path

from dataset import load_and_split, fit_scaler, EnergyDataset, FEATURES
from model import TCN


def run_shap():
    device = torch.device('cpu')
    train_df, val_df, _ = load_and_split()
    scaler = fit_scaler(train_df)

    model = TCN(n_features=len(FEATURES))
    model.load_state_dict(torch.load('models/best_tcn.pt', map_location=device))
    model.eval()

    val_ds = EnergyDataset(val_df, scaler)

    # Use a small background set for SHAP
    n_background = 50
    n_explain    = 20

    background_x = torch.stack([val_ds[i][0] for i in range(n_background)])
    explain_x    = torch.stack([val_ds[i][0] for i in range(n_background, n_background + n_explain)])

    # Wrapper: TCN takes (batch, time, features) — SHAP needs 2D input
    # We reshape: flatten time*features -> SHAP explains feature importance
    def model_wrapper(x_flat):
        x_3d = torch.tensor(
            x_flat.reshape(-1, background_x.shape[1], background_x.shape[2]),
            dtype=torch.float32
        )
        with torch.no_grad():
            out = model(x_3d)
        # Return mean forecast as scalar per sample
        return out.mean(dim=1).numpy()

    bg_flat  = background_x.numpy().reshape(n_background, -1)
    exp_flat = explain_x.numpy().reshape(n_explain, -1)

    print('Running SHAP KernelExplainer (this takes ~2 minutes)...')
    explainer   = shap.KernelExplainer(model_wrapper, bg_flat)
    shap_values = explainer.shap_values(exp_flat, nsamples=100)

    # Reshape back to (n_explain, time, features) and average over time
    n_time     = background_x.shape[1]
    n_features = background_x.shape[2]
    shap_3d    = shap_values.reshape(n_explain, n_time, n_features)
    feature_importance = np.abs(shap_3d).mean(axis=(0, 1))

    print('\nFeature Importance (mean |SHAP|):')
    print('-' * 40)
    sorted_idx = np.argsort(feature_importance)[::-1]
    results = {}
    for rank, i in enumerate(sorted_idx):
        print(f'  {rank+1}. {FEATURES[i]:25s}: {feature_importance[i]:.5f}')
        results[FEATURES[i]] = round(float(feature_importance[i]), 6)

    Path('data').mkdir(exist_ok=True)
    with open('data/shap_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved: data/shap_results.json')
    return results


if __name__ == '__main__':
    run_shap()

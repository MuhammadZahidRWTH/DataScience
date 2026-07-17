import json
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import load_and_split, fit_scaler, EnergyDataset, FEATURES
from model import TCN


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    preds, targets = [], []
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    return total_loss / len(loader), preds, targets


def run_training(
    epochs=50,
    batch_size=64,
    lr=0.001,
    patience=10,
    num_channels=None,
    kernel_size=3,
    dropout=0.2,
):
    if num_channels is None:
        num_channels = [64, 128, 128, 64]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data
    train_df, val_df, test_df = load_and_split()
    scaler = fit_scaler(train_df)

    train_ds = EnergyDataset(train_df, scaler)
    val_ds   = EnergyDataset(val_df,   scaler)
    test_ds  = EnergyDataset(test_df,  scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}')

    # Model
    model = TCN(
        n_features=len(FEATURES),
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    mlflow.set_experiment('energy-price-forecasting')

    with mlflow.start_run():
        mlflow.log_params({
            'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
            'num_channels': str(num_channels), 'kernel_size': kernel_size,
            'dropout': dropout, 'total_params': total_params,
        })

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = 'models/best_tcn.pt'
        Path('models').mkdir(exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_targets = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            val_mae  = mae(val_targets, val_preds)
            val_rmse = rmse(val_targets, val_preds)

            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
            }, step=epoch)

            if epoch % 5 == 0 or epoch == 1:
                print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | '
                      f'Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} EUR/MWh')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # Test evaluation
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        _, test_preds, test_targets = eval_epoch(model, test_loader, criterion, device)

        # Inverse transform to real EUR/MWh
        target_idx = FEATURES.index('price_eur_mwh')
        scale = scaler.scale_[target_idx]
        mean  = scaler.mean_[target_idx]
        test_preds_real   = test_preds * scale + mean
        test_targets_real = test_targets * scale + mean

        test_mae  = mae(test_targets_real, test_preds_real)
        test_rmse = rmse(test_targets_real, test_preds_real)

        mlflow.log_metrics({'test_mae': test_mae, 'test_rmse': test_rmse})
        sample_input = torch.zeros(1, 168, len(FEATURES))
        mlflow.pytorch.log_model(model, 'tcn_model', serialization_format='pickle')

        print('\n' + '=' * 55)
        print('FINAL TEST RESULTS')
        print('=' * 55)
        print(f'  Test MAE:  {test_mae:.2f} EUR/MWh')
        print(f'  Test RMSE: {test_rmse:.2f} EUR/MWh')

        # Load baseline targets
        try:
            with open('data/baseline_results.json') as f:
                baselines = json.load(f)
            target = baselines.get('tcn_target_mae', 9.96)
            status = 'BEAT' if test_mae < target else 'MISSED'
            print(f'  Target MAE: {target:.2f} EUR/MWh  [{status}]')
        except FileNotFoundError:
            pass

        results = {
            'test_mae': round(test_mae, 4),
            'test_rmse': round(test_rmse, 4),
        }
        with open('data/tcn_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print('Results saved to data/tcn_results.json')
        print('=' * 55)

    return model, scaler


if __name__ == '__main__':
    run_training(epochs=50, batch_size=64, lr=0.0003, patience=10, dropout=0.3, num_channels=[32, 64, 64, 32])

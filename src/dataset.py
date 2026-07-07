import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


FEATURES = [
    'price_eur_mwh', 'load_mw', 'solar_mw',
    'wind_mw', 'gas_price_eur_mwh',
    'hour', 'day_of_week', 'month', 'is_weekend',
]
TARGET = 'price_eur_mwh'
INPUT_WINDOW = 168   # 7 days of hourly history
FORECAST_HORIZON = 24  # predict next 24 hours


def load_and_split(path='data/energy_prices.csv'):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    train = df[:'2022-12-31']
    val   = df['2023-01-01':'2023-06-30']
    test  = df['2023-07-01':]
    return train, val, test


def fit_scaler(train_df):
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])
    return scaler


def scale(df, scaler):
    scaled = df.copy()
    scaled[FEATURES] = scaler.transform(df[FEATURES])
    return scaled


class EnergyDataset(Dataset):
    def __init__(self, df, scaler, input_window=INPUT_WINDOW,
                 forecast_horizon=FORECAST_HORIZON):
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.target_idx = FEATURES.index(TARGET)

        scaled = df.copy()
        scaled[FEATURES] = scaler.transform(df[FEATURES])
        self.data = scaled[FEATURES].values.astype(np.float32)
        self.targets = scaled[TARGET].values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.input_window - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_window]          # (168, 9)
        y = self.targets[idx + self.input_window:
                         idx + self.input_window + self.forecast_horizon]  # (24,)
        return torch.tensor(x), torch.tensor(y)

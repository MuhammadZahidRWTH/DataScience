import numpy as np
import pandas as pd
from pathlib import Path


def generate_energy_dataset(output_path='data/energy_prices.csv', seed=42):
    np.random.seed(seed)
    hours = pd.date_range(start='2021-01-01', end='2023-12-31 23:00', freq='h')
    n = len(hours)
    t = np.arange(n)
    crisis_peak = 17520
    trend = 80.0 + 120 * np.exp(-((t - crisis_peak) ** 2) / (2 * 5000 ** 2))
    annual = 25 * np.sin(2 * np.pi * t / 8760 - np.pi / 2)
    weekly = -15 * (hours.dayofweek >= 5).astype(float)
    hod = hours.hour.values
    daily = (20 * np.sin(2 * np.pi * (hod - 6) / 24)
             + 10 * np.sin(4 * np.pi * (hod - 8) / 24))
    noise = np.random.normal(0, 12, n)
    spikes = np.zeros(n)
    spike_idx = np.random.choice(n, size=180, replace=False)
    spikes[spike_idx] = np.random.exponential(150, 180)
    neg_idx = np.where((hours.dayofweek == 6) & (hours.hour.isin([11,12,13,14])))[0]
    neg_sample = np.random.choice(neg_idx, size=min(80, len(neg_idx)), replace=False)
    spikes[neg_sample] = -np.random.uniform(20, 80, len(neg_sample))
    price = trend + annual + weekly + daily + noise + spikes
    load_mw = (45000 + 8000 * np.sin(2 * np.pi * t / 8760)
               + 5000 * np.sin(2 * np.pi * hod / 24)
               + np.random.normal(0, 1500, n))
    solar_mw = np.maximum(
        0, 20000 * np.sin(np.pi * (hod - 6) / 12) * np.random.uniform(0.4, 1.0, n)
    )
    wind_raw = np.cumsum(np.random.randn(n)) / 500
    wind_mw = ((wind_raw - wind_raw.min()) / (wind_raw.max() - wind_raw.min()) * 55000).clip(0, 55000)
    gas = 30 + 60 * np.exp(-((t - crisis_peak) ** 2) / (2 * 6000 ** 2)) + np.random.normal(0, 3, n)
    rm = pd.Series(price).rolling(168).mean().bfill()
    rs = pd.Series(price).rolling(168).std().bfill()
    is_anomaly = ((price > rm + 3 * rs) | (price < rm - 2 * rs)).astype(int)
    df = pd.DataFrame({
        'timestamp': hours,
        'price_eur_mwh': price,
        'load_mw': load_mw,
        'solar_mw': solar_mw,
        'wind_mw': wind_mw,
        'gas_price_eur_mwh': gas,
        'hour': hours.hour,
        'day_of_week': hours.dayofweek,
        'month': hours.month,
        'is_weekend': (hours.dayofweek >= 5).astype(int),
        'is_anomaly': is_anomaly,
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Saved: {output_path}  shape={df.shape}  anomalies={is_anomaly.sum()}')
    return df


if __name__ == '__main__':
    generate_energy_dataset()

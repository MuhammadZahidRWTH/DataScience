import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')


def load_data(path='data/energy_prices.csv'):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df.set_index('timestamp')


def evaluate(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'mae': round(mae, 4), 'rmse': round(rmse, 4)}


def run_baselines():
    print('=' * 55)
    print('BASELINE MODELS')
    print('=' * 55)
    df    = load_data()
    train = df[:'2022-12-31']
    val   = df['2023-01-01':'2023-06-30']

    # Persistence
    lag24  = df['price_eur_mwh'].shift(24).loc[val.index]
    mask   = ~lag24.isna()
    m1     = evaluate(val['price_eur_mwh'].values[mask], lag24.values[mask])
    print(f'Persistence (24h lag)  MAE: {m1["mae"]:.2f}  RMSE: {m1["rmse"]:.2f} EUR/MWh')

    # Holt-Winters
    daily_train = train['price_eur_mwh'].resample('D').mean()
    daily_val   = val['price_eur_mwh'].resample('D').mean()
    hw  = ExponentialSmoothing(daily_train, trend='add', seasonal='add',
                               seasonal_periods=7, damped_trend=True).fit(optimized=True)
    fc  = hw.forecast(len(daily_val))
    m2  = evaluate(daily_val.values, fc.values)
    print(f'Holt-Winters (daily)   MAE: {m2["mae"]:.2f}  RMSE: {m2["rmse"]:.2f} EUR/MWh')

    target = min(m1['mae'], m2['mae']) * 0.7
    print(f'\nTCN target MAE: < {target:.2f} EUR/MWh (30% improvement)')

    results = {'persistence': m1, 'holtwinters': m2, 'tcn_target_mae': round(target, 4)}
    Path('data').mkdir(exist_ok=True)
    with open('data/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved to data/baseline_results.json')
    print('=' * 55)
    return results


if __name__ == '__main__':
    run_baselines()

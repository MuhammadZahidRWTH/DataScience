import pandas as pd
import numpy as np


def load_data(path='data/energy_prices.csv'):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df.set_index('timestamp')


def split_data(df):
    train = df[:'2022-12-31']
    val   = df['2023-01-01':'2023-06-30']
    test  = df['2023-07-01':]
    return train, val, test


def print_summary(df, train, val, test):
    print('=' * 55)
    print('ENERGY PRICE DATASET - EDA SUMMARY')
    print('=' * 55)
    print(f'Shape:      {df.shape}')
    print(f'Date range: {df.index.min().date()} to {df.index.max().date()}')
    print(f'\nPrice stats (EUR/MWh):')
    print(df['price_eur_mwh'].describe().round(2))
    print(f'\nSplit sizes:')
    print(f'  Train: {len(train):,} hours')
    print(f'  Val:   {len(val):,} hours')
    print(f'  Test:  {len(test):,} hours')
    print(f'\nAnomalies: {df["is_anomaly"].sum()} ({df["is_anomaly"].mean()*100:.1f}%)')
    print(f'\nCorrelations with price:')
    feats = ['load_mw', 'solar_mw', 'wind_mw', 'gas_price_eur_mwh']
    for f, c in df[feats].corrwith(df['price_eur_mwh']).items():
        print(f'  {f:25s}: {c:+.3f}')
    print('=' * 55)


if __name__ == '__main__':
    df = load_data()
    train, val, test = split_data(df)
    print_summary(df, train, val, test)

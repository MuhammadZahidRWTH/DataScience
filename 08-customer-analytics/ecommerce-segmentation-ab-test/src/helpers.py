"""
Helper functions I use everywhere
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Get the project root directory
# IMPORTANT: This file is in src/, so project root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_seed(seed=42):
    """Make everything reproducible"""
    np.random.seed(seed)

def print_header(text):
    """Print a nice header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def get_project_root():
    """Get the absolute path to the project root"""
    return PROJECT_ROOT

def get_path(relative_path):
    """Get absolute path from project root"""
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

def save_csv(df, filename, folder='processed'):
    """Save dataframe to CSV with absolute path"""
    # Create the full path
    folder_path = os.path.join(PROJECT_ROOT, 'data', folder)
    os.makedirs(folder_path, exist_ok=True)
    
    full_path = os.path.join(folder_path, filename)
    df.to_csv(full_path, index=False)
    print(f" Saved: {full_path}")
    return full_path

def load_csv(filename, folder='processed'):
    """Load a CSV file"""
    full_path = os.path.join(PROJECT_ROOT, 'data', folder, filename)
    if os.path.exists(full_path):
        return pd.read_csv(full_path)
    else:
        print(f" File not found: {full_path}")
        return None

def get_figures_path():
    """Get the figures directory path"""
    path = os.path.join(PROJECT_ROOT, 'data', 'outputs', 'figures')
    os.makedirs(path, exist_ok=True)
    return path

def get_results_path():
    """Get the results directory path"""
    path = os.path.join(PROJECT_ROOT, 'data', 'outputs', 'results')
    os.makedirs(path, exist_ok=True)
    return path

def get_processed_path():
    """Get the processed data directory path"""
    path = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(path, exist_ok=True)
    return path

def get_raw_path():
    """Get the raw data directory path"""
    path = os.path.join(PROJECT_ROOT, 'data', 'raw')
    os.makedirs(path, exist_ok=True)
    return path
"""Preprocessing pipeline for NASA C-MAPSS dataset (FD001-FD004).

Handles data loading, RUL computation and clipping (at 125), engine-based
80/20 train/validation splitting, zero-variance column dropping, and
MinMaxScaler feature scaling fitted on training engines only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Column definitions for C-MAPSS data files
FEATURE_COLS = [
    'op_setting_1',
    'op_setting_2',
    'op_setting_3',
] + [f'sensor_{i}' for i in range(1, 22)]

ALL_COLS = ['engine_id', 'cycle'] + FEATURE_COLS


def load_cmapss_raw(
    dataset_name: str = 'FD001',
    data_dir: str | Path = 'data/raw/cmapps/CMAPSSData',
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load raw train, test, and RUL txt files for a given C-MAPSS subdataset.

    Never modifies files inside data/raw/.
    """
    data_path = Path(data_dir)
    train_file = data_path / f'train_{dataset_name}.txt'
    test_file = data_path / f'test_{dataset_name}.txt'
    rul_file = data_path / f'RUL_{dataset_name}.txt'

    # Read space-separated files, ignoring any trailing whitespace/empty columns
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, engine='python')
    test_df = pd.read_csv(test_file, sep=r'\s+', header=None, engine='python')
    rul_array = pd.read_csv(rul_file, sep=r'\s+', header=None, engine='python').values.flatten()

    # Drop columns that are entirely NaN (due to trailing spaces in some lines)
    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')

    # Assign column names if column count matches
    if train_df.shape[1] >= len(ALL_COLS):
        train_df = train_df.iloc[:, :len(ALL_COLS)]
        train_df.columns = ALL_COLS
    else:
        raise ValueError(f"Unexpected train column count: {train_df.shape[1]}")

    if test_df.shape[1] >= len(ALL_COLS):
        test_df = test_df.iloc[:, :len(ALL_COLS)]
        test_df.columns = ALL_COLS
    else:
        raise ValueError(f"Unexpected test column count: {test_df.shape[1]}")

    return train_df, test_df, rul_array


def compute_train_rul(train_df: pd.DataFrame, max_rul: float = 125.0) -> pd.DataFrame:
    """Compute RUL for training data and clip at max_rul (default 125)."""
    df = train_df.copy()
    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    df['rul'] = max_cycles - df['cycle']
    if max_rul is not None:
        df['rul'] = df['rul'].clip(upper=max_rul)
    return df


def compute_test_rul(test_df: pd.DataFrame, rul_array: np.ndarray, max_rul: float = 125.0) -> pd.DataFrame:
    """Compute RUL for test data using recorded max cycles plus ground-truth RUL."""
    df = test_df.copy()
    max_cycles = df.groupby('engine_id')['cycle'].max()

    # Map engine_id to its final true RUL
    # rul_array corresponds to test engines 1, 2, ..., N
    engine_ids = df['engine_id'].unique()
    if len(engine_ids) != len(rul_array):
        raise ValueError(f"Mismatch between number of test engines ({len(engine_ids)}) and RUL entries ({len(rul_array)})")

    rul_dict = {eng: rul_array[i] for i, eng in enumerate(sorted(engine_ids))}
    max_cycle_dict = max_cycles.to_dict()

    # RUL at cycle t = (max_cycle - cycle) + true_rul_at_end
    df['max_cycle_for_engine'] = df['engine_id'].map(max_cycle_dict)
    df['true_rul_end'] = df['engine_id'].map(rul_dict)
    df['rul'] = (df['max_cycle_for_engine'] - df['cycle']) + df['true_rul_end']

    if max_rul is not None:
        df['rul'] = df['rul'].clip(upper=max_rul)

    df = df.drop(columns=['max_cycle_for_engine', 'true_rul_end'])
    return df


def split_train_val_by_engine(
    train_df: pd.DataFrame,
    val_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training dataframe strictly by engine_id (80/20) to prevent data leakage."""
    engine_ids = train_df['engine_id'].unique()
    np.random.seed(random_state)
    shuffled_engines = np.random.permutation(engine_ids)

    split_idx = int(len(shuffled_engines) * (1 - val_split))
    train_engines = set(shuffled_engines[:split_idx])
    val_engines = set(shuffled_engines[split_idx:])

    train_split_df = train_df[train_df['engine_id'].isin(train_engines)].copy()
    val_split_df = train_df[train_df['engine_id'].isin(val_engines)].copy()

    return train_split_df, val_split_df


def drop_zero_variance_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop columns with zero variance across the training set."""
    train_features = train_df[feature_cols]
    var = train_features.var()
    zero_var_cols = var[var == 0].index.tolist()

    active_features = [col for col in feature_cols if col not in zero_var_cols]

    return train_df, val_df, test_df, active_features


def preprocess_cmapss(
    dataset_name: str = 'FD001',
    data_dir: str | Path = 'data/raw/cmapps/CMAPSSData',
    output_dir: str | Path = 'data/processed/maintenance',
    val_split: float = 0.2,
    random_state: int = 42,
    max_rul: float = 125.0,
) -> Dict[str, Any]:
    """Full preprocessing pipeline for C-MAPSS dataset."""
    # 1. Load raw data
    train_raw, test_raw, rul_array = load_cmapss_raw(dataset_name, data_dir)

    # 2. Compute RUL and clip at max_rul
    train_df = compute_train_rul(train_raw, max_rul=max_rul)
    test_df = compute_test_rul(test_raw, rul_array, max_rul=max_rul)

    # 3. Split train and validation strictly by engine_id
    train_split_df, val_split_df = split_train_val_by_engine(
        train_df, val_split=val_split, random_state=random_state
    )

    # 4. Identify and drop zero-variance columns based on training split
    _, _, _, active_features = drop_zero_variance_features(
        train_split_df, val_split_df, test_df, FEATURE_COLS
    )

    meta_cols = ['engine_id', 'cycle', 'rul']

    X_train = train_split_df[active_features].values
    y_train = train_split_df['rul'].values
    meta_train = train_split_df[meta_cols].values

    X_val = val_split_df[active_features].values
    y_val = val_split_df['rul'].values
    meta_val = val_split_df[meta_cols].values

    X_test = test_df[active_features].values
    y_test = test_df['rul'].values
    meta_test = test_df[meta_cols].values

    # 5. Fit MinMaxScaler on training features only
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 6. Save processed arrays and scaler
    out_path = Path(output_dir) / dataset_name
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / 'X_train.npy', X_train_scaled)
    np.save(out_path / 'y_train.npy', y_train)
    np.save(out_path / 'meta_train.npy', meta_train)
    np.save(out_path / 'X_val.npy', X_val_scaled)
    np.save(out_path / 'y_val.npy', y_val)
    np.save(out_path / 'meta_val.npy', meta_val)
    np.save(out_path / 'X_test.npy', X_test_scaled)
    np.save(out_path / 'y_test.npy', y_test)
    np.save(out_path / 'meta_test.npy', meta_test)
    joblib.dump(scaler, out_path / 'scaler.joblib')
    joblib.dump(active_features, out_path / 'active_features.joblib')

    result = {
        'dataset': dataset_name,
        'active_features': active_features,
        'X_train_shape': X_train_scaled.shape,
        'y_train_shape': y_train.shape,
        'X_val_shape': X_val_scaled.shape,
        'y_val_shape': y_val.shape,
        'X_test_shape': X_test_scaled.shape,
        'y_test_shape': y_test.shape,
    }
    return result


if __name__ == '__main__':
    # Run on FD001 only and print shapes
    res = preprocess_cmapss('FD001')
    print(f"Processed FD001 successfully:")
    for k, v in res.items():
        print(f"  {k}: {v}")

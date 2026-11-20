"""Training script for M1 maintenance models (Random Forest and XGBoost) on C-MAPSS dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.maintenance.evaluation import nasa_score, rmse


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_sequences(X, y, meta, window_size=30):
    """Create sequences grouped by engine_id to prevent leakage."""
    X_seq, y_seq = [], []
    if len(X) == 0 or len(meta) == 0:
        n_features = X.shape[1] if X.ndim > 1 else 1
        return np.empty((0, window_size, n_features)), np.empty((0,))
    engine_ids = meta[:, 0]
    for eng_id in np.unique(engine_ids):
        mask = (engine_ids == eng_id)
        X_eng = X[mask]
        y_eng = y[mask]
        if len(X_eng) == 0:
            continue
        if len(X_eng) >= window_size:
            for i in range(len(X_eng) - window_size + 1):
                X_seq.append(X_eng[i : i + window_size])
                y_seq.append(y_eng[i + window_size - 1])
        else:
            pad_len = window_size - len(X_eng)
            padded_X = np.pad(X_eng, ((pad_len, 0), (0, 0)), mode='edge')
            X_seq.append(padded_X)
            y_seq.append(y_eng[-1])
    if len(X_seq) == 0:
        n_features = X.shape[1] if X.ndim > 1 else 1
        return np.empty((0, window_size, n_features)), np.empty((0,))
    return np.array(X_seq), np.array(y_seq)


class M1SequentialModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, model_type='lstm'):
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()


def train_pytorch_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                preds = model(x_val)
                val_loss += criterion(preds, y_val).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_and_evaluate_m1(
    dataset_name: str = 'FD001',
    data_dir: str | Path = 'data/processed/maintenance',
    models_dir: str | Path = 'models/maintenance',
    metrics_dir: str | Path = 'outputs/metrics',
    random_state: int = 42,
    patience: int = 5,
) -> Dict[str, Any]:
    """Train Random Forest, XGBoost, LSTM, and GRU models and evaluate them."""
    data_path = Path(data_dir) / dataset_name
    models_path = Path(models_dir)
    metrics_path = Path(metrics_dir)

    models_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    meta_train = np.load(data_path / 'meta_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    y_val = np.load(data_path / 'y_val.npy')
    meta_val = np.load(data_path / 'meta_val.npy')
    X_test = np.load(data_path / 'X_test.npy')
    y_test = np.load(data_path / 'y_test.npy')
    meta_test = np.load(data_path / 'meta_test.npy')

    # 1. Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 2. Train XGBoost
    xgb_model = XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)

    # 3. Prepare sequences for LSTM/GRU
    window_size = 30
    X_train_s, y_train_s = create_sequences(X_train, y_train, meta_train, window_size)
    X_val_s, y_val_s = create_sequences(X_val, y_val, meta_val, window_size)
    X_test_s, y_test_s = create_sequences(X_test, y_test, meta_test, window_size)

    train_loader = DataLoader(SequenceDataset(X_train_s, y_train_s), batch_size=64, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val_s, y_val_s), batch_size=64)
    test_loader = DataLoader(SequenceDataset(X_test_s, y_test_s), batch_size=64)

    # 4. Train LSTM
    lstm_model = M1SequentialModel(input_dim=X_train.shape[1], model_type='lstm')
    lstm_model = train_pytorch_model(lstm_model, train_loader, val_loader, patience=patience)

    # 5. Train GRU
    gru_model = M1SequentialModel(input_dim=X_train.shape[1], model_type='gru')
    gru_model = train_pytorch_model(gru_model, train_loader, val_loader, patience=patience)

    # Evaluation helper
    def get_metrics(y_true, y_pred):
        return {
            'rmse': float(rmse(y_true, y_pred)),
            'nasa_score': float(nasa_score(y_true, y_pred))
        }

    def predict_pytorch(model, loader):
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x, _ in loader:
                p = model(x)
                if p.dim() == 0:
                    all_preds.append(p.unsqueeze(0).cpu().numpy())
                else:
                    all_preds.append(p.cpu().numpy())
        return np.concatenate(all_preds)

    # Evaluate models
    results = {
        'random_forest': {
            'val': get_metrics(y_val, rf_model.predict(X_val)),
            'test': get_metrics(y_test, rf_model.predict(X_test))
        },
        'xgboost': {
            'val': get_metrics(y_val, xgb_model.predict(X_val)),
            'test': get_metrics(y_test, xgb_model.predict(X_test))
        },
        'lstm': {
            'val': get_metrics(y_val_s, predict_pytorch(lstm_model, val_loader)),
            'test': get_metrics(y_test_s, predict_pytorch(lstm_model, test_loader))
        },
        'gru': {
            'val': get_metrics(y_val_s, predict_pytorch(gru_model, val_loader)),
            'test': get_metrics(y_test_s, predict_pytorch(gru_model, test_loader))
        }
    }

    # Save models
    joblib.dump(rf_model, models_path / f'rf_{dataset_name.lower()}.joblib')
    joblib.dump(xgb_model, models_path / f'xgb_{dataset_name.lower()}.joblib')
    torch.save(lstm_model, models_path / f'lstm_{dataset_name.lower()}.pt')
    torch.save(gru_model, models_path / f'gru_{dataset_name.lower()}.pt')

    # Save best model overall for this subset
    val_rmses = {name: m['val']['rmse'] for name, m in results.items()}
    best_name = min(val_rmses, key=val_rmses.get)

    if best_name == 'random_forest':
        joblib.dump(rf_model, models_path / f'best_{dataset_name.lower()}.joblib')
    elif best_name == 'xgboost':
        joblib.dump(xgb_model, models_path / f'best_{dataset_name.lower()}.joblib')
    elif best_name == 'lstm':
        torch.save(lstm_model, models_path / f'best_{dataset_name.lower()}.pt')
    else:
        torch.save(gru_model, models_path / f'best_{dataset_name.lower()}.pt')

    # Update global metrics file
    metrics_file = metrics_path / 'm1_metrics.json'
    all_data = []
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                content = json.load(f)
                # Handle both list and dict formats if needed, but assuming dict with 'dataset' or list
                if isinstance(content, list):
                    all_data = content
                elif isinstance(content, dict):
                    all_data = [content]
        except json.JSONDecodeError:
            all_data = []

    # Remove previous entry for this dataset if it exists
    all_data = [d for d in all_data if d.get('dataset') != dataset_name]

    # Reformat metrics for saving
    summary = {
        'dataset': dataset_name,
        'models': {
            name: {
                'val_rmse': m['val']['rmse'],
                'val_nasa_score': m['val']['nasa_score'],
                'test_rmse': m['test']['rmse'],
                'test_nasa_score': m['test']['nasa_score']
            } for name, m in results.items()
        },
        'best_model': best_name
    }
    all_data.append(summary)

    with open(metrics_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    return summary


if __name__ == '__main__':
    from src.maintenance.preprocessing import preprocess_cmapss

    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    all_res = []

    # Load existing metrics to check what's already there
    metrics_file = Path('outputs/metrics/m1_metrics.json')
    existing_datasets = []
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    existing_datasets = [d['dataset'] for d in data]
                elif isinstance(data, dict):
                    existing_datasets = [data['dataset']]
            except:
                pass

    for ds in datasets:
        print(f"\n{'='*20} Processing {ds} {'='*20}")
        # Only preprocess/train if not FD001 or if user asked to keep FD001 (implied we run others)
        if ds != 'FD001' or ds not in existing_datasets:
            preprocess_cmapss(ds)
            res = train_and_evaluate_m1(ds)
        else:
            # For table printing, find existing result
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    res = next(d for d in data if d['dataset'] == ds)
                else:
                    res = data
        all_res.append(res)

    # Final comparison table
    print(f"\n{'Dataset':<10} | {'Model':<20} | {'Val RMSE':<10} | {'Val Score':<12} | {'Test RMSE':<10} | {'Test Score':<12}")
    print("-" * 88)
    for res in sorted(all_res, key=lambda x: x['dataset']):
        ds_name = res['dataset']
        for model_name, m in res['models'].items():
            print(f"{ds_name:<10} | {model_name.replace('_', ' ').title():<20} | {m['val_rmse']:<10.4f} | {m['val_nasa_score']:<12.2f} | {m['test_rmse']:<10.4f} | {m['test_nasa_score']:<12.2f}")
        print("-" * 88)


"""Tests for M1 maintenance preprocessing pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.maintenance.evaluation import nasa_score, rmse
from src.maintenance.preprocessing import (
    compute_test_rul,
    compute_train_rul,
    load_cmapss_raw,
    preprocess_cmapss,
    split_train_val_by_engine,
)
from src.maintenance.training import train_and_evaluate_m1


def test_load_cmapss_raw():
    """Test loading FD001 raw data files."""
    train_df, test_df, rul_array = load_cmapss_raw('FD001')
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(rul_array, np.ndarray)
    assert 'engine_id' in train_df.columns
    assert 'cycle' in train_df.columns
    assert len(rul_array) == test_df['engine_id'].nunique()


def test_compute_train_rul():
    """Test train RUL computation and clipping at 125."""
    data = {
        'engine_id': [1, 1, 1, 2, 2],
        'cycle': [1, 2, 3, 1, 2],
    }
    df = pd.DataFrame(data)
    result = compute_train_rul(df, max_rul=125.0)
    # Engine 1: max cycle 3 -> RULs: 3-1=2, 3-2=1, 3-3=0
    # Engine 2: max cycle 2 -> RULs: 2-1=1, 2-2=0
    assert list(result['rul']) == [2, 1, 0, 1, 0]


def test_split_train_val_by_engine():
    """Test 80/20 engine-based split ensures no engine overlap."""
    data = {
        'engine_id': list(range(1, 101)),  # 100 engines
        'cycle': [1] * 100,
        'val': [1] * 100,
    }
    df = pd.DataFrame(data)
    train_split, val_split = split_train_val_by_engine(df, val_split=0.2, random_state=42)

    train_engines = set(train_split['engine_id'])
    val_engines = set(val_split['engine_id'])

    # Check no overlap
    assert train_engines.isdisjoint(val_engines)
    # Check total engines
    assert len(train_engines) + len(val_engines) == 100
    # Check approximately 80/20 split
    assert len(train_engines) == 80
    assert len(val_engines) == 20


def test_preprocess_cmapss_fd001():
    """Test full preprocessing pipeline execution on FD001."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = preprocess_cmapss(
            dataset_name='FD001',
            output_dir=tmpdir,
            val_split=0.2,
            random_state=42,
        )
        assert res['dataset'] == 'FD001'
        assert res['X_train_shape'][0] > 0
        assert res['X_val_shape'][0] > 0
        assert res['X_test_shape'][0] > 0

        # Check files saved
        out_path = Path(tmpdir) / 'FD001'
        assert (out_path / 'X_train.npy').exists()
        assert (out_path / 'y_train.npy').exists()
        assert (out_path / 'scaler.joblib').exists()


def test_rmse():
    """Test RMSE calculation."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 22.0, 27.0])
    # squared errors: 0, 4, 9 -> mean = 13/3 -> sqrt = sqrt(13/3) ~ 2.081666
    val = rmse(y_true, y_pred)
    assert np.isclose(val, np.sqrt(13 / 3))


def test_nasa_score():
    """Test NASA asymmetric scoring function."""
    # Perfect predictions
    y_true = np.array([50.0, 50.0])
    y_pred = np.array([50.0, 50.0])
    assert nasa_score(y_true, y_pred) == 0.0

    # Under-estimation (pred < true) vs over-estimation (pred > true)
    # Over-estimation is penalized much more heavily.
    y_true_under = np.array([50.0])
    y_pred_under = np.array([40.0])  # d = -10 (under) -> exp(10/13) - 1
    score_under = nasa_score(y_true_under, y_pred_under)

    y_true_over = np.array([50.0])
    y_pred_over = np.array([60.0])  # d = +10 (over) -> exp(10/10) - 1 = exp(1) - 1
    score_over = nasa_score(y_true_over, y_pred_over)

    assert score_over > score_under


def test_plot_actual_vs_predicted():
    """Test actual vs predicted plot generation."""
    from src.maintenance.evaluation import plot_actual_vs_predicted, save_plot

    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 32.0, 38.0, 52.0])
    fig = plot_actual_vs_predicted(y_true, y_pred, 'FD001', 'Test Model')
    assert fig is not None
    assert hasattr(fig, 'savefig')
    save_plot(fig, 'test_plot.png')


def test_plot_model_comparison():
    """Test model comparison plot generation."""
    from src.maintenance.evaluation import plot_model_comparison

    # Sample metrics data
    metrics_data = [
        {
            'dataset': 'FD001',
            'models': {
                'random_forest': {'test_rmse': 17.36},
                'xgboost': {'test_rmse': 16.99},
                'lstm': {'test_rmse': 38.45},
                'gru': {'test_rmse': 13.88}
            }
        },
        {
            'dataset': 'FD002',
            'models': {
                'random_forest': {'test_rmse': 19.24},
                'xgboost': {'test_rmse': 18.77},
                'lstm': {'test_rmse': 38.27},
                'gru': {'test_rmse': 38.53}
            }
        }
    ]

    fig = plot_model_comparison(metrics_data)
    assert fig is not None


def test_prediction_output():
    """Test prediction output structure and validation."""
    from src.common.schemas import ModuleOutput, validate_output, STATUS_SUCCESS

    # Test creating a valid ModuleOutput
    output = ModuleOutput(
        module='test',
        prediction=10.0,
        risk_score=0.5,
        confidence=0.8,
        status=STATUS_SUCCESS,
        model_version='test_v1',
        explanation=['Prediction made', 'Risk assessed']
    )

    # Validate the output
    validated = validate_output(output)
    assert isinstance(validated, ModuleOutput)
    assert validated.prediction == 10.0
    assert validated.risk_score == 0.5
    assert validated.confidence == 0.8
    assert validated.status == STATUS_SUCCESS

def test_train_and_evaluate_m1():
    """Test training and evaluation pipeline with temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First preprocess to generate dummy/small processed dataset
        processed_dir = Path(tmpdir) / 'processed'
        res_prep = preprocess_cmapss(
            dataset_name='FD001',
            output_dir=processed_dir,
            val_split=0.2,
            random_state=42,
        )

        models_dir = Path(tmpdir) / 'models'
        metrics_dir = Path(tmpdir) / 'metrics'

        metrics = train_and_evaluate_m1(
            dataset_name='FD001',
            data_dir=processed_dir,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            random_state=42,
        )

        assert metrics['dataset'] == 'FD001'
        assert 'random_forest' in metrics['models']
        assert 'xgboost' in metrics['models']
        assert metrics['models']['random_forest']['val_rmse'] > 0
        assert metrics['models']['xgboost']['val_rmse'] > 0

        assert (models_dir / 'rf_fd001.joblib').exists()
        assert (models_dir / 'xgb_fd001.joblib').exists()
        assert (metrics_dir / 'm1_metrics.json').exists()


def test_predict_rul():
    """Test predict_rul prediction function and independent risk vs confidence calculation."""
    from src.maintenance.prediction import predict_rul, load_best_model_and_metadata
    from src.common.schemas import STATUS_SUCCESS, ModuleOutput

    # Dynamically load model to get expected feature count
    _, _, active_features = load_best_model_and_metadata('FD001')
    test_features = np.random.randn(1, len(active_features))
    test_meta = {'engine_id': 'ENG_001', 'cycle': 50}

    output = predict_rul('FD001', test_features, test_meta)

    assert isinstance(output, ModuleOutput)
    assert output.status == STATUS_SUCCESS
    assert isinstance(output.prediction, float)
    assert 0.0 <= output.risk_score <= 1.0
    assert 0.0 <= output.confidence <= 1.0
    assert len(output.explanation) > 0


def test_predict_rul_for_engine_and_batch():
    """Test engine level and batch prediction helpers."""
    from src.maintenance.prediction import predict_rul_for_engine, predict_rul_for_batch
    from src.common.schemas import STATUS_SUCCESS

    single_output = predict_rul_for_engine('FD001', engine_id='1', cycle=100)
    assert single_output.status == STATUS_SUCCESS
    assert single_output.module == 'prediction'

    batch_output = predict_rul_for_batch('FD001', engine_id='1', cycles=[10, 20, 30])
    assert len(batch_output) == 3
    for res in batch_output:
        assert res.status == STATUS_SUCCESS



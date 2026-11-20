import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the official NASA C-MAPSS asymmetric scoring function."""
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(s)

def save_plot(fig, filename: str, plots_dir: str = 'outputs/plots'):
    """Save a matplotlib figure to the specified directory."""
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(plots_dir) / filename)
    plt.close(fig)

def plot_actual_vs_predicted(y_true, y_pred, dataset_name, model_name):
    """Plot actual vs predicted RUL."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'{dataset_name} - {model_name} Actual vs Predicted RUL')
    return fig

def plot_engine_degradation(engine_data, engine_id, dataset_name):
    """Plot engine degradation curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(engine_data['cycle'], engine_data['rul'], label='Actual RUL')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('RUL')
    ax.set_title(f'{dataset_name} - Engine {engine_id} Degradation')
    ax.legend()
    return fig

def plot_model_comparison(metrics_data, plots_dir='outputs/plots'):
    """Plot multi-subset 4-model RMSE comparison chart."""
    datasets = sorted([d['dataset'] for d in metrics_data])
    models = ['random_forest', 'xgboost', 'lstm', 'gru']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.2

    for i, model in enumerate(models):
        rmse_vals = []
        for ds in datasets:
            ds_data = next(d for d in metrics_data if d['dataset'] == ds)
            rmse_vals.append(ds_data['models'].get(model, {}).get('test_rmse', 0))

        ax.bar(x + i*width, rmse_vals, width, label=model.replace('_', ' ').title())

    ax.set_ylabel('Test RMSE')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_title('Model Performance Comparison (Test RMSE)')
    save_plot(fig, 'model_comparison.png', plots_dir)
    return fig

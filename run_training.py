import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.maintenance.training import train_and_evaluate_m1
from src.maintenance.preprocessing import preprocess_cmapss
import json

datasets = ['FD002', 'FD003', 'FD004']
all_res = []

# Load existing metrics
metrics_file = Path('outputs/metrics/m1_metrics.json')
all_data = []
if metrics_file.exists():
    with open(metrics_file, 'r') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                all_data = data
            elif isinstance(data, dict):
                all_data = [data]
        except:
            pass

for ds in datasets:
    print(f"\n{'='*20} Processing {ds} {'='*20}")
    preprocess_cmapss(ds)
    res = train_and_evaluate_m1(ds)
    all_data = [d for d in all_data if d.get('dataset') != ds]
    all_data.append(res)

with open(metrics_file, 'w') as f:
    json.dump(all_data, f, indent=2)

# Final comparison table
print(f"\n{'Dataset':<10} | {'Model':<20} | {'Val RMSE':<10} | {'Val Score':<12} | {'Test RMSE':<10} | {'Test Score':<12}")
print("-" * 88)
for res in sorted(all_data, key=lambda x: x['dataset']):
    ds_name = res['dataset']
    for model_name, m in res['models'].items():
        print(f"{ds_name:<10} | {model_name.replace('_', ' ').title():<20} | {m['val_rmse']:<10.4f} | {m['val_nasa_score']:<12.2f} | {m['test_rmse']:<10.4f} | {m['test_nasa_score']:<12.2f}")
    print("-" * 88)

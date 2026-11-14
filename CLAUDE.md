# AeroNexus Development Guidelines

1. **Project Identity**: AeroNexus — an academic, non-operational multi-model ML framework with human-supervised decision support.
   - **Role**: Farhan builds ONLY M1 (`src/maintenance/`), M4 (`src/simulation/`), Fusion (`src/fusion/`), and Dashboard (`dashboard/`). Never inspect or modify M2 (`trajectory/opensky`) or M3 (`recognition/mar20`).
2. **Modules & Models**:
   - **M1 (`src/maintenance/`)**: Engine RUL prediction on NASA C-MAPSS (Random Forest, XGBoost, LSTM, GRU).
   - **M2 (`src/trajectory/`)**: Flight trajectory & anomaly detection on OpenSky ADS-B (Bi-LSTM, Isolation Forest).
   - **M3 (`src/recognition/`)**: Non-sensitive aircraft image recognition on MAR20 (YOLO).
   - **M4 (`src/simulation/`)**: Synthetic resource-readiness scoring (XGBoost + SHAP).
   - **Fusion (`src/fusion/`)**: Weighted risk aggregation with missing-module handling and SHAP explainability.
   - **Dashboard (`dashboard/`)**: 6-page Streamlit app (Overview, Maintenance, Trajectory, Recognition, Scenario Analysis, Explainability).
3. **Common Output Contract**: Every module must return JSON with keys: `module`, `prediction`, `risk_score`, `confidence`, `status`, `model_version`, `explanation`. Keep `risk_score` and `confidence` strictly separate.
4. **Strict Rules**: 
   - Never modify files inside `data/raw/`.
   - Never hardcode or fabricate metrics.
   - Prevent data leakage (split C-MAPSS by `engine_id`, not random rows).
   - Write `pytest` tests for every module.


# AeroNexus

AeroNexus is an integrated multi-model machine learning framework for air operations intelligence and decision support.

## Project Modules

1. Predictive Maintenance (M1)
2. Trajectory Prediction and Anomaly Research (M2)
3. Aircraft Image Recognition (M3)
4. Synthetic Air-Resource Readiness Simulation (M4)

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- PyTorch
- Streamlit
- Plotly

## Project Structure

- data/: Dataset storage
- notebooks/: ML experiments
- src/: Main source code
- src/common/: Common utilities and contracts
  - schemas.py: Standardized ModuleOutput contract
- models/: Trained models
- dashboard/: User interface
- tests/: Testing
  - test_common.py: ModuleOutput contract tests
- docs/: Documentation

## Project Rules

- **Modules M1 (src/maintenance/), M2 (src/trajectory/), M3 (src/recognition/), M4 (src/simulation/), Fusion (src/fusion/), Dashboard (dashboard/):** Only Farhan builds and maintains these
- **Never modify files inside data/raw/**
- **Every module must return ModuleOutput with keys: module, prediction, risk_score, confidence, status, model_version, explanation**
- **Keep risk_score and confidence strictly separate**
- **Write pytest tests for every module**
- **Prevent data leakage (split C-MAPSS by engine_id, not random rows)**

## Standardized Output Contract

All modules (M1-M4, Fusion) must return a :class:`ModuleOutput` so callers can rely on a single, stable shape regardless of which model produced the result.

### Key Concepts

- ``risk_score``: Module's assessment of how risky the input is (0.0 to 1.0)
- ``confidence``: Module's own certainty about that assessment (0.0 to 1.0)
- ``status``: Lifecycle status (success | unavailable | error)

### Output Format

Every module output includes:
- ``module``: Name of the module
- ``prediction``: Module-specific prediction payload
- ``risk_score``: Module risk assessment in [0.0, 1.0]
- ``confidence``: Module certainty about its prediction in [0.0, 1.0]
- ``status``: Lifecycle status
- ``model_version``: Version identifier
- ``explanation``: Human-readable explanation strings (list)

### Available Helper Methods

The ``ModuleOutput`` class provides:
- ``to_dict()``: Serialize to a plain, JSON-safe dict
- ``unavailable()``: Build response for modules that could not run
- ``error()``: Build response for modules that failed at runtime

### Utility Functions

- ``validate_output()``: Coerce dict or ModuleOutput into a validated ModuleOutput
- ``serialize_output()``: Validate then serialize a module output to a JSON-safe dict

## Usage Example

```python
from src.common.schemas import ModuleOutput, validate_output

output = ModuleOutput(
    module="maintenance",
    prediction={"rul": 120},
    risk_score=0.7,
    confidence=0.85,
    status="success",
    model_version="rf-v1.0",
    explanation=["high degradation trend"],
)

validated = validate_output(output)
serialized = output.to_dict()
```

## Team

- Muhammad Farhan Khalid
- Laiyba Tariq
- Ayesha Tanveer

## Project Scope

Academic proof of concept using open and synthetic datasets. Results are advisory and require human review.

## Project Modules

1. Predictive Maintenance
2. Trajectory Prediction and Anomaly Research
3. Aircraft Image Recognition
4. Synthetic Air-Resource Readiness Simulation

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- PyTorch
- Streamlit
- Plotly

## Project Structure

- data/: Dataset storage
- notebooks/: ML experiments
- src/: Main source code
- models/: Trained models
- dashboard/: User interface
- tests/: Testing
- docs/: Documentation

## Team

- Muhammad Farhan Khalid
- Laiyba Tariq
- Ayesha Tanveer

## Project Scope

Academic proof of concept using open and synthetic datasets. Results are advisory and require human review.

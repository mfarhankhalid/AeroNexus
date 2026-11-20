"""Project path configuration for AeroNexus.

Defines standard paths for data, models, and outputs.
"""

from pathlib import Path

# Base directory of the project (D:/AeroNexus)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Core directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

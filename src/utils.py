"""Simple path constants for the AeroNexus project.

This module holds the main directory paths used across the project.
Each constant is a ``pathlib.Path`` so you can join paths with the ``/``
operator, for example ``MODELS_DIR / 'rf_fd001.joblib'``.
"""

from pathlib import Path

# Root directory of the project (the folder that contains this file's parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where raw datasets live. We never write to this directory.
DATA_DIR = PROJECT_ROOT / 'data'

# Where trained models are saved and loaded from.
MODELS_DIR = PROJECT_ROOT / 'models'

# Where outputs like metrics and plots are written.
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
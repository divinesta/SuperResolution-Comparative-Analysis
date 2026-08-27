"""Project paths shared by local and Google Colab runs."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"
COLAB_DATA_ROOT = Path("/content/drive/MyDrive/FYP_SR_Data")
DATA_ROOT_ENVIRONMENT_VARIABLE = "FYP_SR_DATA_ROOT"


def resolve_data_root(explicit_path: str | Path | None = None) -> Path:
    """Choose the dataset root from an argument, environment, Colab, or local default."""
    if explicit_path is not None:
        return Path(explicit_path).expanduser()

    environment_path = os.environ.get(DATA_ROOT_ENVIRONMENT_VARIABLE)
    if environment_path:
        return Path(environment_path).expanduser()

    if COLAB_DATA_ROOT.is_dir():
        return COLAB_DATA_ROOT

    return LOCAL_DATA_ROOT


def dataset_hr_directory(dataset: str, data_root: str | Path | None = None) -> Path:
    """Return the expected HR directory for one benchmark dataset."""
    dataset_name = dataset.strip()
    if not dataset_name:
        raise ValueError("Dataset name cannot be empty.")

    root = resolve_data_root(data_root)
    return root / dataset_name / f"{dataset_name}_HR"

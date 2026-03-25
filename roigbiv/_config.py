"""ROI G. Biv — Configuration loader."""
from pathlib import Path
import yaml


def load_config(config_path=None) -> dict:
    """Load pipeline YAML config. Returns empty dict if path is None or not found."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}

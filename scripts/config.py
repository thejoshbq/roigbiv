"""ROI G. Biv — Shared configuration and project paths."""
from pathlib import Path
import yaml

BASE_DIR = Path.home() / 'Otis-Lab' / 'Projects' / 'roigbiv'


def load_config(config_path=None):
    default = BASE_DIR / 'configs' / 'pipeline.yaml'
    path = Path(config_path) if config_path else default
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}

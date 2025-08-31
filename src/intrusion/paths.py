from __future__ import annotations
from pathlib import Path


def get_project_root(marker: str = ".git") -> Path:
    path = Path().resolve()
    for parent in [path, *path.parents]:
        if (parent / marker).exists():
            return parent
    return path


ROOT = get_project_root()
DATA_PATH = ROOT / "data" / "cybersecurity_intrusion_data.csv"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

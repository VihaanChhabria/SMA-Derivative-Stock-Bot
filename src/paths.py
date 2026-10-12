"""Repository-relative paths for generated outputs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def output_path(category: str, filename: str) -> Path:
    """Return an output path and create its destination directory."""
    destination = PROJECT_ROOT / "outputs" / category
    destination.mkdir(parents=True, exist_ok=True)
    return destination / filename

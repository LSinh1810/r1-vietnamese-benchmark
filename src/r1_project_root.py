"""Resolve the project root after a notebook has located this bootstrap module."""

from pathlib import Path


def find_project_root():
    """Return the root of the checkout containing this module after validation."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "src" / "dataset.py").is_file():
        raise FileNotFoundError(f"Không tìm thấy src/dataset.py trong repository {root}.")
    return root

"""Locate the sample dataset bundled with the :mod:`blape` package.

Earlier versions downloaded the full dataset from Zenodo. The package now ships a
small, label-balanced sample subset under ``blape/sample_data`` so the demo and
tests run without any network access or external download.

:func:`download_data` is kept as a thin, backward-compatible shim that simply
points to (or copies) the bundled sample data; no network download is performed.
"""

import shutil
from pathlib import Path

# Sample data bundled inside the package: raw/ and baseline_removed/ subfolders.
SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "sample_data"


def get_sample_data_dir() -> Path:
    """Return the path to the sample dataset bundled with the package."""
    if not SAMPLE_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Bundled sample data not found at {SAMPLE_DATA_DIR}. "
            "Reinstall the package, or regenerate it from the full dataset with "
            "scripts/make_sample_data.py."
        )
    return SAMPLE_DATA_DIR


def download_data(path=None, raw=True, baseline_removed=True):
    """Deprecated: the sample dataset now ships with the package.

    No data is downloaded. If ``path`` is given, the bundled sample data is copied
    there so that existing ``download_data(path=...); read_data(path=...)``
    workflows keep working; otherwise the bundled path is returned directly.

    Args:
        path (str | None): Destination to copy the bundled sample data into. If
            ``None`` (default), nothing is copied and the bundled path is returned.
        raw (bool): Copy the ``raw`` split. Defaults to True.
        baseline_removed (bool): Copy the ``baseline_removed`` split. Defaults to True.

    Returns:
        str: Path to the directory that contains the ``raw`` / ``baseline_removed``
        sample data.
    """
    src = get_sample_data_dir()
    if path is None:
        return str(src)

    dst = Path(path)
    dst.mkdir(parents=True, exist_ok=True)

    splits = []
    if raw:
        splits.append("raw")
    if baseline_removed:
        splits.append("baseline_removed")

    for split in splits:
        s = src / split
        d = dst / split
        if not s.exists():
            continue
        if d.exists():
            print(f"[=] {d} already exists → skip")
            continue
        shutil.copytree(s, d)
        print(f"[+] Copied bundled {split} sample data → {d}")

    print("All requested sample data ready (no download needed).")
    return str(dst.resolve())


if __name__ == "__main__":
    print(get_sample_data_dir())

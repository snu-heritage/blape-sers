"""Generate the bundled sample dataset shipped inside the ``blape`` package.

This script carves a small, label-balanced subset out of the full SERS dataset
(``data/raw`` and ``data/baseline_removed``, downloaded separately) and writes it
to ``blape/sample_data`` at reduced numeric precision so that the demo and tests
can run without downloading the full dataset.

It is kept in the repository purely to document how the sample data was produced;
running it requires a local copy of the full dataset, which is *not* part of this
repository.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd

# ---- configuration ---------------------------------------------------------
SRC = Path("data")                       # full dataset (not committed)
DST = Path("blape") / "sample_data"      # bundled subset (committed)
SPLITS = ["raw", "baseline_removed"]
PER_BASE = 4                             # number of sample codes kept per base material
MAX_SPECTRA = 6                          # number of spectra (columns) kept per file
FLOAT_FORMAT = "%.5g"                    # reduced precision to keep files small


def parse(code):
    base, dye, mordant, aging = code.split("-")
    return base, dye, mordant, aging


def select_codes():
    """Greedily pick a label-balanced subset of sample codes."""
    raw_dir = SRC / "raw"
    codes = sorted(p.stem for p in raw_dir.glob("*.csv"))

    by_base = defaultdict(list)
    for c in codes:
        by_base[parse(c)[0]].append(c)

    covered = {"dye": set(), "mordant": set(), "aging": set()}
    counts = {"dye": defaultdict(int), "aging": defaultdict(int)}
    selected = []

    for base in sorted(by_base):
        pool = by_base[base][:]
        for _ in range(min(PER_BASE, len(pool))):
            def key(c):
                _, d, m, a = parse(c)
                # maximise newly-covered labels first, then balance dye/aging counts
                new_cov = (
                    (d not in covered["dye"])
                    + (m not in covered["mordant"])
                    + (a not in covered["aging"])
                )
                return (-new_cov, counts["aging"][a], counts["dye"][d], c)

            pool.sort(key=key)
            best = pool.pop(0)
            _, d, m, a = parse(best)
            covered["dye"].add(d)
            covered["mordant"].add(m)
            covered["aging"].add(a)
            counts["aging"][a] += 1
            counts["dye"][d] += 1
            selected.append(best)

    return sorted(selected)


def main():
    selected = select_codes()

    for split in SPLITS:
        (DST / split).mkdir(parents=True, exist_ok=True)

    n_spectra = 0
    for code in selected:
        for split in SPLITS:
            src = SRC / split / f"{code}.csv"
            if not src.exists():
                print(f"  ! missing {src} (skipped)")
                continue
            df = pd.read_csv(src)
            # keep the wavenumber column plus up to MAX_SPECTRA spectra columns
            keep = [df.columns[0]] + list(df.columns[1 : 1 + MAX_SPECTRA])
            df[keep].to_csv(DST / split / f"{code}.csv", index=False,
                            float_format=FLOAT_FORMAT)
            if split == "raw":
                n_spectra += len(keep) - 1

    print(f"Wrote {len(selected)} codes ({n_spectra} raw spectra) to {DST}/")
    print("codes:", selected)


if __name__ == "__main__":
    main()

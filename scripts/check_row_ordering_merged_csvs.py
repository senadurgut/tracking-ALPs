#!/usr/bin/env python3
"""
Check row ordering (ALP vs photons) in merged MadGraph CSV files.

For each merged file (e.g. data/cmsrun3-csvs-merged/merged_ma_0p7000GeV.csv),
we sample N events and examine rows 4–6 (default expected ordering):

  ROW_ALP (default 4) should:
    - have the highest energy among rows 4–6 (typically true)
    - satisfy the mass-shell relation  E^2 - |p|^2 ~= m_a^2
      (with m_a read from the filename)

If ordering appears swapped, the script suggests which row index (4/5/6)
behaves most like the ALP for that file. Photon ordering between g1/g2 is not
validated here.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


MASS_RE = re.compile(r"merged_ma_(\d+p\d+)GeV\.csv$")


def mass_key_to_float_geV(key: str) -> float:
    # e.g. "0p7000" -> 0.7000
    return float(key.replace("p", "."))


def inv_mass_sq(row: np.ndarray) -> float:
    # row: [E, px, py, pz]
    E, px, py, pz = row
    return float(E * E - (px * px + py * py + pz * pz))


@dataclass
class FileCheckResult:
    path: str
    ma: float
    n_events: int
    alp_row_suggested: int  # 4/5/6 (index within event block)
    frac_alp_is_maxE: float
    frac_alp_mass_ok: float


def iter_merged_csvs(csv_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(csv_dir)):
        if name.endswith(".csv") and name.startswith("merged_ma_"):
            yield os.path.join(csv_dir, name)


def extract_ma_from_filename(path: str) -> Optional[float]:
    m = MASS_RE.search(os.path.basename(path))
    if not m:
        return None
    return mass_key_to_float_geV(m.group(1))


def load_rows_for_events(csv_path: str, n_events: int, rows_per_event: int = 7) -> np.ndarray:
    """
    Return shape: (n_events, rows_per_event, 4)
    """
    nrows = rows_per_event * n_events
    temp = pd.read_csv(csv_path, sep=";", header=None, nrows=nrows)
    n_available_events = len(temp) // rows_per_event
    if n_available_events == 0:
        raise ValueError(f"no complete events found in {csv_path}")
    n_use = min(n_events, n_available_events)

    rows = np.empty((n_use, rows_per_event, 4), dtype=float)
    for i in range(n_use):
        for j in range(rows_per_event):
            s = temp.values[rows_per_event * i + j, 0]
            rows[i, j, :] = np.fromstring(s, sep=",", dtype=float)
    return rows


def choose_alp_row_index(rows456: np.ndarray, ma: float) -> int:
    """
    rows456: shape (n_events, 3, 4) corresponding to original rows [4,5,6]
    Returns: 0/1/2 (index within rows456) for ALP candidate.
    """
    target = ma * ma
    m2 = np.apply_along_axis(inv_mass_sq, 2, rows456)  # (n_events, 3)
    # Score each of the 3 rows by how close it is to ma^2 (robust median)
    scores = np.median(np.abs(m2 - target), axis=0)
    return int(np.argmin(scores))


def check_file(
    csv_path: str,
    ma: float,
    n_events: int,
    tol_rel_alp_m2: float,
    require_alp_maxE: bool,
) -> FileCheckResult:
    rows = load_rows_for_events(csv_path, n_events=n_events, rows_per_event=7)
    n_use = rows.shape[0]

    rows456 = rows[:, 4:7, :]  # (n_events, 3, 4)
    alp_idx_012 = choose_alp_row_index(rows456, ma=ma)
    alp_row = 4 + alp_idx_012

    # Energies in rows 4–6
    E = rows456[:, :, 0]  # (n_events, 3)
    maxE_idx = np.argmax(E, axis=1)
    frac_alp_is_maxE = float(np.mean(maxE_idx == alp_idx_012))

    # Mass-shell checks
    m2 = np.apply_along_axis(inv_mass_sq, 2, rows456)  # (n_events, 3)
    alp_m2 = m2[:, alp_idx_012]
    target = ma * ma
    denom = max(abs(target), 1e-12)
    alp_mass_ok = np.abs(alp_m2 - target) / denom <= tol_rel_alp_m2
    frac_alp_mass_ok = float(np.mean(alp_mass_ok))

    # Optionally require ALP maxE in the suggested mapping: if badly violated,
    # fall back to picking the max-E row as ALP (common heuristic).
    if require_alp_maxE and frac_alp_is_maxE < 0.5:
        # Pick most-often-maxE row as ALP.
        counts = np.bincount(maxE_idx, minlength=3)
        alp_idx_012 = int(np.argmax(counts))
        alp_row = 4 + alp_idx_012
        frac_alp_is_maxE = float(np.mean(maxE_idx == alp_idx_012))
        alp_m2 = m2[:, alp_idx_012]
        alp_mass_ok = np.abs(alp_m2 - target) / denom <= tol_rel_alp_m2
        frac_alp_mass_ok = float(np.mean(alp_mass_ok))

    return FileCheckResult(
        path=csv_path,
        ma=ma,
        n_events=n_use,
        alp_row_suggested=alp_row,
        frac_alp_is_maxE=frac_alp_is_maxE,
        frac_alp_mass_ok=frac_alp_mass_ok,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "cmsrun3-csvs-merged"),
        help="Directory containing merged_ma_*GeV.csv files",
    )
    ap.add_argument("--max-events", type=int, default=200, help="Events to sample per file")
    ap.add_argument(
        "--tol-rel-alp-m2",
        type=float,
        default=0.05,
        help="Relative tolerance for ALP mass-shell: |m2-ma^2|/ma^2",
    )
    ap.add_argument(
        "--require-alp-maxE",
        action="store_true",
        help="If enabled and suggested ALP is rarely max-E, fall back to max-E heuristic.",
    )
    args = ap.parse_args()

    csv_dir = os.path.abspath(args.csv_dir)
    if not os.path.isdir(csv_dir):
        raise SystemExit(f"csv dir not found: {csv_dir}")

    results: List[FileCheckResult] = []
    for path in iter_merged_csvs(csv_dir):
        ma = extract_ma_from_filename(path)
        if ma is None:
            continue
        res = check_file(
            path,
            ma=ma,
            n_events=args.max_events,
            tol_rel_alp_m2=args.tol_rel_alp_m2,
            require_alp_maxE=args.require_alp_maxE,
        )
        results.append(res)

    if not results:
        print(f"No merged_ma_*GeV.csv files found in {csv_dir}")
        return 0

    print(f"Checked {len(results)} file(s) in {csv_dir}")
    print("Columns: file  ma  events  suggested_ROW_ALP  frac(ALP=maxE)  frac(ALP mass ok)")
    for r in results:
        print(
            f"{os.path.basename(r.path)}"
            f"\tma={r.ma:.4g}"
            f"\tN={r.n_events}"
            f"\tROW_ALP~{r.alp_row_suggested}"
            f"\tmaxE={r.frac_alp_is_maxE:.3f}"
            f"\tmassOK={r.frac_alp_mass_ok:.3f}"
        )

    # Summarize: do we see a consistent suggested ALP row?
    counts = {4: 0, 5: 0, 6: 0}
    for r in results:
        counts[r.alp_row_suggested] += 1
    best_row = max(counts, key=lambda k: counts[k])
    print()
    print(f"Suggested ROW_ALP per-file counts: {counts}  (most common: {best_row})")
    print("If this is consistently not 4, consider updating ROW_ALP/ROW_G1/ROW_G2 in module_VBF.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


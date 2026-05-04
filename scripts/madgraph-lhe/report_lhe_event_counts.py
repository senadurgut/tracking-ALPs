#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Record:
    run_dir: Path
    lhe_path: Optional[Path]
    n_events: Optional[int]
    error: Optional[str]

MASS_RE = re.compile(r"_ma_(?P<mass>[^_]+GeV)(?:_\d+)?$")


def mass_point_from_cmsrun_dirname(name: str) -> str:
    """
    Extract the mass token from a CMSRun3 directory name.

    Examples:
      CMSRun3_vbf_ax_ma_0p0100GeV      -> 0p0100GeV
      CMSRun3_vbf_ax_ma_0p0100GeV_4    -> 0p0100GeV
    """
    m = MASS_RE.search(name)
    return m.group("mass") if m else "UNKNOWN"


def iter_cmsrun_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("CMSRun3_"):
            yield p


def iter_run_dirs(cmsrun_dir: Path) -> Iterable[Path]:
    events = cmsrun_dir / "Events"
    if not events.is_dir():
        return
    for p in sorted(events.iterdir()):
        if p.is_dir() and p.name.startswith("run_"):
            yield p


def pick_lhe_file(run_dir: Path, pattern: str) -> Optional[Path]:
    # If pattern is an exact filename, prefer that.
    direct = run_dir / pattern
    if direct.exists():
        return direct
    matches = sorted(run_dir.glob(pattern))
    if matches:
        return matches[0]
    # Common fallbacks
    for candidate in ("unweighted_events.lhe.gz", "unweighted_events.lhe", "*.lhe.gz", "*.lhe"):
        c_matches = sorted(run_dir.glob(candidate))
        if c_matches:
            return c_matches[0]
    return None


def count_lhe_events(path: Path) -> int:
    """
    Count events in an LHE/LHE.GZ file by counting lines that start with '<event'.
    This is fast and robust for standard MG5 LHE formatting.
    """
    count = 0
    if path.suffix == ".gz":
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8", errors="replace")  # noqa: E731
    else:
        opener = lambda p: p.open("r", encoding="utf-8", errors="replace")  # noqa: E731

    with opener(path) as f:
        for line in f:
            # Allow leading whitespace just in case.
            if line.lstrip().startswith("<event"):
                count += 1
    return count


def scan(root: Path, pattern: str) -> list[Record]:
    out: list[Record] = []
    for cmsrun in iter_cmsrun_dirs(root):
        for run_dir in iter_run_dirs(cmsrun):
            lhe = pick_lhe_file(run_dir, pattern=pattern)
            if lhe is None:
                out.append(Record(run_dir=run_dir, lhe_path=None, n_events=None, error="no lhe file found"))
                continue
            try:
                n = count_lhe_events(lhe)
            except OSError as e:
                out.append(Record(run_dir=run_dir, lhe_path=lhe, n_events=None, error=str(e)))
                continue
            out.append(Record(run_dir=run_dir, lhe_path=lhe, n_events=n, error=None))
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count number of <event> blocks in each LHE file under CMSRun3_*/Events/run_*."
    )
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Path to MG5_aMC_v3_5_13 directory (default: script directory).",
    )
    p.add_argument(
        "--pattern",
        default="unweighted_events.lhe.gz",
        help=(
            "Glob pattern (or exact filename) to select the LHE file inside each run_ directory "
            "(default: unweighted_events.lhe.gz)."
        ),
    )
    p.add_argument(
        "--only-ok",
        action="store_true",
        help="Only print runs where an LHE file was found and counted successfully.",
    )
    p.add_argument(
        "--totals",
        action="store_true",
        help="Print total events summed over all successful runs.",
    )
    p.add_argument(
        "--mass-totals",
        action="store_true",
        help="Also print totals grouped by mass point (the ..._ma_<MASS>GeV suffix).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    records = scan(root=root, pattern=args.pattern)

    total = 0
    n_ok = 0
    n_warn = 0
    mass_totals: dict[str, int] = {}
    mass_ok_runs: dict[str, int] = {}
    mass_dirs: dict[str, set[str]] = {}

    for r in records:
        ok = r.error is None
        if args.only_ok and not ok:
            continue
        if ok:
            assert r.lhe_path is not None and r.n_events is not None
            print(f"{r.run_dir}\t{r.lhe_path}\t{r.n_events}")
            total += r.n_events
            n_ok += 1

            cmsrun_dir = r.run_dir.parent.parent  # .../CMSRun3_*/Events/run_*
            mass = mass_point_from_cmsrun_dirname(cmsrun_dir.name)
            mass_totals[mass] = mass_totals.get(mass, 0) + r.n_events
            mass_ok_runs[mass] = mass_ok_runs.get(mass, 0) + 1
            mass_dirs.setdefault(mass, set()).add(cmsrun_dir.name)
        else:
            n_warn += 1
            lhe_str = str(r.lhe_path) if r.lhe_path is not None else "MISSING_LHE"
            print(f"{r.run_dir}\t{lhe_str}\tERROR\t{r.error}")

    if args.mass_totals:
        for mass in sorted(mass_totals.keys()):
            dirs = mass_dirs.get(mass, set())
            print(f"MASS_TOTAL\t{mass}\tRUNS_OK={mass_ok_runs.get(mass, 0)}\tDIRS={len(dirs)}\tEVENTS={mass_totals[mass]}")

    if args.totals:
        print(f"TOTAL_OK_RUNS\t{n_ok}")
        print(f"TOTAL_WARN_RUNS\t{n_warn}")
        print(f"TOTAL_EVENTS\t{total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


#!/usr/bin/env python3
"""
Delete Events/run_* directories whose banner lists ebeam1 and ebeam2 equal to a
given **wrong** beam energy (default 6500 GeV).

Runs with **6800** GeV beams are the correct CMS 13 TeV setting and are **not**
selected for deletion unless you pass a different ``--ebeam``.

Traverses MG5 root: only subfolders whose name starts with ``CMSRun3``.
For each ``Events/run_*``, finds ``run_*_tag_*_banner.txt`` and checks that both
``<energy> = ebeam1`` and ``<energy> = ebeam2`` appear (MadGraph-style lines).
If both match, the entire ``run_*`` directory is removed.

After that, if ``Events/`` contains no remaining ``run_*`` subdirectories, the whole
``CMSRun3*`` process directory is removed as well.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def iter_run_dirs(events: Path) -> list[Path]:
    if not events.is_dir():
        return []
    return sorted(
        p
        for p in events.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )


def cmsrun_process_dir(run_dir: Path) -> Path:
    """run_dir is .../CMSRun3_*/Events/run_NN."""
    return run_dir.parent.parent


def predict_cmsrun_dirs_to_remove(matches: list[Candidate]) -> list[Path]:
    """CMSRun3 dirs that would have no run_* left after deleting ``matches``."""
    if not matches:
        return []
    to_delete_runs = {c.run_dir for c in matches}
    affected = {cmsrun_process_dir(c.run_dir) for c in matches}
    out: list[Path] = []
    for cms in sorted(affected):
        events = cms / "Events"
        if not events.is_dir():
            out.append(cms)
            continue
        all_runs = iter_run_dirs(events)
        remaining = [p for p in all_runs if p not in to_delete_runs]
        if not remaining:
            out.append(cms)
    return out


@dataclass(frozen=True)
class Candidate:
    run_dir: Path
    banner: Path


def iter_cmsrun_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("CMSRun3"):
            yield p


def find_banner(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("run_*_tag_*_banner.txt"))
    if not matches:
        return None
    return matches[0]


def banner_has_both_beams(banner_path: Path, energy: float) -> bool:
    text = banner_path.read_text(errors="replace")
    # Match lines like "  6500.0	= ebeam1 ! ..." (tabs or spaces OK)
    esc = re.escape(f"{energy:g}")
    esc_dot0 = re.escape(f"{energy:.1f}")
    pat1 = re.compile(
        rf"(?m)^\s*(?:{esc}|{esc_dot0})\s*=\s*ebeam1\b",
    )
    pat2 = re.compile(
        rf"(?m)^\s*(?:{esc}|{esc_dot0})\s*=\s*ebeam2\b",
    )
    return pat1.search(text) is not None and pat2.search(text) is not None


def scan(root: Path, energy: float) -> list[Candidate]:
    out: list[Candidate] = []
    for cmsrun in iter_cmsrun_dirs(root):
        events = cmsrun / "Events"
        if not events.is_dir():
            continue
        for run_dir in sorted(events.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            banner = find_banner(run_dir)
            if banner is None:
                continue
            if banner_has_both_beams(banner, energy):
                out.append(Candidate(run_dir=run_dir, banner=banner))
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Delete Events/run_* dirs whose banner has ebeam1=ebeam2=<energy>. "
            "Default energy is 6500 GeV (obsolete); 6800 GeV runs are kept."
        )
    )
    p.add_argument(
        "--root",
        type=str,
        default="/home/export/sdurgut/scratch/alps/tools/MG5_aMC_v3_5_13",
        help="Path to MG5_aMC installation root (contains CMSRun3_* process dirs).",
    )
    p.add_argument(
        "--ebeam",
        type=float,
        default=6500.0,
        help="Beam energy in GeV to match on ebeam1 and ebeam2 (default: 6500).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matching run_* directories. Otherwise dry-run.",
    )
    p.add_argument(
        "--print-matches",
        action="store_true",
        help="Print each run_* directory that would be deleted.",
    )
    p.add_argument(
        "--print-cmsrun-removals",
        action="store_true",
        help="Print each CMSRun3* directory that would be deleted (no runs left).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    if not root.is_dir():
        print(f"[ERROR] Root is not a directory: {root}", file=sys.stderr)
        return 1

    matches = scan(root, args.ebeam)

    cmsrun_to_remove = predict_cmsrun_dirs_to_remove(matches)

    if args.print_matches:
        for c in matches:
            print(str(c.run_dir))

    if args.print_cmsrun_removals:
        for cms in cmsrun_to_remove:
            print(str(cms))

    if not args.execute:
        print(
            f"[DRY-RUN] Would delete {len(matches)} run_* directories under "
            f"{root} (ebeam1=ebeam2={args.ebeam:g} GeV)"
        )
        print(
            f"[DRY-RUN] Would delete {len(cmsrun_to_remove)} entire CMSRun3* "
            f"directories (no run_* left under Events/)"
        )
        return 0

    deleted = 0
    for c in matches:
        try:
            shutil.rmtree(c.run_dir)
            deleted += 1
        except OSError as e:
            print(f"[ERROR] Failed to delete {c.run_dir}: {e}", file=sys.stderr)

    affected_cmsruns = sorted({cmsrun_process_dir(c.run_dir) for c in matches})
    deleted_cmsrun = 0
    for cms in affected_cmsruns:
        if not cms.is_dir():
            continue
        events_dir = cms / "Events"
        if events_dir.is_dir():
            if iter_run_dirs(events_dir):
                continue
        try:
            shutil.rmtree(cms)
            deleted_cmsrun += 1
        except OSError as e:
            print(f"[ERROR] Failed to delete {cms}: {e}", file=sys.stderr)

    print(
        f"[OK] Deleted {deleted}/{len(matches)} run_* directories "
        f"(ebeam={args.ebeam:g} GeV)"
    )
    print(
        f"[OK] Removed {deleted_cmsrun} CMSRun3* process directories "
        f"(Events/ had no remaining run_* subdirectories)"
    )
    return 0 if deleted == len(matches) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

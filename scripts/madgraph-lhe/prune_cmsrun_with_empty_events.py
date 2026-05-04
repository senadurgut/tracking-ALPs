#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IGNORE_EVENTS_ENTRIES = {".DS_Store"}


@dataclass(frozen=True)
class Candidate:
    cmsrun_dir: Path
    events_dir: Path


def iter_cmsrun_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("CMSRun3_"):
            yield p


def events_effectively_empty(events_dir: Path) -> bool:
    try:
        for entry in events_dir.iterdir():
            if entry.name in IGNORE_EVENTS_ENTRIES:
                continue
            return False
    except OSError:
        return False
    return True


def scan(root: Path) -> list[Candidate]:
    out: list[Candidate] = []
    for cmsrun in iter_cmsrun_dirs(root):
        events = cmsrun / "Events"
        if not events.is_dir():
            continue
        if events_effectively_empty(events):
            out.append(Candidate(cmsrun_dir=cmsrun, events_dir=events))
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delete CMSRun3_* directories whose Events/ directory is empty."
    )
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Path to MG5_aMC_v3_5_13 directory (default: script directory).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matching CMSRun3_* directories. Otherwise do a dry-run.",
    )
    p.add_argument(
        "--print-matches",
        action="store_true",
        help="Print each CMSRun3_* directory that would be deleted.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    matches = scan(root)

    if args.print_matches:
        for c in matches:
            print(f"{c.cmsrun_dir}\t(empty {c.events_dir})")

    if not args.execute:
        print(f"[DRY-RUN] Would delete {len(matches)} CMSRun3_* directories under {root}")
        return 0

    deleted = 0
    for c in matches:
        try:
            shutil.rmtree(c.cmsrun_dir)
            deleted += 1
        except OSError as e:
            print(f"[ERROR] Failed to delete {c.cmsrun_dir}: {e}", file=sys.stderr)

    print(f"[OK] Deleted {deleted}/{len(matches)} CMSRun3_* directories under {root}")
    return 0 if deleted == len(matches) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


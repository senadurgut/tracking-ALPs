#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DECAY_36_RE = re.compile(r"^\s*DECAY\s+36\b", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    run_dir: Path
    banner_path: Optional[Path]
    lines: Optional[list[str]]  # list of "LINE_NO:content" strings (no trailing newline)
    error: Optional[str]


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


def pick_banner_file(run_dir: Path) -> Optional[Path]:
    preferred = sorted(run_dir.glob("*_tag_1_banner.txt"))
    if preferred:
        return preferred[0]
    any_banner = sorted(run_dir.glob("*banner*.txt"))
    if any_banner:
        return any_banner[0]
    txt = sorted(run_dir.glob("*.txt"))
    return txt[0] if txt else None


def read_matching_lines(path: Path, pattern: re.Pattern[str]) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if pattern.match(line):
                out.append(f"{i}:{line.rstrip()}")
    return out


def scan(root: Path, pattern: re.Pattern[str]) -> list[Record]:
    records: list[Record] = []
    for cmsrun in iter_cmsrun_dirs(root):
        for run_dir in iter_run_dirs(cmsrun):
            banner = pick_banner_file(run_dir)
            if banner is None:
                records.append(
                    Record(run_dir=run_dir, banner_path=None, lines=None, error="no .txt banner found")
                )
                continue
            try:
                lines = read_matching_lines(banner, pattern=pattern)
            except OSError as e:
                records.append(
                    Record(run_dir=run_dir, banner_path=banner, lines=None, error=str(e))
                )
                continue

            if len(lines) == 0:
                records.append(
                    Record(
                        run_dir=run_dir,
                        banner_path=banner,
                        lines=lines,
                        error="no matching lines",
                    )
                )
            else:
                records.append(Record(run_dir=run_dir, banner_path=banner, lines=lines, error=None))
    return records


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Print lines matching a pattern from each run_*/...banner*.txt under "
            "CMSRun3_*/Events/run_*. Default pattern matches lines starting with 'DECAY  36'."
        )
    )
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Path to MG5_aMC_v3_5_13 directory (default: script directory).",
    )
    p.add_argument(
        "--pattern",
        default=r"^\s*DECAY\s+36\b",
        help=r"Regex pattern to match (default: ^\s*DECAY\s+36\b).",
    )
    p.add_argument(
        "--only-ok",
        action="store_true",
        help="Only print entries where at least one matching line was found.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        pattern = re.compile(args.pattern, re.IGNORECASE)
    except re.error as e:
        print(f"[ERROR] Invalid regex pattern: {e}", file=sys.stderr)
        return 2

    root = Path(args.root).expanduser().resolve()
    records = scan(root=root, pattern=pattern)

    for r in records:
        if r.banner_path is None:
            if not args.only_ok:
                print(f"{r.run_dir}\tMISSING_BANNER\t{r.error}")
            continue

        ok = r.error is None
        if args.only_ok and not ok:
            continue

        status = "OK" if ok else "WARN"
        print(f"{r.run_dir}\t{r.banner_path}\t{status}")
        if r.lines:
            for ln in r.lines:
                print(f"  {ln}")
        if r.error and not args.only_ok:
            print(f"  ERROR: {r.error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


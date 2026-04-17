#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DECAY_36_RE = re.compile(r"^\s*DECAY\s+36\s+(?P<width>\S+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    cmsrun_dir: Path
    param_card: Path
    width_token: Optional[str]  # None if missing/unreadable


def iter_cmsrun_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("CMSRun3_"):
            yield p


def extract_decay_36_width_token(param_card: Path) -> Optional[str]:
    try:
        with param_card.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = DECAY_36_RE.match(line)
                if m:
                    return m.group("width")
    except OSError:
        return None
    return None


def scan(root: Path) -> list[Record]:
    out: list[Record] = []
    for cmsrun in iter_cmsrun_dirs(root):
        param_card = cmsrun / "Cards" / "param_card.dat"
        if not param_card.is_file():
            out.append(Record(cmsrun_dir=cmsrun, param_card=param_card, width_token=None))
            continue
        token = extract_decay_36_width_token(param_card)
        out.append(Record(cmsrun_dir=cmsrun, param_card=param_card, width_token=token))
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report DECAY 36 width tokens from Cards/param_card.dat in CMSRun3_* directories."
    )
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Path to MG5_aMC_v3_5_13 directory (default: script directory).",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Only print CMSRun3_* entries where DECAY 36 is missing or param_card.dat is missing.",
    )
    p.add_argument(
        "--print-name-only",
        action="store_true",
        help="Print only the CMSRun3_* directory name (useful with --only-missing).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    rows = scan(root)

    printed = 0
    for r in rows:
        missing = r.width_token is None
        if args.only_missing and not missing:
            continue

        if args.print_name_only:
            print(r.cmsrun_dir.name)
        else:
            width = r.width_token if r.width_token is not None else "MISSING"
            print(f"{r.cmsrun_dir}\t{width}")
        printed += 1

    return 0 if printed >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


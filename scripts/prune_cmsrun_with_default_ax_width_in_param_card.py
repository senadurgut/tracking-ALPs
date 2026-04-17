#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DECAY_LINE_RE = re.compile(r"^\s*DECAY\s+36\s+(?P<width>\S+)\b")


@dataclass(frozen=True)
class Match:
    cmsrun_dir: Path
    param_card: Path
    width_token: str


def iter_cmsrun_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("CMSRun3_"):
            yield p


def find_width_token_in_param_card(param_card: Path) -> Optional[str]:
    try:
        with param_card.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = DECAY_LINE_RE.match(line)
                if m:
                    return m.group("width")
    except OSError:
        return None
    return None


def width_matches(width_token: str, target_token: str, float_tol: Optional[float]) -> bool:
    if width_token == target_token:
        return True
    if float_tol is None:
        return False
    try:
        w = float(width_token)
        t = float(target_token)
    except ValueError:
        return False
    return abs(w - t) <= float_tol


def scan(root: Path, target_width: str, float_tol: Optional[float]) -> list[Match]:
    matches: list[Match] = []
    for cmsrun in iter_cmsrun_dirs(root):
        param_card = cmsrun / "Cards" / "param_card.dat"
        if not param_card.is_file():
            continue
        token = find_width_token_in_param_card(param_card)
        if token is None:
            continue
        if width_matches(token, target_token=target_width, float_tol=float_tol):
            matches.append(Match(cmsrun_dir=cmsrun, param_card=param_card, width_token=token))
    return matches


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Delete CMSRun3_* directories if Cards/param_card.dat contains "
            "DECAY  36 <width> with a matching width token."
        )
    )
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Path to MG5_aMC_v3_5_13 directory (default: script directory).",
    )
    p.add_argument(
        "--width",
        default="1.000000e+00",
        help="Width token to match for DECAY 36 (default: 1.000000e+00).",
    )
    p.add_argument(
        "--float-tol",
        type=float,
        default=None,
        help=(
            "Optional float tolerance to match widths numerically if string token differs. "
            "Example: --float-tol 1e-12. If unset, matching is exact-token only."
        ),
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matching CMSRun3_* directories. Otherwise do a dry-run.",
    )
    p.add_argument(
        "--print-matches",
        action="store_true",
        help="Print each matched CMSRun3_* directory and the file/token used.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    matches = scan(root=root, target_width=args.width, float_tol=args.float_tol)

    if args.print_matches:
        for m in matches:
            print(f"{m.cmsrun_dir}\t{m.param_card}\tWIDTH={m.width_token}")

    if not args.execute:
        print(f"[DRY-RUN] Would delete {len(matches)} CMSRun3_* directories under {root}")
        return 0

    deleted = 0
    for m in matches:
        try:
            shutil.rmtree(m.cmsrun_dir)
            deleted += 1
        except OSError as e:
            print(f"[ERROR] Failed to delete {m.cmsrun_dir}: {e}", file=sys.stderr)

    print(f"[OK] Deleted {deleted}/{len(matches)} CMSRun3_* directories under {root}")
    return 0 if deleted == len(matches) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


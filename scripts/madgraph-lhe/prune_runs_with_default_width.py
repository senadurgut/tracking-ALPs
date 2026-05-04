#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DECAY_RE = re.compile(r"^\s*DECAY\s+(?P<pdg>-?\d+)\s+(?P<width>\S+)\b")


@dataclass(frozen=True)
class Match:
    cmsrun_dir: Path
    run_dir: Path
    banner_path: Path
    pdg: int
    width_token: str


def iter_cmsrun_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("CMSRun3_"):
            yield p


def iter_run_dirs(cmsrun_dir: Path) -> Iterable[Path]:
    events_dir = cmsrun_dir / "Events"
    if not events_dir.is_dir():
        return
    for p in sorted(events_dir.iterdir()):
        if p.is_dir() and p.name.startswith("run_"):
            yield p


def candidate_text_files(run_dir: Path) -> list[Path]:
    banner = sorted(run_dir.glob("*banner*.txt"))
    if banner:
        return banner
    txt = sorted(run_dir.glob("*.txt"))
    return txt


def find_decay_width_token(text_path: Path, pdg_id: int) -> Optional[str]:
    try:
        with text_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = DECAY_RE.match(line)
                if not m:
                    continue
                if int(m.group("pdg")) == pdg_id:
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


def scan(root: Path, pdg_id: int, target_width: str, float_tol: Optional[float]) -> list[Match]:
    matches: list[Match] = []
    for cmsrun in iter_cmsrun_dirs(root):
        for run_dir in iter_run_dirs(cmsrun):
            for text_path in candidate_text_files(run_dir):
                token = find_decay_width_token(text_path, pdg_id=pdg_id)
                if token is None:
                    continue
                if width_matches(token, target_token=target_width, float_tol=float_tol):
                    matches.append(
                        Match(
                            cmsrun_dir=cmsrun,
                            run_dir=run_dir,
                            banner_path=text_path,
                            pdg=pdg_id,
                            width_token=token,
                        )
                    )
                    break
    return matches


def delete_run_dirs(matches: list[Match], execute: bool) -> int:
    deleted = 0
    for m in matches:
        if execute:
            try:
                shutil.rmtree(m.run_dir)
                deleted += 1
            except OSError as e:
                print(f"[ERROR] Failed to delete {m.run_dir}: {e}", file=sys.stderr)
        else:
            deleted += 1
    return deleted


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Delete MG5 run directories whose banner text contains a DECAY width "
            "line matching a specific PDG id and width token."
        )
    )
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Path to MG5_aMC_v3_5_13 directory (default: script directory).",
    )
    p.add_argument("--pdg", type=int, default=36, help="PDG id to check (default: 36).")
    p.add_argument(
        "--width",
        default="1.000000e+00",
        help="Width token to match (default: 1.000000e+00).",
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
        help="Actually delete matching run directories. Otherwise do a dry-run.",
    )
    p.add_argument(
        "--print-matches",
        action="store_true",
        help="Print each matched run directory and the file/line token used.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    matches = scan(root=root, pdg_id=args.pdg, target_width=args.width, float_tol=args.float_tol)

    if args.print_matches:
        for m in matches:
            print(f"{m.run_dir}\t{m.banner_path}\tPDG={m.pdg}\tWIDTH={m.width_token}")

    if not args.execute:
        print(f"[DRY-RUN] Would delete {len(matches)} run directories under {root}")
        return 0

    deleted = delete_run_dirs(matches, execute=True)
    print(f"[OK] Deleted {deleted}/{len(matches)} run directories under {root}")
    return 0 if deleted == len(matches) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


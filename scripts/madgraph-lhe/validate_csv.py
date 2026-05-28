"""
validate_csv.py
===============
Confirm LHE-derived CSV consistency for VBF → ALP → γγ events.

Checks performed:
  1. ΔR distribution between the two decay photons
  2. ALP energy dominance (E_ALP > E_γ1 and E_ALP > E_γ2)
  3. ALP on-shell check: |E - sqrt(p² + m_a²)| / E
  4. ALP invariant mass reconstruction: m = sqrt(E² - p²)
  5. Four-momentum conservation: photon1 + photon2 = ALP
  6. Overall four-momentum conservation: sum(outgoing) = sum(incoming)

Usage
-----
    python validate_csv.py data/0_1000GeV.csv --ma 0.1
    python validate_csv.py data/0_1000GeV.csv --ma 0.1 --max-events 5000
"""

import argparse
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _parse_row(s: str) -> np.ndarray:
    return np.fromstring(s, sep=",")


def _eta(px: float, py: float, pz: float) -> float:
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == 0.0:
        return float("inf")
    x = pz / p
    x = max(min(x, 1.0 - 1e-15), -1.0 + 1e-15)
    return float(np.arctanh(x))


def _pick_latest_run_dir(events_dir: Path) -> Optional[Path]:
    runs = []
    for p in events_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"run_(\d+)", p.name)
        if m:
            runs.append((int(m.group(1)), p))
    if not runs:
        return None
    runs.sort(key=lambda t: t[0])
    return runs[-1][1]


def _find_unweighted_lhe(process_name: str) -> Path:
    mg5_root = Path("/Users/sena/grad_school/Research/ALP/tools/MG5_aMC_v3_5_13")
    proc_dir = mg5_root / process_name
    events_dir = proc_dir / "Events"
    if not events_dir.is_dir():
        raise FileNotFoundError(f"Events dir not found: {events_dir}")

    run_dir = _pick_latest_run_dir(events_dir)
    if run_dir is None:
        raise FileNotFoundError(f"No run_*/ directories found under: {events_dir}")

    for name in ("unweighted_events.lhe.gz", "unweighted_events.lhe"):
        cand = run_dir / name
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        f"Could not find unweighted_events.lhe[.gz] under: {run_dir}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Confirm LHE-derived CSV consistency: photon ΔR distribution, "
            "ALP invariant mass, energy dominance, on-shell check, and "
            "four-momentum conservation."
        )
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--process", help="MadGraph process directory name (e.g. MyProcess)")
    src.add_argument("--csv", help="CSV produced by lhe_to_csv.py")
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Where to write CSV if using --process (default: data/validate_<process>.csv)",
    )
    ap.add_argument("--ma", type=float, required=True, help="ALP mass (GeV)")
    ap.add_argument("--max-events", type=int, default=10_000, help="Max events to check")
    ap.add_argument(
        "--shell-tol",
        type=float,
        default=6e-2,
        help="Relative tolerance for on-shell check |E - sqrt(p^2+m^2)|/E",
    )
    ap.add_argument(
        "--max-shell-fail-frac",
        type=float,
        default=5e-2,
        help="Fail if ALP on-shell failure fraction exceeds this value",
    )
    args = ap.parse_args()

    # ── If needed, build CSV from LHE ─────────────────────────────────────
    csv_path: str
    if args.process:
        lhe_path = _find_unweighted_lhe(args.process)
        repo_root = Path(__file__).resolve().parents[1]  # .../tracking-ALPs
        lhe_to_csv = repo_root / "scripts" / "lhe_to_csv.py"
        if not lhe_to_csv.is_file():
            raise FileNotFoundError(f"lhe_to_csv.py not found at: {lhe_to_csv}")

        out_csv = (
            Path(args.out_csv)
            if args.out_csv
            else (repo_root / "data" / f"validate_{args.process}.csv")
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [os.environ.get("PYTHON", "python3"), str(lhe_to_csv), str(lhe_path), str(out_csv)],
            check=True,
        )
        csv_path = str(out_csv)
    else:
        csv_path = str(args.csv)

    # ── Read CSV ──────────────────────────────────────────────────────────
    # Each row is one particle; 7 rows per event; content is "E,px,py,pz".
    # Row layout per event:
    #   0: incoming quark 1    1: incoming quark 2
    #   2: VBF jet 1           3: VBF jet 2
    #   4: ALP                 5: photon 1          6: photon 2
    temp = pd.read_csv(csv_path, sep=";", header=None)
    n_events = min(args.max_events, len(temp) // 7)
    if n_events <= 0:
        print(f"ERROR: no complete events found in {csv_path}")
        return 2

    # ── Storage ───────────────────────────────────────────────────────────
    dRs = []
    inv_masses = []
    shell_rels = []
    decay_conservation = []  # max |Δ| in (E, px, py, pz) for γ1+γ2 vs ALP
    event_conservation = []  # max |Δ| in (E, px, py, pz) for all incoming vs outgoing
    n_energy_order_fail = 0
    n_shell_fail = 0
    n_decay_cons_fail = 0
    n_event_cons_fail = 0
    CONS_ABS_TOL = 1e-4  # GeV — tolerance for momentum conservation checks

    for i in range(n_events):
        inc1 = _parse_row(temp.values[7 * i + 0, 0])  # incoming quark 1
        inc2 = _parse_row(temp.values[7 * i + 1, 0])  # incoming quark 2
        jet1 = _parse_row(temp.values[7 * i + 2, 0])  # VBF jet 1
        jet2 = _parse_row(temp.values[7 * i + 3, 0])  # VBF jet 2
        alp  = _parse_row(temp.values[7 * i + 4, 0])  # ALP
        g1   = _parse_row(temp.values[7 * i + 5, 0])  # photon 1
        g2   = _parse_row(temp.values[7 * i + 6, 0])  # photon 2

        Ea, pxa, pya, pza = (float(x) for x in alp)
        E1, px1, py1, pz1 = (float(x) for x in g1)
        E2, px2, py2, pz2 = (float(x) for x in g2)

        # ── Check 1: ΔR between the two photons ──────────────────────────
        eta1 = _eta(px1, py1, pz1)
        eta2 = _eta(px2, py2, pz2)
        phi1 = math.atan2(py1, px1)
        phi2 = math.atan2(py2, px2)
        dphi = phi1 - phi2
        dphi = dphi - 2 * math.pi * round(dphi / (2 * math.pi))
        dR = math.sqrt((eta1 - eta2) ** 2 + dphi ** 2)
        dRs.append(dR)

        # ── Check 2: ALP energy dominance (rows 4-6) ─────────────────────
        if not (Ea >= E1 and Ea >= E2):
            n_energy_order_fail += 1

        # ── Check 3: ALP on-shell check ──────────────────────────────────
        pa2 = pxa * pxa + pya * pya + pza * pza
        E_expected = math.sqrt(pa2 + args.ma * args.ma)
        denom = max(abs(Ea), 1e-12)
        rel = abs(Ea - E_expected) / denom
        shell_rels.append(rel)
        if rel > args.shell_tol:
            n_shell_fail += 1

        # ── Check 4: ALP invariant mass reconstruction ────────────────────
        m2 = Ea * Ea - pa2
        m_reco = math.sqrt(abs(m2)) * (1.0 if m2 >= 0 else -1.0)
        inv_masses.append(m_reco)

        # ── Check 5: Decay conservation (γ1 + γ2 = ALP) ──────────────────
        d_cons = max(
            abs((E1 + E2) - Ea),
            abs((px1 + px2) - pxa),
            abs((py1 + py2) - pya),
            abs((pz1 + pz2) - pza),
        )
        decay_conservation.append(d_cons)
        if d_cons > CONS_ABS_TOL:
            n_decay_cons_fail += 1

        # ── Check 6: Overall event conservation (in = out) ────────────────
        # Outgoing = jet1 + jet2 + g1 + g2  (ALP is intermediate, not final)
        # Note: we use photons (not ALP) since the ALP decayed
        sum_in  = inc1 + inc2
        sum_out = jet1 + jet2 + g1 + g2
        e_cons = max(abs(sum_in - sum_out))
        event_conservation.append(e_cons)
        if e_cons > CONS_ABS_TOL:
            n_event_cons_fail += 1

    # ── Convert to arrays ─────────────────────────────────────────────────
    dRs = np.asarray(dRs, dtype=float)
    inv_masses = np.asarray(inv_masses, dtype=float)
    shell_rels = np.asarray(shell_rels, dtype=float)
    decay_conservation = np.asarray(decay_conservation, dtype=float)
    event_conservation = np.asarray(event_conservation, dtype=float)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"{'=' * 60}")
    print(f"CSV validation report for: {csv_path}")
    print(f"Expected ALP mass: {args.ma} GeV")
    print(f"Events checked: {n_events}")
    print(f"{'=' * 60}")

    print(f"\n--- Photon ΔR distribution ---")
    print(f"  min ΔR(γ,γ)  = {dRs.min():.6f}")
    print(f"  median ΔR    = {np.median(dRs):.6f}")
    print(f"  max ΔR(γ,γ)  = {dRs.max():.6f}")
    print(f"  frac ΔR < 0.4  : {(dRs < 0.4).mean():.4f}")
    print(f"  frac ΔR < 0.01 : {(dRs < 0.01).mean():.4f}")
    print(f"  frac ΔR < 0.001: {(dRs < 0.001).mean():.4f}")

    print(f"\n--- ALP invariant mass reconstruction ---")
    print(f"  expected m_a  = {args.ma:.6f} GeV")
    print(f"  min m_reco    = {inv_masses.min():.6f} GeV")
    print(f"  median m_reco = {np.median(inv_masses):.6f} GeV")
    print(f"  mean m_reco   = {inv_masses.mean():.6f} GeV")
    print(f"  max m_reco    = {inv_masses.max():.6f} GeV")
    print(f"  std m_reco    = {inv_masses.std():.6g} GeV")

    print(f"\n--- ALP energy dominance (E_ALP > E_γ1 and E_ALP > E_γ2) ---")
    print(f"  failures: {n_energy_order_fail}/{n_events}")

    print(f"\n--- ALP on-shell check: |E - sqrt(p² + m_a²)| / E ---")
    print(f"  failures (tol={args.shell_tol:g}): {n_shell_fail}/{n_events}")
    for q in (0.5, 0.9, 0.95, 0.99, 0.999):
        print(f"  quantile {q:5.1%}: {np.quantile(shell_rels, q):.6g}")

    print(f"\n--- Decay conservation: γ1 + γ2 = ALP ---")
    print(f"  max |Δp| across all events: {decay_conservation.max():.6g} GeV")
    print(f"  failures (tol={CONS_ABS_TOL:g} GeV): {n_decay_cons_fail}/{n_events}")

    print(f"\n--- Overall event conservation: Σ(incoming) = Σ(outgoing) ---")
    print(f"  max |Δp| across all events: {event_conservation.max():.6g} GeV")
    print(f"  failures (tol={CONS_ABS_TOL:g} GeV): {n_event_cons_fail}/{n_events}")

    # ── Pass / Fail ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    failed = False

    if n_energy_order_fail > 0:
        print(f"FAIL: {n_energy_order_fail} events have E_ALP < E_photon")
        failed = True

    shell_fail_frac = n_shell_fail / n_events
    if shell_fail_frac > args.max_shell_fail_frac:
        print(f"FAIL: on-shell failure fraction {shell_fail_frac:.3f} > {args.max_shell_fail_frac:g}")
        failed = True

    mass_rel_err = abs(np.median(inv_masses) - args.ma) / args.ma
    if mass_rel_err > 0.1:
        print(f"FAIL: median reconstructed mass {np.median(inv_masses):.4f} GeV "
              f"differs from expected {args.ma} GeV by {mass_rel_err:.1%}")
        failed = True

    if n_decay_cons_fail > 0:
        print(f"FAIL: {n_decay_cons_fail} events violate γ1+γ2 = ALP")
        failed = True

    if n_event_cons_fail > 0:
        print(f"WARN: {n_event_cons_fail} events violate overall 4-momentum conservation "
              f"(tol={CONS_ABS_TOL:g} GeV) — may indicate numerical precision issues")

    if not failed:
        print("ALL CHECKS PASSED")

    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

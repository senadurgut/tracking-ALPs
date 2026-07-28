"""
Hadronic-isolation discriminant for the K_L -> gamma gamma background.

Left  : normalized isolation distributions, Run 3 (VBF/hard) vs Phase-2 (inclusive/soft) K_L.
Right : fraction of K_L retained by an isolation cut `keep iso < X`, vs X.

The ALP signal is a clean diphoton -> iso ~ 0, so `iso < X` keeps ~all signal; the right panel
is then the K_L background RETAINED at ~unit signal efficiency (its X->0 floor = the fraction of
K_L that are already isolated and which isolation therefore CANNOT remove).

Run (after the samples are regenerated with the iso columns):  python plot_kl_iso.py
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BKG = Path(__file__).resolve().parent.parent    # tracking_ALPs/bkg (data/ lives here)

C_RUN3, C_PH2 = "#0072B2", "#D55E00"   # Okabe-Ito blue / vermillion (colorblind-safe, fixed order)

def load_iso(path, col):
    s = pd.to_numeric(pd.read_csv(path, usecols=[col], low_memory=False)[col], errors="coerce")
    bad = int(s.isna().sum())
    if bad:
        print(f"  {path}: dropped {bad} non-numeric rows in '{col}' (stray repeated headers?)")
    return s.dropna().to_numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", default=str(_BKG / "data" / "kl_vbf_sample.csv"))
    ap.add_argument("--soft", default=str(_BKG / "data" / "kl_sample.csv"))
    ap.add_argument("--xmax", type=float, default=20.0, help="GeV, upper edge for the plots")
    ap.add_argument("--out", default="plots/kl_iso.png")
    args = ap.parse_args()

    r3  = load_iso(args.hard, "kl_iso_had_pt")
    ph2 = load_iso(args.soft, "iso_had_pt")
    series = [(r3, C_RUN3, "Run 3 (VBF K_L)"), (ph2, C_PH2, "Phase-2 (inclusive K_L)")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.linspace(0, args.xmax, 60)
    for x, c, lab in series:
        ax1.hist(x, bins=bins, density=True, histtype="step", lw=2, color=c, label=lab)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"hadronic activity  $\sum p_T\ (\Delta R<0.3)$  [GeV]")
    ax1.set_ylabel("normalized K_L / bin")
    ax1.legend(frameon=False)
    ax1.grid(True, which="both", alpha=0.15)

    xs = np.linspace(0, args.xmax, 200)
    for x, c, lab in series:
        xsort = np.sort(x)
        frac = np.searchsorted(xsort, xs, side="left") / len(xsort)   # fraction with iso < X
        ax2.plot(xs, frac, lw=2, color=c, label=lab)
    ax2.set_xlabel(r"activity cut  $X$  [GeV]   (keep activity $< X$)")
    ax2.set_ylabel("fraction of K_L retained")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.15)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)

    for x, _, lab in series:
        print(f"{lab:<26}: iso==0 -> {np.mean(x <= 1e-9):.3f} retained (floor);  "
              f"iso<1 -> {np.mean(x < 1):.3f};  iso<2 -> {np.mean(x < 2):.3f};  iso<5 -> {np.mean(x < 5):.3f}")

if __name__ == "__main__":
    main()

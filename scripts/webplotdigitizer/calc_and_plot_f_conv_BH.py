"""Plot photon conversion probability from a CMS material-budget file.

Reads a two-column (eta, x/X0) text file and computes:
    f_conv_true = 1 - exp(-(7/9) * x/X0)              (Bethe-Heitler)

Then plots it as a function of eta.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT = "data/material-budget/phase1-material-budget-histo.txt"
FNAME = "plots/material-budget/phase0-f-conv"


def compute_f_conv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("eta").reset_index(drop=True)
    df["f_conv_true"] = 1.0 - np.exp(-(7.0 / 9.0) * df["t_over_X0"])
    return df


def fold_abs_eta(df: pd.DataFrame, bin_width: float = 0.1,
                 eta_max: float = 2.5) -> pd.DataFrame:
    """Fold negative eta over and re-bin into uniform |eta| bins.

    Per-row f_conv_true (already nonlinear in t/X0) is averaged within each
    |eta| bin so that ±eta contribute equally — assumes equal photon
    occupancy on either side of the detector.
    """
    df = df.copy()
    df["abs_eta"] = df["eta"].abs()
    df = df[df["abs_eta"] <= eta_max]

    edges = np.arange(0.0, eta_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = pd.cut(df["abs_eta"], bins=edges, include_lowest=True, labels=False)
    df = df.assign(bin_idx=bin_idx).dropna(subset=["bin_idx"])

    grouped = df.groupby("bin_idx", observed=True).agg(
        t_over_X0=("t_over_X0", "mean"),
        f_conv_true=("f_conv_true", "mean"),
    )
    grouped["eta"] = centers[grouped.index.astype(int)]
    return grouped.reset_index(drop=True).sort_values("eta").reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Two-column file: eta, x/X0")
    p.add_argument("--out", default="cms_fconv_vs_eta",
                   help="Output basename (writes .pdf and .png)")
    p.add_argument("--print-table", action="store_true",
                   help="Print the computed table to stdout")
    eta_mode = p.add_mutually_exclusive_group()
    eta_mode.add_argument("--positive-eta", action="store_true",
                          help="Restrict plot to 0 <= eta <= 2.5")
    eta_mode.add_argument("--abs-eta", action="store_true",
                          help="Fold negative eta: bin in |eta| over [0, 2.5] "
                               "and average f_conv per bin")
    args = p.parse_args(argv)

    df = pd.read_csv(args.input, header=None, names=["eta", "t_over_X0"])
    df = compute_f_conv(df)
    if args.positive_eta:
        df = df[(df["eta"] >= 0.0) & (df["eta"] <= 2.5)].reset_index(drop=True)
    elif args.abs_eta:
        df = fold_abs_eta(df)

    xlabel = r"$|\eta|$" if args.abs_eta else r"$\eta$"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(df["eta"], df["f_conv_true"], where="mid", color="b", lw=1.2,
            label=r"$f^\mathrm{true}_\mathrm{conv} = 1 - e^{-(7/9)\,t/X_0}$")
    ax.set_ylim(0.0, 1.0)
    if args.abs_eta:
        ax.set_xlim(0.0, 2.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Conversion probability")
    ax.set_title(f"Photon conversion probability — CMS Phase-0 — {os.path.basename(args.input)}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if args.positive_eta:
        out = f"{args.out}_pos-eta"
    elif args.abs_eta:
        out = f"{args.out}_abs-eta"
    else:
        out = args.out
    plt.tight_layout()
    plt.savefig(f"{out}.png", dpi=150)
    print(f"Wrote {out}.png")

    if args.print_table:
        print(df[["eta", "t_over_X0", "f_conv_true"]].to_string())


if __name__ == "__main__":
    main()

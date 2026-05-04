"""Plot CMS material budget (t/X0 vs eta) from a two-column text file.

Usage:
    python plot_material_budget.py path/to/file.txt
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Two-column txt file: eta, t/X0")
    args = p.parse_args()

    df = pd.read_csv(args.path, header=None, names=["eta", "t_over_X0"],
                     comment="#", sep=None, engine="python")
    df = df.dropna().sort_values("eta").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["eta"], df["t_over_X0"], "k.", ms=3)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$t/X_0$")
    ax.set_title(f"CMS Phase-0 material budget — {os.path.basename(args.path)}")
    ax.set_xlim(-4.0, 4.0)
    #autocalculate ylim based on t_over_X0
    ylim = ax.get_ylim()
    ylim = (0.0, max(ylim[1], df["t_over_X0"].max()))
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots/material-budget", exist_ok=True)
    fname = os.path.basename(args.path).replace(".txt", "")
    plt.savefig(f"plots/material-budget/{fname}.png", dpi=150)
    print(f"Wrote plots/material-budget/{fname}.png")


if __name__ == "__main__":
    main()

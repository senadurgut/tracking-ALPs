"""Plot CMS material budget and photon conversion probability vs eta on one figure.

Both quantities share the eta abscissa but differ in scale:
  - t/X0          : dimensionless, ~ 0 - 2.5
  - f_conv_*      : probability  ,    0 - 1

So we use a twin y-axis (twinx): left y for t/X0, right y for the
conversion probabilities computed from the same data.

Usage:
    python plot_material_and_fconv.py path/to/file.txt
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_f_conv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("eta").reset_index(drop=True)
    df["f_conv_true"] = 1.0 - np.exp(-(7.0 / 9.0) * df["t_over_X0"])
    df["f_conv_total"] = df["f_conv_true"] * 0.70 + (1.0 - df["f_conv_true"]) * 0.05
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Two-column txt file: eta, t/X0")
    args = p.parse_args()

    df = pd.read_csv(args.path, header=None, names=["eta", "t_over_X0"],
                     comment="#", sep=None, engine="python")
    df = df.dropna()
    df = compute_f_conv(df)

    fig, ax_left = plt.subplots(figsize=(10, 5.5))
    ax_right = ax_left.twinx()

    l_mat, = ax_left.plot(df["eta"], df["t_over_X0"], "k.", ms=3,
                          label=r"$t/X_0$")
    l_true, = ax_right.plot(df["eta"], df["f_conv_true"], "b.", ms=3,
                            label=r"$f^\mathrm{true}_\mathrm{conv} = 1 - e^{-(7/9)\,t/X_0}$")
    l_total, = ax_right.plot(df["eta"], df["f_conv_total"], "r.", ms=3,
                             label=r"$f^\mathrm{total}_\mathrm{conv}$")

    ax_left.set_xlabel(r"$\eta$")
    ax_left.set_ylabel(r"$t/X_0$")
    ax_right.set_ylabel("Conversion probability")

    ax_left.set_xlim(-4.0, 4.0)
    ax_left.set_ylim(0.0, max(ax_left.get_ylim()[1], df["t_over_X0"].max()))
    ax_right.set_ylim(0.0, 1.0)

    ax_left.grid(True, alpha=0.3)
    ax_left.set_title(
        f"CMS Phase-0 material budget & photon conversion — {os.path.basename(args.path)}"
    )

    handles = [l_mat, l_true, l_total]
    ax_left.legend(handles, [h.get_label() for h in handles], loc="upper center")

    plt.tight_layout()
    os.makedirs("plots/combined", exist_ok=True)
    fname = os.path.basename(args.path).replace(".txt", "")
    plt.savefig(f"plots/combined/{fname}.png", dpi=150)
    print(f"Wrote plots/combined/{fname}.png")


if __name__ == "__main__":
    main()

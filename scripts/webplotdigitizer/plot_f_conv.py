"""Plot photon conversion probability vs eta from a two-column text file.

The input is expected to contain rows of `eta, f_conv` (already a
probability), e.g. as produced by webplotdigitizer from a CMS reference
plot. Drawn as an outlined step histogram in the CMS style.

Usage:
    python plot_f_conv.py path/to/file.txt
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Two-column txt file: eta, f_conv")
    args = p.parse_args()

    df = pd.read_csv(args.path, header=None, names=["eta", "f_conv"],
                     comment="#", sep=None, engine="python")
    df = df.dropna().sort_values("eta").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(df["eta"], df["f_conv"], where="mid", color="b", lw=1.2)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Conversion probability")
    ax.set_title(f"Photon conversion probability — {os.path.basename(args.path)}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots/conv-prob", exist_ok=True)
    fname = os.path.basename(args.path).replace(".txt", "")
    plt.savefig(f"plots/conv-prob/{fname}.png", dpi=150)
    print(f"Wrote plots/conv-prob/{fname}.png")


if __name__ == "__main__":
    main()

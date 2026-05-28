"""Reusable plotting + conversion utilities for notebooks.

This module consolidates the logic from the small CLI plotting scripts into
importable functions that return matplotlib objects.

Typical notebook usage:

```python
import sys
sys.path.append("tracking-ALPs")  # if needed from repo root
import plotting_utils as pu

fig, ax, df = pu.plot_f_conv_from_material_budget_txt(
    "data/material-budget/phase0-material-budget-histo.txt",
    eta_mode="abs",  # "full" | "positive" | "abs"
)
```
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EtaMode = Literal["full", "positive", "abs"]
PathLike = Union[str, "os.PathLike[str]"]


def _basename_no_ext(path: PathLike) -> str:
    base = os.path.basename(os.fspath(path))
    return os.path.splitext(base)[0]


def _phase_from_filename(path: PathLike) -> Optional[str]:
    """Infer Phase-0/Phase-1 tag from filename."""
    name = _basename_no_ext(path).lower()
    if "phase0" in name or "phase-0" in name:
        return "Phase-0"
    if "phase1" in name or "phase-1" in name:
        return "Phase-1"
    return None


def _with_phase(label: str, path: PathLike) -> str:
    phase = _phase_from_filename(path)
    return f"{label} ({phase})" if phase else label


def _save_png(fig: plt.Figure, savepath: Optional[PathLike]) -> Optional[str]:
    if savepath is None:
        return None
    sp = os.fspath(savepath)
    if not sp.lower().endswith(".png"):
        sp = f"{sp}.png"
    fig.savefig(sp, dpi=150)
    return sp


def read_two_column_txt(
    path: str,
    names: Tuple[str, str],
    *,
    comment: str = "#",
) -> pd.DataFrame:
    """Read a two-column CSV/whitespace-separated txt file into a dataframe."""
    df = pd.read_csv(
        path,
        header=None,
        names=list(names),
        comment=comment,
        sep=None,
        engine="python",
    )
    return df.dropna()


def bethe_heitler_f_conv_true(t_over_x0: np.ndarray | pd.Series) -> np.ndarray:
    """Bethe–Heitler conversion probability: 1 - exp(-(7/9) * t/X0)."""
    return 1.0 - np.exp(-(7.0 / 9.0) * np.asarray(t_over_x0, dtype=float))


def add_f_conv_true(df_eta_t: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with f_conv_true added (expects columns eta, t_over_X0)."""
    df = df_eta_t.copy()
    df = df.sort_values("eta").reset_index(drop=True)
    df["f_conv_true"] = bethe_heitler_f_conv_true(df["t_over_X0"])
    return df


def fold_abs_eta(
    df: pd.DataFrame,
    *,
    eta_col: str = "eta",
    t_col: str = "t_over_X0",
    f_col: str = "f_conv_true",
    eta_max: float = 2.5,
    bin_width: float = 0.1,
) -> pd.DataFrame:
    """Fold negative eta over and re-bin into uniform |eta| bins.

    Notes:
    - This averages the already-computed f_conv_true within each |eta| bin.
    - The returned dataframe uses `eta` as the bin-center in |eta|.
    """
    d = df.copy()
    d["abs_eta"] = d[eta_col].abs()
    d = d[d["abs_eta"] <= eta_max]

    edges = np.arange(0.0, eta_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = pd.cut(d["abs_eta"], bins=edges, include_lowest=True, labels=False)
    d = d.assign(bin_idx=bin_idx).dropna(subset=["bin_idx"])

    grouped = d.groupby("bin_idx", observed=True).agg(
        **{
            t_col: (t_col, "mean"),
            f_col: (f_col, "mean"),
        }
    )
    grouped[eta_col] = centers[grouped.index.astype(int)]
    return grouped.reset_index(drop=True).sort_values(eta_col).reset_index(drop=True)


def plot_material_budget_txt(
    path: str,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    savepath: Optional[PathLike] = None,
    xlim: Tuple[float, float] = (-4.0, 4.0),
    ms: float = 3.0,
    grid_alpha: float = 0.3,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot t/X0 vs eta from a two-column txt file."""
    df = read_two_column_txt(path, ("eta", "t_over_X0")).sort_values("eta").reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    ax.plot(df["eta"], df["t_over_X0"], "k.", ms=ms)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$t/X_0$")
    ax.set_xlim(*xlim)
    ylim = ax.get_ylim()
    ax.set_ylim(0.0, max(ylim[1], float(df["t_over_X0"].max())))
    ax.grid(True, alpha=grid_alpha)
    if title is not None:
        ax.set_title(title)
    elif not ax.get_title():
        ax.set_title(f"Material budget — {_basename_no_ext(path)}")
    _save_png(fig, savepath if savepath is not None else None)
    return fig, ax, df


def plot_f_conv_from_material_budget_txt(
    path: str,
    *,
    eta_mode: EtaMode = "full",
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    savepath: Optional[PathLike] = None,
    step: bool = True,
    lw: float = 1.2,
    color: str = "b",
    grid_alpha: float = 0.3,
    abs_eta_bin_width: float = 0.1,
    eta_max: float = 2.5,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Compute BH conversion probability from (eta, t/X0) txt and plot vs eta.

    - eta_mode="full": plot signed eta range as-is
    - eta_mode="positive": restrict to 0 <= eta <= eta_max
    - eta_mode="abs": fold to |eta| and re-bin with width abs_eta_bin_width
    """
    df = read_two_column_txt(path, ("eta", "t_over_X0"))
    df = add_f_conv_true(df)

    if eta_mode == "positive":
        df = df[(df["eta"] >= 0.0) & (df["eta"] <= eta_max)].reset_index(drop=True)
    elif eta_mode == "abs":
        df = fold_abs_eta(
            df,
            eta_max=eta_max,
            bin_width=abs_eta_bin_width,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    default_label = _with_phase(
        r"$f^\mathrm{true}_\mathrm{conv} = 1 - e^{-(7/9)\,t/X_0}$ (BH)",
        path,
    )
    curve_label = default_label if label is None else label

    if step:
        ax.step(
            df["eta"],
            df["f_conv_true"],
            where="mid",
            color=color,
            lw=lw,
            label=curve_label,
        )
    else:
        ax.plot(
            df["eta"],
            df["f_conv_true"],
            ".",
            color=color,
            ms=3,
            label=curve_label,
        )

    ax.set_ylim(0.0, 1.0)
    if eta_mode == "abs":
        ax.set_xlim(0.0, eta_max)
        ax.set_xlabel(r"$|\eta|$")
    else:
        ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Conversion probability")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    elif not ax.get_title():
        ax.set_title(
            f"Photon conversion probability (calculated via Bethe–Heitler from t/X0) — {_basename_no_ext(path)}"
        )
    _save_png(fig, savepath if savepath is not None else None)
    return fig, ax, df


def plot_f_conv_direct_txt(
    path: str,
    *,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    savepath: Optional[PathLike] = None,
    step: bool = True,
    lw: float = 1.2,
    color: str = "b",
    grid_alpha: float = 0.3,
    ylim: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot conversion probability vs eta from a two-column (eta, f_conv) txt."""
    df = read_two_column_txt(path, ("eta", "f_conv")).sort_values("eta").reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    default_label = _with_phase("direct from txt", path)
    curve_label = default_label if label is None else label

    if step:
        ax.step(df["eta"], df["f_conv"], where="mid", color=color, lw=lw, label=curve_label)
    else:
        ax.plot(df["eta"], df["f_conv"], ".", color=color, ms=3, label=curve_label)

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Conversion probability")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=grid_alpha)
    if title is not None:
        ax.set_title(title)
    elif not ax.get_title():
        ax.set_title(f"Photon conversion probability (plotted directly from txt) — {_basename_no_ext(path)}")
    ax.legend()
    _save_png(fig, savepath if savepath is not None else None)
    return fig, ax, df


def plot_material_and_fconv_txt(
    path: str,
    *,
    include_total: bool = False,
    ax_left: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    savepath: Optional[PathLike] = None,
    ms: float = 3.0,
    grid_alpha: float = 0.3,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes, pd.DataFrame]:
    """Twin-axis plot: t/X0 (left) and f_conv (right) vs eta from (eta, t/X0) txt."""
    df = read_two_column_txt(path, ("eta", "t_over_X0"))
    df = df.sort_values("eta").reset_index(drop=True)
    df = add_f_conv_true(df)

    if ax_left is None:
        fig, ax_left = plt.subplots(figsize=(10, 5.5))
    else:
        fig = ax_left.figure
    ax_right = ax_left.twinx()

    ax_left.plot(df["eta"], df["t_over_X0"], "k.", ms=ms, label=r"$t/X_0$")
    ax_right.plot(
        df["eta"],
        df["f_conv_true"],
        "b.",
        ms=ms,
        label=r"$f^\mathrm{true}_\mathrm{conv} = 1 - e^{-(7/9)\,t/X_0}$",
    )

    if include_total:
        df["f_conv_total"] = df["f_conv_true"] * 0.70 + (1.0 - df["f_conv_true"]) * 0.05
        ax_right.plot(df["eta"], df["f_conv_total"], "r.", ms=ms, label=r"$f^\mathrm{total}_\mathrm{conv}$")

    ax_left.set_xlabel(r"$\eta$")
    ax_left.set_ylabel(r"$t/X_0$")
    ax_right.set_ylabel("Conversion probability")
    ax_left.set_xlim(-4.0, 4.0)
    ax_left.set_ylim(0.0, max(ax_left.get_ylim()[1], float(df["t_over_X0"].max())))
    ax_right.set_ylim(0.0, 1.0)

    ax_left.grid(True, alpha=grid_alpha)

    # Combined legend
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper center")

    if title is not None:
        ax_left.set_title(title)
    elif not ax_left.get_title():
        ax_left.set_title(f"Material budget & photon conversion — {_basename_no_ext(path)}")
    _save_png(fig, savepath if savepath is not None else None)
    return fig, ax_left, ax_right, df


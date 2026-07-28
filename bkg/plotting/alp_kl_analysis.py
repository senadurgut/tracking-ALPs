#!/usr/bin/env python
"""
alp_kl_analysis.py
==================

One script to replace the tangle of plotting notebooks (bkg / displacement /
signal_plots / hard_sample_plots).  It:

  * loads each dataset ONCE, lazily, and caches it
        - hard K_L sample   (HardQCD pThat>80, VBF-triggered)   kl_vbf_sample.csv
        - soft K_L sample   (SoftQCD inclusive)                 kl_sample.csv
        - ALP MadGraph samples                                  data/cmsrun3-csvs/
  * builds every observable the notebooks plot (pT spectra, VBF acceptance,
    reconstructed displacement b~, survival curves, S/B, ...)
  * fills histograms for the number of events at Run 3 (312 fb^-1) and
    Phase 2 (3000 fb^-1) via per-sample physical weights
  * lets you pick which plot to make on the command line

Examples
--------
    python alp_kl_analysis.py --list
    python alp_kl_analysis.py sig_vs_bkg_pt --vbf --lumi both
    python alp_kl_analysis.py displacement --selection vbf --lumi run3
    python alp_kl_analysis.py pt_cut_efficiency
    python alp_kl_analysis.py alp_yield_vs_gagg --ma 0.5
    python alp_kl_analysis.py all --no-show          # regenerate every plot

Notes
-----
Heavy inputs (multi-GB CSVs) are read only when a plot needs them, and the
per-kaon pT arrays / reconstructed-displacement arrays are cached to .npy so
repeated runs are fast.  Use --rebuild to force recomputation.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

# ---------------------------------------------------------------------------
# paths / imports of the analysis package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/export/sdurgut/scratch/alps/tracking_ALPs")
BKG_DIR      = Path(__file__).resolve().parent.parent   # tracking_ALPs/bkg

for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "analysis"), str(BKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from module_VBF import (                      # noqa: E402
    TRT_length, displaced_vertex_TRT, conv_prob_novec,
    read_data, raw_to_events,
)
from scripts.analysis_helpers.file_helpers import ma_to_name          # noqa: E402
from scripts.analysis_helpers.kinematics_helpers import compute_mjj   # noqa: E402

# ---------------------------------------------------------------------------
# configuration  (single source of truth -- was scattered across notebooks)
# ---------------------------------------------------------------------------
HARD_CSV   = BKG_DIR / "data" / "kl_vbf_sample.csv"
SOFT_CSV   = BKG_DIR / "data" / "kl_sample.csv"
ALP_CSVDIR = PROJECT_ROOT / "data" / "cmsrun3-csvs"
CACHE_DIR  = BKG_DIR / "data"                # .npy caches now live in data/
PLOT_DIR   = BKG_DIR / "plots"

# physics constants
BR_KL_GG  = 5.47e-4          # Br(K_L -> gamma gamma)
M_KL      = 0.497611         # GeV
CTAU_KL   = 15.34            # m
TRACK_RES = 4e-5             # m, tracker spatial resolution

# sample normalisations
SIGMA_HARD_PB = 3.6413e-3 * 1e9   # HardQCD pThat>80 xsec, mb -> pb
N_HARD        = 20_000_000        # hard events generated (400 x 50k)
SIGMA_INEL_PB = 78.585e9          # inelastic xsec, mb -> pb (SoftQCD)
N_SOFT        = 2_000_000         # soft events generated (40 x 50k)

ALP_XSEC_1E2_PB = 180.0           # ALP VBF xsec [pb] at gagg = 1e-2 GeV^-1
ALP_GAGG_REF    = 1e-2            # reference coupling for the scaling
N_ALP_GEN       = 40_000          # events per ALP MadGraph sample

# lumi points
LUMI = {"run3": 312.0, "phase2": 3000.0}     # fb^-1

# VBF trigger selection
VBF_LEAD_PT, VBF_SUB_PT, VBF_MJJ, VBF_DETA = 105.0, 40.0, 720.0, 3.0

# defaults
DEFAULT_MASSES = [0.01, 0.10, 1.00, 10.00]           # GeV
DEFAULT_GAGGS  = [1e-4, 1e-3, 1e-2]                  # GeV^-1
DEFAULT_ERA    = "run3"

PUB_STYLE = {
    "font.size": 13, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
    "ytick.minor.visible": True, "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# ---------------------------------------------------------------------------
# weights:  cross-section-per-row  ->  physical N(events) at a given lumi
# ---------------------------------------------------------------------------
def w_hard_kaon(lumi_fb):
    """Physical K_L->gg events represented by one row of the hard sample."""
    return (SIGMA_HARD_PB / N_HARD) * (lumi_fb * 1e3) * BR_KL_GG

def w_soft_kaon(lumi_fb):
    """Physical K_L->gg events represented by one row of the soft sample."""
    return (SIGMA_INEL_PB / N_SOFT) * (lumi_fb * 1e3) * BR_KL_GG

def alp_xsec_pb(gagg):
    return ALP_XSEC_1E2_PB * (gagg / ALP_GAGG_REF) ** 2

def w_alp(gagg, lumi_fb, n_gen=N_ALP_GEN):
    """Physical ALP events represented by one generated ALP event."""
    return (lumi_fb * 1e3) * alp_xsec_pb(gagg) / n_gen


# ===========================================================================
# DATA LOADERS  --  each dataset is read once and memoised in _CACHE
# ===========================================================================
_CACHE = {}

def _memo(key, builder):
    if key not in _CACHE:
        _CACHE[key] = builder()
    return _CACHE[key]


def get_hard_kl_pt(rebuild=False):
    """pT of every K_L in the hard (VBF) sample."""
    def build():
        npy = CACHE_DIR / "hard_kl_sample_pt.npy"
        if npy.exists() and not rebuild:
            return np.load(npy)
        import pandas as pd
        pt = pd.read_csv(HARD_CSV, usecols=["kl_pt"])["kl_pt"].to_numpy()
        np.save(npy, pt)
        return pt
    return _memo("hard_kl_pt", build)


def get_soft_kl_pt(rebuild=False):
    """pT of every K_L in the soft (inclusive) sample."""
    def build():
        npy = CACHE_DIR / "soft_kl_sample_pt.npy"
        if npy.exists() and not rebuild:
            return np.load(npy)
        import pandas as pd
        pt = pd.read_csv(SOFT_CSV, usecols=["pt"], engine="python")["pt"].to_numpy()
        np.save(npy, pt)
        return pt
    return _memo("soft_kl_pt", build)


def get_hard_kl_full():
    """Full hard-sample columns needed for the displacement calculation."""
    def build():
        import pandas as pd
        return pd.read_csv(
            HARD_CSV,
            usecols=["kl_pt", "kl_eta", "kl_phi",
                     "g1_eta", "g1_phi", "g2_eta", "g2_phi"],
        )
    return _memo("hard_kl_full", build)


def get_soft_kl_full(nmax=None):
    """Full soft-sample columns needed for the displacement calculation."""
    def build():
        import pandas as pd
        df = pd.read_csv(SOFT_CSV, usecols=["pt", "eta", "phi", "E"],
                         engine="python", nrows=nmax)
        return df
    key = f"soft_kl_full_{nmax}"
    return _memo(key, build)


def get_alp_events(ma, gaggs, era=DEFAULT_ERA, n_gen=N_ALP_GEN):
    """Processed ALP events (kinematics + per-gagg conv/l_track/decay length)."""
    gaggs = tuple(round(float(g), 15) for g in gaggs)
    key = ("alp", round(float(ma), 6), gaggs, era, n_gen)

    def build():
        raw = read_data(ma_to_name(ma), num=n_gen, data_dir=str(ALP_CSVDIR))
        return raw_to_events(raw, gaggs=list(gaggs), ma=ma, era=era)
    return _memo(key, build)


# ===========================================================================
# OBSERVABLE BUILDERS
# ===========================================================================
def passes_vbf(ev):
    lead = max(ev["j1"]["pt"], ev["j2"]["pt"])
    sub  = min(ev["j1"]["pt"], ev["j2"]["pt"])
    return (lead > VBF_LEAD_PT and sub > VBF_SUB_PT
            and compute_mjj(ev) > VBF_MJJ
            and abs(ev["j1"]["eta"] - ev["j2"]["eta"]) > VBF_DETA)


def alp_vbf_events(ma, era=DEFAULT_ERA, gaggs=(1e-4,), n_gen=N_ALP_GEN):
    """(all events, vbf-passing events, acceptance) for one ALP mass."""
    events = get_alp_events(ma, gaggs, era, n_gen)
    vbf = [ev for ev in events if passes_vbf(ev)]
    return events, vbf, len(vbf) / len(events)


def alp_pt(events):
    return np.array([ev["a"]["pt"] for ev in events])


def survival_fraction(pt, cuts):
    """Fraction of `pt` above each value in `cuts` (keep pt > cut)."""
    s = np.sort(np.asarray(pt))
    return (len(s) - np.searchsorted(s, cuts, side="right")) / len(s)


# ---- reconstructed displacement b~ -----------------------------------------
def build_b_kl_hard(seed=1234, rebuild=False):
    """b~ for K_L->gg in the hard (VBF) sample. Photons taken from the sample."""
    npy = CACHE_DIR / "b_kl_hard.npy"
    if npy.exists() and not rebuild:
        return np.load(npy)

    hard = get_hard_kl_full()
    rng = np.random.default_rng(seed)
    b_kl, n_intracker, n_2conv = [], 0, 0
    for r in hard.itertuples():
        L = TRT_length(r.kl_eta)
        if L <= 0:
            continue
        ell = (r.kl_pt * np.cosh(r.kl_eta) / M_KL) * CTAU_KL
        l_a = -ell * np.log(rng.uniform())
        if l_a >= L:
            continue
        n_intracker += 1
        l_tracks = []
        for eta_g in (r.g1_eta, r.g2_eta):
            P = conv_prob_novec(abs(eta_g)) * (1.0 - l_a / L)
            if rng.uniform() > P:
                break
            x = rng.uniform(l_a, L)
            l_tracks.append(L - x)
        if len(l_tracks) < 2:
            continue
        n_2conv += 1
        b_kl.append(displaced_vertex_TRT(
            r.kl_eta, r.g1_phi, r.g2_phi, r.kl_phi,
            l_tracks[0], l_tracks[1], l_a, TRACK_RES))
    b_kl = np.array(b_kl)
    print(f"[b_kl hard] in-tracker={n_intracker:,}  2-conv={n_2conv:,}  "
          f"entries={len(b_kl):,}")
    np.save(npy, b_kl)
    return b_kl


def _decay_photons(pt_k, eta_k, phi_k, E_k, rng):
    """Isotropic K_L->gg boosted along the true kaon direction -> (eta,phi) x2."""
    pvec = np.array([pt_k * np.cos(phi_k), pt_k * np.sin(phi_k),
                     pt_k * np.sinh(eta_k)])
    beta = pvec / E_k
    b2 = beta @ beta
    gam = E_k / M_KL
    Es = M_KL / 2
    cth = rng.uniform(-1, 1)
    sth = np.sqrt(1 - cth * cth)
    ph = rng.uniform(0, 2 * np.pi)
    ps = Es * np.array([sth * np.cos(ph), sth * np.sin(ph), cth])
    out = []
    for s in (ps, -ps):
        fac = ((gam - 1) * (beta @ s) / b2 + gam * Es) if b2 > 0 else gam * Es
        pl = s + fac * beta
        ptl = np.hypot(pl[0], pl[1])
        out.append((np.arcsinh(pl[2] / ptl), np.arctan2(pl[1], pl[0])))
    return out


def build_b_kl_soft(seed=None, nmax=None, rebuild=False):
    """b~ for K_L->gg in the soft (inclusive) sample. Photons are sampled."""
    npy = CACHE_DIR / "b_kl_soft.npy"
    if npy.exists() and not rebuild and nmax is None:
        return np.load(npy)

    df = get_soft_kl_full(nmax)
    pt, eta, phi, E = (df[c].to_numpy() for c in ("pt", "eta", "phi", "E"))
    rng = np.random.default_rng(seed)
    p = pt * np.cosh(eta)
    ell = (p / M_KL) * CTAU_KL
    L = TRT_length(eta)
    l_a = -ell * np.log(rng.uniform(size=len(p)))
    ok = (L > 0) & (l_a < L)
    print(f"[b_kl soft] in tracker: {ok.sum():,} / {len(p):,}")

    b = []
    for i in np.where(ok)[0]:
        (e1, f1), (e2, f2) = _decay_photons(pt[i], eta[i], phi[i], E[i], rng)
        ls = []
        for eg in (e1, e2):
            P = conv_prob_novec(abs(eg)) * (1.0 - l_a[i] / L[i])
            if rng.uniform() > P:
                break
            x = rng.uniform(l_a[i], L[i])
            ls.append(L[i] - x)
        if len(ls) < 2:
            continue
        b.append(displaced_vertex_TRT(eta[i], f1, f2, phi[i],
                                      ls[0], ls[1], l_a[i], TRACK_RES))
    b = np.array(b)
    print(f"[b_kl soft] b~ entries: {len(b):,}")
    if nmax is None:
        np.save(npy, b)
    return b


def build_b_alp(ma, gagg, era=DEFAULT_ERA, vbf=False, n_gen=N_ALP_GEN):
    """b~ for ALP->gg at one (ma, gagg); optionally VBF-selected."""
    events = get_alp_events(ma, (gagg,), era, n_gen)
    if vbf:
        events = [ev for ev in events if passes_vbf(ev)]
    out = []
    for ev in events:
        if not (ev["g1"]["conv"][0] and ev["g2"]["conv"][0]):
            continue
        out.append(displaced_vertex_TRT(
            ev["a"]["eta"], ev["g1"]["phi"], ev["g2"]["phi"], ev["a"]["phi"],
            ev["g1"]["l_track"][0], ev["g2"]["l_track"][0],
            ev["a"]["l"][0], TRACK_RES))
    return np.array(out)


# ===========================================================================
# PLOTS
# ===========================================================================
def _lumi_list(args):
    if args.lumi == "both":
        return ["run3", "phase2"]
    return [args.lumi]


def _finish(fig, args, stem):
    if not args.no_save:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOT_DIR / f"{stem}.{args.format}"
        fig.savefig(path)
        print(f"saved  {path}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


# --- K_L pT spectra ---------------------------------------------------------
def plot_kl_pt(args, which):
    import matplotlib.pyplot as plt
    if which == "hard":
        pt, w_row, title = get_hard_kl_pt(), None, r"$K_L$ (VBF-triggered)"
        binlo = -2
    else:
        pt, w_row, title = get_soft_kl_pt(), None, r"$K_L$ (inclusive)"
        binlo = -2
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bins = np.logspace(binlo, 3 if which == "hard" else 2, 60)
    # dsigma/dpT [pb/bin] -- lumi-independent cross section
    w_pb = (SIGMA_HARD_PB / N_HARD) if which == "hard" else (SIGMA_INEL_PB / N_SOFT)
    ax.hist(pt, bins=bins, weights=np.full(len(pt), w_pb),
            histtype="stepfilled", color="#3b6fb0", edgecolor="#1f3b5c",
            alpha=0.85, lw=1.2, label=title)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$K_L$ transverse momentum $p_T$ [GeV]")
    ax.set_ylabel(r"$d\sigma/dp_T$ [pb / bin]")
    ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    _finish(fig, args, f"kl_pt_{which}")


def plot_kl_pt_hard(args): plot_kl_pt(args, "hard")
def plot_kl_pt_soft(args): plot_kl_pt(args, "soft")


# --- K_L survival vs pT cut -------------------------------------------------
def plot_kl_survival(args, which):
    import matplotlib.pyplot as plt
    if which == "hard":
        pt, w_pb, cmax = get_hard_kl_pt(), SIGMA_HARD_PB / N_HARD, 150
    else:
        pt, w_pb, cmax = get_soft_kl_pt(), SIGMA_INEL_PB / N_SOFT, 100
    cuts = np.linspace(0, cmax, 201)
    frac = survival_fraction(pt, cuts)
    sigma_above = frac * len(pt) * w_pb

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(cuts, sigma_above, color="#b0403b", lw=2.0)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$p_T$ cut [GeV]  (keep $p_T >$ cut)")
    ax[0].set_ylabel(r"surviving $\sigma$ [pb]")
    ax[0].set_title(rf"$K_L$ cross section vs $p_T$ cut ({which})")
    ax[1].plot(cuts, frac, color="#3b6fb0", lw=2.0)
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$p_T$ cut [GeV]")
    ax[1].set_ylabel("surviving fraction")
    ax[1].set_title(r"survival fraction vs $p_T$ cut")
    for a in ax:
        a.set_xlim(cuts.min(), cuts.max())
        a.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
    fig.tight_layout()
    _finish(fig, args, f"kl_survival_{which}")

    # table
    print(f"{'pT cut':>8} {'frac':>12} {'sigma[pb]':>14}")
    for c in [0, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70]:
        n = (pt > c).sum()
        print(f"{c:>8} {n/len(pt):>12.3e} {n*w_pb:>14.3e}")


def plot_kl_survival_hard(args): plot_kl_survival(args, "hard")
def plot_kl_survival_soft(args): plot_kl_survival(args, "soft")


# --- ALP pT distributions (before/after VBF) --------------------------------
def plot_alp_pt(args):
    import matplotlib.pyplot as plt
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.masses)))
    for stage in (["incl", "vbf"] if args.both_stages else [args.stage]):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for ma, c in zip(args.masses, colors):
            events, vbf, acc = alp_vbf_events(ma, args.era)
            sel = vbf if stage == "vbf" else events
            ax.hist(alp_pt(sel), bins=np.linspace(0, 500, 60),
                    histtype="step", lw=2, color=c,
                    label=fr"$m_a$ = {ma:g} GeV")
        ax.set_yscale("log")
        ax.set_xlabel(r"ALP $p_T$ [GeV]"); ax.set_ylabel("Events")
        ax.set_title(fr"ALP $p_T$ ({'after VBF' if stage=='vbf' else 'inclusive'})")
        ax.legend(fontsize=10)
        fig.tight_layout()
        _finish(fig, args, f"alp_pt_{stage}")


# --- ALP VBF acceptance vs mass ---------------------------------------------
def plot_alp_acceptance(args):
    import matplotlib.pyplot as plt
    ms = sorted(args.masses)
    accs = [alp_vbf_events(ma, args.era)[2] for ma in ms]
    for ma, a in zip(ms, accs):
        print(f"m_a = {ma:>6g} GeV : VBF acceptance = {a:.4f}")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ms, accs, "o-", color="#3b6fb0", label="ALP (VBF)")
    ax.set_xscale("log")
    ax.set_xlabel(r"ALP mass [GeV]"); ax.set_ylabel("VBF acceptance")
    ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    _finish(fig, args, "alp_vbf_acceptance")


# --- signal vs background pT overlay (N events @ lumi) ----------------------
def plot_sig_vs_bkg_pt(args):
    import matplotlib.pyplot as plt
    stage = "vbf" if args.vbf else "incl"
    if args.vbf:
        bkg_pt, w_row_fn, bkg_lbl = get_hard_kl_pt(), w_hard_kaon, r"$K_L\to\gamma\gamma$ (VBF)"
    else:
        bkg_pt, w_row_fn, bkg_lbl = get_soft_kl_pt(), w_soft_kaon, r"$K_L\to\gamma\gamma$ (inclusive)"

    # signal arrays + per-event weight base (gagg reference)
    sig = []
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.masses)))
    for ma, c in zip(args.masses, colors):
        events, vbf, acc = alp_vbf_events(ma, args.era)
        sel = vbf if args.vbf else events
        sig.append((ma, alp_pt(sel), len(events), c))

    for lu in _lumi_list(args):
        L = LUMI[lu]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bins = np.linspace(0, 300, 60)
        w_bkg = np.full(len(bkg_pt), w_row_fn(L))
        ax.hist(bkg_pt, bins=bins, weights=w_bkg, histtype="step",
                lw=2.2, color="black", ls="--", label=bkg_lbl)
        for ma, pt, n_gen, c in sig:
            # N = (surviving count / n_gen) * sigma(gagg) * L  -- per-event weight
            # is w_alp; VBF vs inclusive differs only through the surviving count.
            ax.hist(pt, bins=bins, weights=np.full(len(pt), w_alp(args.gagg, L, n_gen)),
                    histtype="step", lw=2.0, color=c,
                    label=fr"ALP, $m={ma:g}$ GeV")
        ax.set_yscale("log")
        ax.set_xlabel(r"Transverse momentum $p_T$ [GeV]")
        ax.set_ylabel(fr"Events @ {L:g} fb$^{{-1}}$ / bin")
        ax.set_xlim(bins.min(), bins.max())
        ax.text(0.03, 0.95,
                f"{'VBF selection' if args.vbf else 'no VBF selection'}\n"
                fr"$g_{{a\gamma\gamma}}={args.gagg:.0e}\,\mathrm{{GeV}}^{{-1}}$",
                transform=ax.transAxes, fontsize=10.5, va="top")
        ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
        ax.legend(fontsize=9)
        fig.tight_layout()
        _finish(fig, args, f"sig_vs_bkg_pt_{stage}_{lu}")


# --- pT-cut efficiency (surviving fraction) ---------------------------------
def plot_pt_cut_efficiency(args):
    import matplotlib.pyplot as plt
    cuts = np.linspace(0, 120, 400)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(cuts, survival_fraction(get_hard_kl_pt(), cuts), lw=2.2,
            color="black", label=r"$K_L$, VBF-triggered")
    ax.plot(cuts, survival_fraction(get_soft_kl_pt(), cuts), lw=2.2,
            color="black", ls="--", label=r"$K_L$, inclusive")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.masses)))
    for ma, c in zip(args.masses, colors):
        _, vbf, _ = alp_vbf_events(ma, args.era)
        ax.plot(cuts, survival_fraction(alp_pt(vbf), cuts), lw=2.0,
                color=c, label=fr"ALP, $m={ma:g}$ GeV")
    ax.axvline(60, color="red", ls=":", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_T$ cut [GeV]  (keep $p_T >$ cut)")
    ax.set_ylabel("surviving fraction")
    ax.set_xlim(0, 120); ax.set_ylim(1e-7, 2)
    ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    _finish(fig, args, "pt_cut_efficiency")


# --- reconstructed displacement b~ ------------------------------------------
def plot_displacement(args):
    import matplotlib.pyplot as plt
    vbf = args.selection == "vbf"
    if vbf:
        b_kl, w_row_fn, kl_lbl = build_b_kl_hard(rebuild=args.rebuild), w_hard_kaon, r"$K_L\to\gamma\gamma$ (VBF)"
    else:
        b_kl, w_row_fn, kl_lbl = build_b_kl_soft(rebuild=args.rebuild), w_soft_kaon, r"$K_L\to\gamma\gamma$ (inclusive)"

    b_alp = {g: build_b_alp(args.ma, g, args.era, vbf) for g in args.gaggs}

    for lu in _lumi_list(args):
        L = LUMI[lu]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bins = np.logspace(-5, 0, 50)
        pos = b_kl > 0
        ax.hist(b_kl[pos], bins=bins, weights=np.full(pos.sum(), w_row_fn(L)),
                histtype="step", lw=2.2, color="black", ls="--", label=kl_lbl)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.gaggs)))
        for g, c in zip(args.gaggs, colors):
            b = b_alp[g]; m = b > 0
            if m.sum() == 0:
                continue
            ax.hist(b[m], bins=bins, weights=np.full(m.sum(), w_alp(g, L)),
                    histtype="step", lw=2.0, color=c,
                    label=fr"ALP, $g={g:.0e}$")
        ax.axvline(1e-2, color="red", ls="--", lw=1.2, label=r"$\tilde b=1$ cm")
        ax.axvline(1e-1, color="gray", ls=":", lw=1.2, label=r"$\tilde b=10$ cm")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"reconstructed displacement $\tilde b$ [m]")
        ax.set_ylabel(fr"events @ {L:g} fb$^{{-1}}$ / bin")
        ax.text(0.03, 0.95,
                f"{'VBF selection' if vbf else 'no trigger selection'}, "
                fr"$m_a={args.ma:g}$ GeV",
                transform=ax.transAxes, fontsize=10, va="top")
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout()
        _finish(fig, args, f"displacement_{args.selection}_{lu}")


# --- displacement efficiency (surviving fraction vs b~ cut) -----------------
def plot_displacement_efficiency(args):
    import matplotlib.pyplot as plt
    vbf = args.selection == "vbf"
    b_kl = (build_b_kl_hard(rebuild=args.rebuild) if vbf
            else build_b_kl_soft(rebuild=args.rebuild))
    b_alp = {g: build_b_alp(args.ma, g, args.era, vbf) for g in args.gaggs}

    def eff(b, cuts):
        b = np.asarray(b)
        return np.array([(b > c).mean() for c in cuts])

    cuts = np.logspace(-2, -1, 200)          # 1 cm -> 10 cm
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(cuts, eff(b_kl, cuts), lw=2.2, color="black", ls="--",
            label=r"$K_L$" + (" (VBF)" if vbf else " (inclusive)"))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.gaggs)))
    for g, c in zip(args.gaggs, colors):
        if (np.asarray(b_alp[g]) > 0).sum() == 0:
            continue
        ax.plot(cuts, eff(b_alp[g], cuts), lw=2.0, color=c,
                label=fr"ALP, $g={g:.0e}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\tilde b_{\min}$ [m]  (keep $\tilde b > \tilde b_{\min}$)")
    ax.set_ylabel("surviving fraction")
    ax.set_xlim(1e-2, 1e-1)
    ax.text(0.03, 0.95, fr"$m_a={args.ma:g}$ GeV", transform=ax.transAxes,
            fontsize=10, va="top")
    ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    _finish(fig, args, f"displacement_efficiency_{args.selection}")


# --- ALP yield & S/sqrt(B) vs gagg ------------------------------------------
def plot_alp_yield_vs_gagg(args):
    import matplotlib.pyplot as plt
    _, vbf, acc = alp_vbf_events(args.ma, args.era)
    n_bkg = get_hard_kl_pt()           # VBF background reference
    gaggs = np.logspace(-7, -2, 40) if args.gagg_scan is None else np.asarray(args.gagg_scan)

    for lu in _lumi_list(args):
        L = LUMI[lu]
        N_bkg = len(n_bkg) * w_hard_kaon(L)
        N_sig = acc * (L * 1e3) * alp_xsec_pb(gaggs)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(gaggs, N_sig, "o-", color="#3b6fb0", label="VBF ALP->gg")
        ax[0].axhline(N_bkg, color="red", ls="--", label=r"VBF $K_L\to\gamma\gamma$")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel(r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]")
        ax[0].set_ylabel(fr"VBF yield @ {L:g} fb$^{{-1}}$")
        ax[0].legend(fontsize=9)
        ax[1].plot(gaggs, N_sig / np.sqrt(N_bkg), "o-", color="k")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel(r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]")
        ax[1].set_ylabel(r"$S/\sqrt{B}$")
        for a in ax:
            a.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
        fig.suptitle(fr"$m_a={args.ma:g}$ GeV")
        fig.tight_layout()
        _finish(fig, args, f"alp_yield_vs_gagg_{lu}")


# ===========================================================================
# registry + CLI
# ===========================================================================
PLOTS = {
    "kl_pt_hard":               (plot_kl_pt_hard,            "K_L pT spectrum, hard/VBF sample [pb/bin]"),
    "kl_pt_soft":               (plot_kl_pt_soft,            "K_L pT spectrum, soft/inclusive sample [pb/bin]"),
    "kl_survival_hard":         (plot_kl_survival_hard,      "K_L surviving sigma & fraction vs pT cut (hard)"),
    "kl_survival_soft":         (plot_kl_survival_soft,      "K_L surviving sigma & fraction vs pT cut (soft)"),
    "alp_pt":                   (plot_alp_pt,                "ALP pT distributions per mass (incl or VBF)"),
    "alp_acceptance":           (plot_alp_acceptance,        "ALP VBF acceptance vs mass"),
    "sig_vs_bkg_pt":            (plot_sig_vs_bkg_pt,         "signal vs K_L pT overlay, N events @ lumi"),
    "pt_cut_efficiency":        (plot_pt_cut_efficiency,     "surviving fraction vs pT cut, signal + both bkgs"),
    "displacement":             (plot_displacement,          "reconstructed displacement b~ dist, K_L + ALP"),
    "displacement_efficiency":  (plot_displacement_efficiency, "surviving fraction vs b~ cut"),
    "alp_yield_vs_gagg":        (plot_alp_yield_vs_gagg,     "ALP VBF yield & S/sqrt(B) vs gagg"),
}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plot", nargs="?", default=None,
                   help="which plot to make (or 'all', or omit with --list)")
    p.add_argument("--list", action="store_true", help="list available plots and exit")

    # lumi / physics selection
    p.add_argument("--lumi", choices=["run3", "phase2", "both"], default="both",
                   help="luminosity point(s) for event-count histograms")
    p.add_argument("--era", choices=["run3", "phase2"], default=DEFAULT_ERA,
                   help="detector geometry / material budget era for ALP samples")
    p.add_argument("--masses", type=float, nargs="+", default=DEFAULT_MASSES,
                   help="ALP masses [GeV] for multi-mass plots")
    p.add_argument("--ma", type=float, default=0.10,
                   help="single ALP mass [GeV] for displacement / yield plots")
    p.add_argument("--gaggs", type=float, nargs="+", default=DEFAULT_GAGGS,
                   help="gagg benchmarks [GeV^-1] for displacement plots")
    p.add_argument("--gagg", type=float, default=1e-4,
                   help="single reference gagg for signal-vs-bkg overlays")
    p.add_argument("--gagg-scan", type=float, nargs="+", default=None,
                   help="explicit gagg grid for alp_yield_vs_gagg (default: logspace)")

    # per-plot options
    p.add_argument("--vbf", action="store_true",
                   help="sig_vs_bkg_pt: use VBF-selected signal + hard bkg")
    p.add_argument("--stage", choices=["incl", "vbf"], default="incl",
                   help="alp_pt: which selection stage to plot")
    p.add_argument("--both-stages", action="store_true",
                   help="alp_pt: make both incl and vbf figures")
    p.add_argument("--selection", choices=["incl", "vbf"], default="incl",
                   help="displacement*: inclusive (soft) or VBF (hard) background")

    # output / behaviour
    p.add_argument("--outdir", default=str(PLOT_DIR), help="directory for saved figures")
    p.add_argument("--format", default="pdf", help="figure format (pdf/png/...)")
    p.add_argument("--no-save", action="store_true", help="do not write figures to disk")
    p.add_argument("--no-show", action="store_true", help="do not open interactive windows")
    p.add_argument("--rebuild", action="store_true",
                   help="ignore .npy caches and recompute (slow)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    global PLOT_DIR
    PLOT_DIR = Path(args.outdir)

    if args.list or args.plot is None:
        print("Available plots:\n")
        for name, (_, desc) in PLOTS.items():
            print(f"  {name:<26} {desc}")
        print("\nSpecial: 'all' runs every plot.")
        return

    import matplotlib.pyplot as plt
    if args.no_show:
        matplotlib.use("Agg")
    plt.rcParams.update(PUB_STYLE)

    if args.plot == "all":
        args.no_show = True
        for name, (fn, _) in PLOTS.items():
            print(f"\n=== {name} ===")
            try:
                fn(args)
            except Exception as e:
                print(f"  !! {name} failed: {e}")
        return

    if args.plot not in PLOTS:
        print(f"unknown plot {args.plot!r}. use --list to see options.")
        sys.exit(1)
    PLOTS[args.plot][0](args)


if __name__ == "__main__":
    main()

"""
Signal (ALP->gg, VBF) vs background (K_L->gg): 3x3 figure. Columns are the three observables --
diphoton pT (log-x), diphoton pT (linear-x), and reconstructed displacement b~. The top row shows
fraction-per-bin (unit-area shape) and the middle row the same observables as events @ lumi / bin.
The bottom-left panel is text: the exact selection, K_L counts + yield, and signal survivors + yields.

CONFIG-DRIVEN: every cut is an independent on/off toggle with a value, set in a JSON config under
bkg/configs/ (config7-style schema, extended -- see bkg/configs/README.md). Vary cuts across configs
to produce different plots. `--analysis run3|phase2` is a shortcut for bkg/configs/<era>_nominal.json.

Selection mirrors analyze_configurable_v2.py `count_passing` (signal) and run3_parking.py /
phase2_scouting.py (K_L). Displacement follows notebooks/displacement.py: signal uses the MEAN decay
length ev['a']['l']; K_L uses the SAMPLED decay position (both into displaced_vertex_TRT); the b~
observable needs the two photons to convert. The displacement panel is a coupling scan for one
representative mass (benchmark g + 1e-3 + 1e-2, distinguished by linestyle), since b~ ~ 1/g^2.

Run (heavy -- send to SLURM or a node):
    python plot_sig_vs_bkg.py --analysis run3
    python plot_sig_vs_bkg.py --config bkg/configs/run3_novbf.json --max-kl 0
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({                            # publication style (matches the standalone pT plots)
    "font.size": 13, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

_BKG = Path(__file__).resolve().parent.parent    # tracking_ALPs/bkg (data/ lives here)
TRACKING = "/home/export/sdurgut/scratch/alps/tracking_ALPs"
sys.path.insert(0, TRACKING)
sys.path.insert(0, TRACKING + "/analysis")
sys.path.insert(0, str(_BKG / "analysis"))       # kl_common lives here
import kl_common as kl
from module_VBF import (read_data, raw_to_events, TRT_length, Delta_R,
                        separation_TRT, displaced_vertex_TRT, eta_region, hgcal_cell_size)
from scripts.analysis_helpers.file_helpers import ma_to_name
from scripts.analysis_helpers.kinematics_helpers import compute_mjj

M_KL      = 0.497611
TRACK_RES = 4e-5                     # default track resolution if a config's phase2_cuts omits it
N_GEN     = 40000
BR_KL_GG     = 5.47e-4               # Br(K_L -> gamma gamma)
SIG_XSEC_1E2 = 180.0                 # sigma_VBF(a) [pb] at gagg=1e-2 GeV^-1 (~gagg^2, Br(a->gg)=1)
SIG_MASSES_DEFAULT = [0.3, 0.5, 1.0]                             # used when a config omits "signal"
GAGG_BENCH_DEFAULT = {0.3: 8.0e-5, 0.5: 2.63e-5, 1.0: 6.4e-6}   # displaced-in-tracker benchmark per mass
DISP_MASS          = 0.5                # reconstructed-displacement panel: FIXED representative mass (GeV).
DISP_SCAN_FACTORS  = [1.0, 3.0, 10.0]   # displacement-panel couplings as MULTIPLES of the DISP_MASS benchmark g.
                                        # Relative (not absolute like 1e-3/1e-2) so the curves stay in the
                                        # b~ window across eras: absolute large g -> prompt decay (b~ below
                                        # tracker resolution) and empty curves, which hit run3 hardest since
                                        # the VBF cut leaves too few survivors to populate the b~>0 tail.
# era -> K_L sample + normalization. The sample choice is FIXED per era (run3 = VBF-triggered hard
# sample with VBF baked in; phase2 = inclusive soft sample). Config `lumi` overrides the default.
ERA_PARAMS = {
    "run3":   dict(sample="data/kl_vbf_sample.csv", iso_col="kl_iso_had_pt", hard=True,
                   kl_sigma_pb=3.6413e-3 * 1e9, kl_ngen=20e6, lumi=312,  eta_max_conv=2.5),
    "phase2": dict(sample="data/kl_sample.csv",     iso_col="iso_had_pt",    hard=False,
                   kl_sigma_pb=78.585e9,        kl_ngen=2e6,  lumi=3000, eta_max_conv=3.0),
}
CONFIG_DIR     = _BKG / "configs"                                # plot configs live here
NOMINAL_CONFIG = {"run3": "run3_nominal.json", "phase2": "phase2_nominal.json"}   # --analysis shortcuts

def _w(x):
    """Per-entry weight so a step histogram shows FRACTION of events per bin. Correct on log-x axes,
       unlike density=True (which divides by linear bin width and spikes narrow low-value bins)."""
    return np.full(len(x), 1.0 / len(x)) if len(x) else None


def gfmt(g):
    """g_agg as a LaTeX string: 10^{n} for a clean power of ten, else scientific."""
    e = int(round(np.log10(g)))
    return rf"10^{{{e}}}" if np.isclose(g, 10.0 ** e) else f"{g:.2g}"


def selection_lines(cfg, cuts):
    """One line per cut: 'ON <value>' / 'OFF' -- reflects exactly what the config applied."""
    def onoff(flag, detail=""):
        return ("ON   " + detail).rstrip() if flag else "OFF"
    lo, hi = cuts["eta_range"]
    vbf_d = (f"lead>{cuts['leading_jet_pt_cut']:.0f} sub>{cuts['sub_jet_pt_cut']:.0f} "
             f"mjj>{cuts['mjj_cut']:.0f} |dEta|>{cuts['deta_cut']:.0f}")
    if cfg["era"] == "run3":
        vbf_d += "  [+baked into K_L]"
    eta_d   = f"[{lo}, {hi}] both photons"
    merge_d = f"<= {cuts['delta_r_max']}  (cell {cuts['ecal_cell_size']}, use_tracks={cuts['use_tracks']})"
    sep_d   = f">= {cuts['sep_cut']:g} m"
    disp_d  = f">= {cuts['disp_cut']:g} m (res {cuts['track_resolution']:g})"
    act_d   = f"<= {cuts['act_cut']:g} GeV"
    pt_d    = f"> {cuts['pT_cut']:g} GeV"
    return [f"in-tracker decay : {onoff(cuts['mask_tracker_decay'])}",
            f"eta              : {onoff(cuts['mask_eta'], eta_d)}",
            f"VBF (signal)     : {onoff(cuts['mask_vbf'], vbf_d)}",
            f"merge dR         : {onoff(cuts['mask_merge'], merge_d)}",
            f"both convert     : {onoff(cuts['mask_convert'])}",
            f"separation       : {onoff(cuts['mask_separation'], sep_d)}",
            f"displacement     : {onoff(cuts['mask_displacement'], disp_d)}",
            f"hadronic activity: {onoff(cuts['mask_activity'], act_d)}",
            f"alp pT           : {onoff(cuts['mask_alp_pT'], pt_d)}"]


def load_cuts(path):
    """Resolve a plot-config JSON to the cut params used here. Mirrors analyze_configurable_v2.resolve_cuts
       for the shared fields and adds granular plot toggles (convert/separation/displacement/activity/pT).
       Backward compatible: an unextended config7/config10 reproduces the original selection because the
       convert/separation/displacement toggles default to (analysis_mode == 'scouting'). Not imported from
       analyze_configurable_v2 because that module pulls in `yaml` (absent in this env)."""
    with open(path) as f:
        c = json.load(f)
    mode  = c.get("analysis_mode", "parking")
    scout = (mode == "scouting")
    em, vm, mm, p2 = (c.get(k, {}) for k in ("mask_eta", "mask_vbf", "mask_merge", "phase2_cuts"))
    conv, sep, disp = (c.get(k, {}) for k in ("mask_convert", "mask_separation", "mask_displacement"))
    act, apt, tdc = (c.get(k, {}) for k in ("activity_cut", "alp_pT_cut", "tracker_decay_cut"))
    sig = c.get("signal", {})
    masses  = [float(x) for x in sig.get("masses", SIG_MASSES_DEFAULT)]
    gagg_in = {float(k2): float(v2) for k2, v2 in sig.get("gagg", {}).items()}
    return dict(
        era=c["era"], analysis_mode=mode, lumi=c.get("lumi"),
        mask_tracker_decay=tdc.get("value", True),
        mask_eta=em.get("value", False), eta_range=em.get("eta_range", [0.0, 0.0]),
        mask_vbf=vm.get("value", False), leading_jet_pt_cut=vm.get("lead_pt", 0.0),
        sub_jet_pt_cut=vm.get("sub_pt", 0.0), mjj_cut=vm.get("mjj", 0.0), deta_cut=vm.get("deta", 0.0),
        mask_merge=mm.get("value", True), delta_r_max=mm.get("delta_r_max", 0.3),
        ecal_cell_size=0.025, use_tracks=c.get("use_tracks", True),
        mask_convert=conv.get("value", scout), mask_separation=sep.get("value", scout),
        mask_displacement=disp.get("value", scout),
        sep_cut=p2.get("sep_cut", 5e-4), disp_cut=p2.get("disp_cut", 1e-1),
        track_resolution=p2.get("track_resolution", TRACK_RES),
        mask_activity=act.get("value", False), act_cut=act.get("cut_value", 0.0),
        mask_alp_pT=apt.get("value", False), pT_cut=apt.get("cut_value", 0.0),
        masses=masses,
        gagg={m: gagg_in.get(m, GAGG_BENCH_DEFAULT.get(m, 1e-5)) for m in masses},
    )


def merge_pass_signal(ev, cuts, i_g, dR):
    """config7/config10 merge branch, mirroring analyze_configurable_v2.py:185-224 (signal)."""
    era, mode = cuts["era"], cuts["analysis_mode"]
    both_conv = ev["g1"]["conv"][i_g] and ev["g2"]["conv"][i_g]
    if era == "run3" and mode == "parking":
        if dR < cuts["ecal_cell_size"]:
            return both_conv if cuts["use_tracks"] else False
        return dR < cuts["delta_r_max"]
    if era == "phase2" and mode == "scouting":
        return dR <= cuts["delta_r_max"]
    raise ValueError("unsupported era/mode")


def select_signal(events, cuts):
    """Config-driven selection at i_g=0; return (pt[], disp[]) for survivors. Each cut is an
       independent toggle. disp is only filled for survivors whose two photons convert (the
       observable needs tracks), regardless of whether the displacement CUT is on."""
    i_g, pt, disp = 0, [], []
    need_conv = cuts["mask_convert"] or cuts["mask_separation"] or cuts["mask_displacement"]
    tr = cuts["track_resolution"]
    for ev in events:
        if cuts["mask_tracker_decay"]:
            if -ev["a"]["l"][i_g] * np.log(np.random.uniform()) >= TRT_length(ev["a"]["eta"]):
                continue                                                    # in-tracker decay
        if cuts["mask_alp_pT"] and ev["a"]["pt"] <= cuts["pT_cut"]:
            continue                                                        # alp pT
        e1, e2 = ev["g1"]["eta"], ev["g2"]["eta"]
        if cuts["mask_eta"]:
            lo, hi = cuts["eta_range"]
            if not (lo <= abs(e1) <= hi and lo <= abs(e2) <= hi):
                continue
        if cuts["mask_vbf"]:
            lead, sub = max(ev["j1"]["pt"], ev["j2"]["pt"]), min(ev["j1"]["pt"], ev["j2"]["pt"])
            if not (lead > cuts["leading_jet_pt_cut"] and sub > cuts["sub_jet_pt_cut"]
                    and compute_mjj(ev) > cuts["mjj_cut"]
                    and abs(ev["j1"]["eta"] - ev["j2"]["eta"]) > cuts["deta_cut"]):
                continue
        l_a = ev["a"]["l"][i_g]
        if cuts["mask_merge"]:
            dR = Delta_R(e1, e2, ev["g1"]["phi"], ev["g2"]["phi"], l_a)
            if not merge_pass_signal(ev, cuts, i_g, dR):
                continue
        both_conv = ev["g1"]["conv"][i_g] and ev["g2"]["conv"][i_g]
        if need_conv and not both_conv:
            continue                                                        # both photons convert
        if cuts["mask_separation"]:
            if separation_TRT(e1, e2, ev["a"]["eta"], ev["g1"]["phi"], ev["g2"]["phi"],
                              ev["a"]["phi"], l_a) < cuts["sep_cut"]:
                continue
        if cuts["mask_displacement"]:
            if displaced_vertex_TRT(ev["a"]["eta"], ev["g1"]["phi"], ev["g2"]["phi"], ev["a"]["phi"],
                                    ev["g1"]["l_track"][i_g], ev["g2"]["l_track"][i_g], l_a, tr) < cuts["disp_cut"]:
                continue
        pt.append(ev["a"]["pt"])
        if both_conv:
            disp.append(displaced_vertex_TRT(ev["a"]["eta"], ev["g1"]["phi"], ev["g2"]["phi"],
                        ev["a"]["phi"], ev["g1"]["l_track"][i_g], ev["g2"]["l_track"][i_g], l_a, tr))
    return np.array(pt), np.array(disp)


def select_kl(cfg, cuts, rng, max_rows, tick, act_cut, chunk=250_000):
    """Stream the K_L sample in chunks (bounded memory); full selection + activity cut (act_cut = the
       threshold, None disables it). max_rows caps rows read (0 = all). Returns (pt[], disp[], n_read)."""
    cols = (["kl_pt", "kl_eta", "kl_phi", "g1_eta", "g1_phi", "g2_eta", "g2_phi", cfg["iso_col"]]
            if cfg["hard"] else ["pt", "eta", "phi", cfg["iso_col"]])
    P_conv = kl.make_pconv(cfg["era"], cfg["eta_max_conv"])
    pt_all, b_all, total = [], [], 0
    for df in pd.read_csv(_BKG / cfg["sample"], usecols=cols, chunksize=chunk):   # stream: bounded memory
        if max_rows and total >= max_rows:
            break
        if max_rows and total + len(df) > max_rows:
            df = df.iloc[: max_rows - total]
        total += len(df)
        pt_s, b_s = _select_chunk(df, cfg, cuts, rng, P_conv, act_cut)
        pt_all.append(pt_s); b_all.append(b_s)
        tick(f"  read {total:,} rows, survivors so far {sum(len(x) for x in pt_all):,}")
    pt = np.concatenate(pt_all) if pt_all else np.array([])
    b  = np.concatenate(b_all) if b_all else np.array([])
    tick(f"K_L done: {total:,} rows -> {len(pt):,} survivors, {len(b):,} with displacement")
    return pt, b, total


def _select_chunk(df, cfg, cuts, rng, P_conv, act_cut):
    """Vectorized K_L selection on one chunk; returns (pt_surv[], disp[]). Cuts are independent
       toggles, mirroring select_signal. Run 3 merged bin (dR<cell) needs both convert when use_tracks."""
    act = pd.to_numeric(df[cfg["iso_col"]], errors="coerce").to_numpy()
    if cfg["hard"]:
        pt_k, eta_k, phi_k = df["kl_pt"].to_numpy(), df["kl_eta"].to_numpy(), df["kl_phi"].to_numpy()
        e1, f1 = df["g1_eta"].to_numpy(), df["g1_phi"].to_numpy()
        e2, f2 = df["g2_eta"].to_numpy(), df["g2_phi"].to_numpy()
    else:
        pt_k, eta_k, phi_k = df["pt"].to_numpy(), df["eta"].to_numpy(), df["phi"].to_numpy()
        theta = 2.0 * np.arctan(np.exp(-eta_k))
        e1, f1, e2, f2 = kl.decay_kl_to_gg(pt_k * np.cosh(eta_k), theta, rng)

    n = len(pt_k)
    p = pt_k * np.cosh(eta_k)
    L = TRT_length(eta_k)
    decay_dist = -(p / M_KL) * kl.CTAU_KL * np.log(rng.uniform(size=n))
    tr = cuts["track_resolution"]
    lo, hi = cuts["eta_range"]

    act_ok = np.ones(n, dtype=bool) if act_cut is None else (act <= act_cut)
    intrk  = ((L > 0) & (decay_dist < L)) if cuts["mask_tracker_decay"] else np.ones(n, dtype=bool)
    eta_ok = (((np.abs(e1) >= lo) & (np.abs(e1) <= hi) & (np.abs(e2) >= lo) & (np.abs(e2) <= hi))
              if cuts["mask_eta"] else np.ones(n, dtype=bool))
    pt_ok  = (pt_k > cuts["pT_cut"]) if cuts["mask_alp_pT"] else np.ones(n, dtype=bool)
    keep = act_ok & intrk & eta_ok & pt_ok

    # binary conversion of both photons (decay-position corrected)
    both = ((rng.uniform(size=n) < kl.conv_prob_disp(P_conv, e1, decay_dist))
            & (rng.uniform(size=n) < kl.conv_prob_disp(P_conv, e2, decay_dist)))
    l1 = np.where(both, L - rng.uniform(decay_dist, np.maximum(L, decay_dist), size=n), 0.0)
    l2 = np.where(both, L - rng.uniform(decay_dist, np.maximum(L, decay_dist), size=n), 0.0)

    if cuts["mask_merge"]:
        dR = kl.delta_R_disp(e1, e2, f1, f2, decay_dist)
        if cfg["era"] == "run3":                                           # parking merge (config7)
            merged = dR < cuts["ecal_cell_size"]
            keep &= np.where(merged, both if cuts["use_tracks"] else False, dR < cuts["delta_r_max"])
        else:                                                              # scouting merge (config10)
            keep &= dR <= cuts["delta_r_max"]

    if cuts["mask_convert"] or cuts["mask_separation"] or cuts["mask_displacement"]:
        keep &= both                                                       # both photons convert
    if cuts["mask_separation"] or cuts["mask_displacement"]:
        sep, dsp = np.zeros(n), np.zeros(n)
        sep[keep] = kl.separation_TRT(e1[keep], e2[keep], eta_k[keep], f1[keep], f2[keep],
                                      phi_k[keep], decay_dist[keep])
        dsp[keep] = kl.displaced_vertex_TRT(eta_k[keep], f1[keep], f2[keep], phi_k[keep],
                                            l1[keep], l2[keep], decay_dist[keep], tr)
        if cuts["mask_separation"]:
            keep &= (sep >= cuts["sep_cut"])
        if cuts["mask_displacement"]:
            keep &= (dsp >= cuts["disp_cut"])

    disp_mask = keep & both
    b = kl.displaced_vertex_TRT(eta_k[disp_mask], f1[disp_mask], f2[disp_mask], phi_k[disp_mask],
                                l1[disp_mask], l2[disp_mask], decay_dist[disp_mask], tr)
    return pt_k[keep], np.asarray(b)


def main():
    import os
    ap = argparse.ArgumentParser(description="Config-driven signal-vs-K_L discriminator plots.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="path to a plot config JSON (see bkg/configs/)")
    src.add_argument("--analysis", choices=list(NOMINAL_CONFIG),
                     help="shortcut for bkg/configs/<era>_nominal.json")
    ap.add_argument("--max-kl", type=int, default=400000, help="cap K_L rows read (0 = all; yields exact only then)")
    ap.add_argument("--no-activity-cut", action="store_true", help="force the activity cut OFF (overrides the config)")
    ap.add_argument("--chunk", type=int, default=250_000, help="CSV read chunk size (lower on tight-memory nodes)")
    ap.add_argument("--norm", choices=["fraction", "events"], default="fraction",
                    help="DEPRECATED / ignored: both fraction (top row) and events @ lumi (middle row) are always drawn")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config_path = args.config or str(CONFIG_DIR / NOMINAL_CONFIG[args.analysis])
    stem = Path(config_path).stem
    cuts = load_cuts(config_path)
    era = cuts["era"]
    cfg = dict(ERA_PARAMS[era], era=era)
    if cuts["lumi"] is not None:
        cfg["lumi"] = cuts["lumi"]
    masses, gaggs = cuts["masses"], cuts["gagg"]
    _sig_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(masses)))     # match signal_plots.ipynb
    C_SIG = {ma: _sig_colors[i] for i, ma in enumerate(sorted(masses))}    # ascending mass -> low->high viridis
    act_cut = None if args.no_activity_cut else (cuts["act_cut"] if cuts["mask_activity"] else None)

    t0 = time.time()
    def tick(m): print(f"  [{time.time()-t0:6.1f}s] {m}", flush=True)
    rng = np.random.default_rng(0)
    tick(f"=== {stem} (era={era}) ===  activity_cut={act_cut}")
    np.random.seed(1234)
    kl_pt, kl_b, kl_read = select_kl(cfg, cuts, rng, args.max_kl, tick, act_cut, args.chunk)

    sig = {}
    for ma in masses:
        np.random.seed(1234)
        raw = read_data(ma_to_name(ma), num=N_GEN, data_dir=TRACKING + "/data/cmsrun3-csvs")
        events = raw_to_events(raw, gaggs=[gaggs[ma]], ma=ma, era=era)
        sig[ma] = select_signal(events, cuts)
        tick(f"signal m_a={ma}: {len(sig[ma][0]):,} survivors, {len(sig[ma][1]):,} with displacement")

    # --- physical yields (exact when --max-kl 0; else K_L is a subsample) ---
    w_kl = cfg["kl_sigma_pb"] / cfg["kl_ngen"] * cfg["lumi"] * 1000.0 * BR_KL_GG
    N_kl = len(kl_pt) * w_kl
    def sig_yield(ma):
        return len(sig[ma][0]) / N_GEN * cfg["lumi"] * 1000.0 * SIG_XSEC_1E2 * (gaggs[ma] / 1e-2) ** 2
    def w_sig_g(g):                                                         # per-event physical weight at coupling g
        return cfg["lumi"] * 1000.0 * SIG_XSEC_1E2 * (g / 1e-2) ** 2 / N_GEN
    def w_sig(ma):                                                          # per-event physical weight
        return w_sig_g(gaggs[ma])
    def hw(x, w_evt, norm):                                                # histogram weights per normalization
        if not len(x):
            return None
        return np.full(len(x), w_evt) if norm == "events" else _w(x)
    def ylab_for(norm):
        return (rf"events @ {cfg['lumi']:g} fb$^{{-1}}$ / bin" if norm == "events" else "fraction / bin")

    def draw_pt(ax, bins, xscale, title, norm):
        for ma in masses:
            if len(sig[ma][0]):
                ax.hist(sig[ma][0], bins=bins, weights=hw(sig[ma][0], w_sig(ma), norm), histtype="step",
                        lw=2, color=C_SIG[ma],
                        label=rf"ALP $m_a={ma}$ GeV, $g={gfmt(gaggs[ma])}$")
        if len(kl_pt):
            ax.hist(kl_pt, bins=bins, weights=hw(kl_pt, w_kl, norm), histtype="step", lw=2.2,
                    color="black", ls="--", label=r"$K_L$")
        ax.set_xscale(xscale); ax.set_yscale("log")
        ax.set_xlabel(r"diphoton $p_T$ [GeV]"); ax.set_ylabel(ylab_for(norm)); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.grid(True, which="both", alpha=0.15)

    # --- displacement: FIXED mass (DISP_MASS), coupling scan (benchmark x DISP_SCAN_FACTORS) ---
    # b~ shifts with coupling because the decay length scales as ~1/g^2 (larger g -> shorter, smaller b~).
    # Mass is held fixed (not the pT-panel masses) so the panel is a clean 1/g^2 coupling scan; couplings
    # are multiples of the DISP_MASS benchmark so the curves stay near the displaced-in-tracker regime.
    disp_mass   = DISP_MASS
    disp_gbench = GAGG_BENCH_DEFAULT.get(disp_mass, 1e-5)
    disp_color  = plt.cm.viridis(0.5)
    def disp_at_gagg(ma, g):                                               # re-run the signal selection at coupling g
        np.random.seed(1234)
        raw = read_data(ma_to_name(ma), num=N_GEN, data_dir=TRACKING + "/data/cmsrun3-csvs")
        _, b = select_signal(raw_to_events(raw, gaggs=[g], ma=ma, era=era), cuts)
        return b
    scan_gs = sorted({disp_gbench * f for f in DISP_SCAN_FACTORS})
    dbins = np.logspace(-5, 0, 50)
    disp_curves = []                                                       # precompute once (disp_at_gagg re-reads data)
    for ls, g in zip(["-", "--", ":", "-."], scan_gs):
        b = disp_at_gagg(disp_mass, g)
        disp_curves.append((ls, g, b[b > 0]))
    kl_b_pos = kl_b[kl_b > 0]

    def draw_disp(ax, norm):
        for ls, g, b in disp_curves:
            if len(b):
                ax.hist(b, bins=dbins, weights=hw(b, w_sig_g(g), norm), histtype="step", lw=2, ls=ls,
                        color=disp_color, label=rf"ALP $m_a={disp_mass}$ GeV, $g={gfmt(g)}$")
        if len(kl_b_pos):
            ax.hist(kl_b_pos, bins=dbins, weights=hw(kl_b_pos, w_kl, norm), histtype="step", lw=2.2,
                    color="black", ls="--", label=r"$K_L$")
        ax.axvline(1e-2, color="red", ls=":", lw=1.1); ax.axvline(1e-1, color="gray", ls=":", lw=1.1)
        ax.set_xscale("log")
        if norm == "events":
            ax.set_yscale("log")
        ax.set_xlabel(r"reconstructed displacement $\tilde{b}$ [m]"); ax.set_ylabel(ylab_for(norm))
        ax.legend(fontsize=8, frameon=False); ax.grid(True, which="both", alpha=0.15)

    # Two normalizations shown side by side: top row = fraction / bin, middle row = events @ lumi / bin.
    pt_log_bins = np.logspace(-1, 2.7, 60)
    pt_lin_bins = np.linspace(0.0, 300.0, 61)
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    for row, norm in enumerate(("fraction", "events")):
        draw_pt(axes[row, 0], pt_log_bins, "log",    r"diphoton $p_T$", norm)
        draw_pt(axes[row, 1], pt_lin_bins, "linear", r"diphoton $p_T$", norm)
        draw_disp(axes[row, 2], norm)

    # --- text panel: selection + counts + yields ---
    axt = axes[2, 0]
    axes[2, 1].axis("off"); axes[2, 2].axis("off")
    axt.axis("off")
    info = [f"config: {stem}", f"era = {era}    lumi = {cfg['lumi']} fb^-1", "", "SELECTION"]
    info += ["  " + s for s in selection_lines(cfg, cuts)]
    sub = "" if args.max_kl == 0 else "   [SUBSAMPLE -- use --max-kl 0 for full yields]"
    info += ["", "K_L BACKGROUND" + sub,
             f"  rows read : {kl_read:,}",
             f"  survivors : {len(kl_pt):,}",
             f"  w/ 2-conv : {len(kl_b):,}   (enter displacement)",
             f"  yield     : {N_kl:.3e} events",
             "", "SIGNAL   (survivors / N_gen  ->  yield @ g)"]
    for ma in masses:
        info.append(f"  m={ma} (g={gaggs[ma]:.1e}) : {len(sig[ma][0]):>4}/{N_GEN} -> {sig_yield(ma):.2e}")
    axt.text(0.0, 1.0, "\n".join(info), va="top", ha="left", family="monospace",
             fontsize=8.5, transform=axt.transAxes)

    fig.suptitle(f"{stem}: signal vs $K_L$   (era {era})", fontsize=13)
    fig.tight_layout()
    out = args.out or str(_BKG / "plots" / f"{stem}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

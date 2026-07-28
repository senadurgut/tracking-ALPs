"""
Signal (ALP->gg, VBF) vs background (K_L->gg) distributions in the three discriminating
observables -- diphoton mass, diphoton pT, reconstructed displacement b~ -- AFTER the full
config7/config10 selection, PLUS a hadronic-activity == 0 requirement on the background.

Shape-normalized (unit area): shows discrimination power, not rate. Signal shown at a few masses;
K_L only ever lives at m_gg = 0.498 GeV.

Selection is a faithful copy of analyze_configurable_v2.py `count_passing` (signal) and of
run3_parking.py / phase2_scouting.py (K_L). Displacement follows notebooks/displacement.py:
signal uses the MEAN decay length ev['a']['l']; K_L uses the SAMPLED decay position (both go into
displaced_vertex_TRT). Both require the two photons to convert (displacement is only defined then).

Run (heavy -- send to SLURM or a node):  python plot_sig_vs_bkg.py --analysis run3
                                          python plot_sig_vs_bkg.py --analysis phase2
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
TRACK_RES = 4e-5
USE_TRACKS = True                    # merged bin (dR<cell) resolved via conversion, as in run3_parking.py
ACT_CUT   = 0.0                      # keep events with hadronic activity <= ACT_CUT (0 = perfectly isolated)
SIG_MASSES = [0.3, 0.5, 1.0]         # GeV
GAGG_BENCH = {0.3: 8.0e-5, 0.5: 2.63e-5, 1.0: 6.4e-6}   # displaced-in-tracker benchmark per mass
N_GEN     = 40000
BR_KL_GG     = 5.47e-4               # Br(K_L -> gamma gamma)
SIG_XSEC_1E2 = 180.0                 # sigma_VBF(a) [pb] at gagg=1e-2 GeV^-1 (~gagg^2, Br(a->gg)=1)
C_SIG     = {0.3: "#0072B2", 0.5: "#009E73", 1.0: "#E69F00"}   # Okabe-Ito, fixed per mass

ANALYSES = {
    "run3":   dict(era="run3",   config=TRACKING + "/configs/config7.json",
                   sample="data/kl_vbf_sample.csv",  iso_col="kl_iso_had_pt", hard=True,  lumi=312,
                   kl_sigma_pb=3.6413e-3 * 1e9, kl_ngen=20e6),   # sigma_hard(pThat>80); N_HARD (VBF baked in)
    "phase2": dict(era="phase2", config=TRACKING + "/configs/config10.json",
                   sample="data/kl_sample.csv",      iso_col="iso_had_pt",    hard=False, lumi=3000,
                   kl_sigma_pb=78.585e9,        kl_ngen=2e6),    # sigma_inel; N_SOFT
}

def selection_lines(cfg, cuts, act_cut):
    """Human-readable list of the cuts applied (for the text panel)."""
    lo, hi = cuts["eta_range"]
    L = ["in-tracker decay (always on)",
         f"eta in [{lo}, {hi}]  (both photons)"]
    if cfg["era"] == "run3":
        L += ["VBF: baked into K_L sample; applied to signal:",
              f"     lead>{cuts['leading_jet_pt_cut']:.0f}, sub>{cuts['sub_jet_pt_cut']:.0f}, "
              f"mjj>{cuts['mjj_cut']:.0f}, |dEta|>{cuts['deta_cut']:.0f}",
              f"merge: dR < {cuts['delta_r_max']}  (dR<{cuts['ecal_cell_size']} needs both convert)"]
    else:
        L += [f"merge: dR <= {cuts['delta_r_max']}", "both photons convert",
              f"separation_TRT >= {cuts['sep_cut']:g} m",
              f"displaced_vertex_TRT >= {cuts['disp_cut']:g} m  (res {cuts['track_resolution']:g} m)"]
    L.append(f"hadronic activity: {'<= %g GeV' % act_cut if act_cut is not None else 'NO CUT'}")
    return L


def load_cuts(path):
    """Resolve the config JSON to the cut params used here (mirrors analyze_configurable_v2.resolve_cuts
       for the fields we need; use_tracks pinned to USE_TRACKS for signal/background consistency)."""
    with open(path) as f:
        c = json.load(f)
    em, vm, mm, p2 = (c.get(k, {}) for k in ("mask_eta", "mask_vbf", "mask_merge", "phase2_cuts"))
    return dict(
        era=c["era"], analysis_mode=c.get("analysis_mode", "parking"),
        mask_eta=em.get("value", False), eta_range=em.get("eta_range", [0.0, 0.0]),
        mask_vbf=vm.get("value", False), leading_jet_pt_cut=vm.get("lead_pt", 0.0),
        sub_jet_pt_cut=vm.get("sub_pt", 0.0), mjj_cut=vm.get("mjj", 0.0), deta_cut=vm.get("deta", 0.0),
        mask_merge=mm.get("value", True), delta_r_max=mm.get("delta_r_max", 0.3),
        ecal_cell_size=0.025, use_tracks=USE_TRACKS,
        sep_cut=p2.get("sep_cut", 5e-4), disp_cut=p2.get("disp_cut", 1e-1),
        track_resolution=p2.get("track_resolution", 2e-4),
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
    """Full config selection at i_g=0; return (pt[], m_gg[], disp[]) for survivors.
       disp only filled for survivors whose two photons convert (else that event is dropped
       from the displacement array only)."""
    i_g, pt, disp = 0, [], []
    for ev in events:
        if -ev["a"]["l"][i_g] * np.log(np.random.uniform()) >= TRT_length(ev["a"]["eta"]):
            continue                                                        # in-tracker decay
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
        if cuts["analysis_mode"] == "scouting":
            if not both_conv:
                continue
            if separation_TRT(e1, e2, ev["a"]["eta"], ev["g1"]["phi"], ev["g2"]["phi"],
                              ev["a"]["phi"], l_a) < cuts["sep_cut"]:
                continue
            if displaced_vertex_TRT(ev["a"]["eta"], ev["g1"]["phi"], ev["g2"]["phi"], ev["a"]["phi"],
                                    ev["g1"]["l_track"][i_g], ev["g2"]["l_track"][i_g], l_a,
                                    cuts["track_resolution"]) < cuts["disp_cut"]:
                continue
        pt.append(ev["a"]["pt"])
        if both_conv:
            disp.append(displaced_vertex_TRT(ev["a"]["eta"], ev["g1"]["phi"], ev["g2"]["phi"],
                        ev["a"]["phi"], ev["g1"]["l_track"][i_g], ev["g2"]["l_track"][i_g],
                        l_a, TRACK_RES))
    return np.array(pt), np.array(disp)


def select_kl(cfg, cuts, rng, max_rows, tick, act_cut, chunk=250_000):
    """Stream the K_L sample in chunks (bounded memory); full selection + activity cut (act_cut = the
       threshold, None disables it). max_rows caps rows read (0 = all). Returns (pt[], disp[], n_read)."""
    cols = (["kl_pt", "kl_eta", "kl_phi", "g1_eta", "g1_phi", "g2_eta", "g2_phi", cfg["iso_col"]]
            if cfg["hard"] else ["pt", "eta", "phi", cfg["iso_col"]])
    P_conv = kl.make_pconv(cfg["era"], cuts["eta_range"][1])
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
    """Vectorized K_L selection on one chunk; returns (pt_surv[], disp[]). Run 3 merged bin
       (dR<cell) requires both convert (use_tracks=True, as in run3_parking.py)."""
    act = pd.to_numeric(df[cfg["iso_col"]], errors="coerce").to_numpy()
    if cfg["hard"]:
        pt_k, eta_k, phi_k = df["kl_pt"].to_numpy(), df["kl_eta"].to_numpy(), df["kl_phi"].to_numpy()
        e1, f1 = df["g1_eta"].to_numpy(), df["g1_phi"].to_numpy()
        e2, f2 = df["g2_eta"].to_numpy(), df["g2_phi"].to_numpy()
    else:
        pt_k, eta_k, phi_k = df["pt"].to_numpy(), df["eta"].to_numpy(), df["phi"].to_numpy()
        theta = 2.0 * np.arctan(np.exp(-eta_k))
        e1, f1, e2, f2 = kl.decay_kl_to_gg(pt_k * np.cosh(eta_k), theta, rng)

    p = pt_k * np.cosh(eta_k)
    L = TRT_length(eta_k)
    decay_dist = -(p / M_KL) * kl.CTAU_KL * np.log(rng.uniform(size=len(p)))
    lo, hi = cuts["eta_range"]
    act_ok = np.ones(len(p), dtype=bool) if act_cut is None else (act <= act_cut)
    keep = (act_ok & (L > 0) & (decay_dist < L)                             # activity + in-tracker decay
            & (np.abs(e1) >= lo) & (np.abs(e1) <= hi)
            & (np.abs(e2) >= lo) & (np.abs(e2) <= hi))                      # mask_eta

    # binary conversion of both photons (decay-position corrected); gates displacement everywhere,
    # and is a hard cut for scouting + the run3 merged bin.
    both = ((rng.uniform(size=len(p)) < kl.conv_prob_disp(P_conv, e1, decay_dist))
            & (rng.uniform(size=len(p)) < kl.conv_prob_disp(P_conv, e2, decay_dist)))
    l1 = np.where(both, L - rng.uniform(decay_dist, np.maximum(L, decay_dist), size=len(p)), 0.0)
    l2 = np.where(both, L - rng.uniform(decay_dist, np.maximum(L, decay_dist), size=len(p)), 0.0)

    dR = kl.delta_R_disp(e1, e2, f1, f2, decay_dist)
    if cfg["era"] == "run3":                                               # parking merge (config7)
        merged = dR < cuts["ecal_cell_size"]
        keep &= np.where(merged, both, dR < cuts["delta_r_max"])           # merged->2conv, else dR<max
    else:                                                                  # scouting merge (config10)
        keep &= dR <= cuts["delta_r_max"]
        keep &= both                                                       # scouting: both convert always
        sep, dsp = np.zeros(len(p)), np.zeros(len(p))
        sep[keep] = kl.separation_TRT(e1[keep], e2[keep], eta_k[keep], f1[keep], f2[keep],
                                      phi_k[keep], decay_dist[keep])
        dsp[keep] = kl.displaced_vertex_TRT(eta_k[keep], f1[keep], f2[keep], phi_k[keep],
                                            l1[keep], l2[keep], decay_dist[keep], TRACK_RES)
        keep &= (sep >= cuts["sep_cut"]) & (dsp >= cuts["disp_cut"])

    disp_mask = keep & both
    b = kl.displaced_vertex_TRT(eta_k[disp_mask], f1[disp_mask], f2[disp_mask], phi_k[disp_mask],
                                l1[disp_mask], l2[disp_mask], decay_dist[disp_mask], TRACK_RES)
    return pt_k[keep], np.asarray(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", choices=list(ANALYSES), required=True)
    ap.add_argument("--max-kl", type=int, default=400000, help="cap K_L rows read (0 = all; yields exact only then)")
    ap.add_argument("--no-activity-cut", action="store_true", help="drop the hadronic-activity requirement")
    ap.add_argument("--chunk", type=int, default=250_000, help="CSV read chunk size (lower on tight-memory nodes)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = ANALYSES[args.analysis]
    cuts = load_cuts(cfg["config"])
    act_cut = None if args.no_activity_cut else ACT_CUT
    t0 = time.time()
    def tick(m): print(f"  [{time.time()-t0:6.1f}s] {m}", flush=True)

    rng = np.random.default_rng(0)
    tick(f"=== {args.analysis} (era={cfg['era']}) ===  activity_cut={act_cut}")
    np.random.seed(1234)
    kl_pt, kl_b, kl_read = select_kl(cfg, cuts, rng, args.max_kl, tick, act_cut, args.chunk)

    sig = {}
    for ma in SIG_MASSES:
        np.random.seed(1234)
        raw = read_data(ma_to_name(ma), num=N_GEN, data_dir=TRACKING + "/data/cmsrun3-csvs")
        events = raw_to_events(raw, gaggs=[GAGG_BENCH[ma]], ma=ma, era=cfg["era"])
        sig[ma] = select_signal(events, cuts)
        tick(f"signal m_a={ma}: {len(sig[ma][0]):,} survivors, {len(sig[ma][1]):,} with displacement")

    # --- physical yields (exact when --max-kl 0; else K_L is a subsample) ---
    w_kl = cfg["kl_sigma_pb"] / cfg["kl_ngen"] * cfg["lumi"] * 1000.0 * BR_KL_GG
    N_kl = len(kl_pt) * w_kl
    def sig_yield(ma):
        g = GAGG_BENCH[ma]
        return len(sig[ma][0]) / N_GEN * cfg["lumi"] * 1000.0 * SIG_XSEC_1E2 * (g / 1e-2) ** 2

    fig, ((axm, axp), (axd, axt)) = plt.subplots(2, 2, figsize=(13, 9))

    # --- mass ---
    mbins = np.linspace(0.0, 1.2, 121)
    for ma in SIG_MASSES:
        if len(sig[ma][0]):
            axm.hist(np.full(len(sig[ma][0]), ma), bins=mbins, density=True, histtype="step",
                     lw=2, color=C_SIG[ma], label=rf"ALP $m_a={ma}$ GeV")
    axm.hist(np.full(len(kl_pt), M_KL), bins=mbins, density=True, histtype="step", lw=2.2,
             color="black", ls="--", label=r"$K_L$")
    axm.set_xlabel(r"diphoton mass $m_{\gamma\gamma}$ [GeV]"); axm.set_ylabel("normalized / bin")
    axm.legend(fontsize=8, frameon=False); axm.grid(True, alpha=0.15)

    # --- pT ---
    pbins = np.logspace(-1, 2.7, 60)
    for ma in SIG_MASSES:
        if len(sig[ma][0]):
            axp.hist(sig[ma][0], bins=pbins, density=True, histtype="step", lw=2,
                     color=C_SIG[ma], label=rf"ALP $m_a={ma}$ GeV")
    if len(kl_pt):
        axp.hist(kl_pt, bins=pbins, density=True, histtype="step", lw=2.2, color="black", ls="--",
                 label=r"$K_L$")
    axp.set_xscale("log"); axp.set_yscale("log")
    axp.set_xlabel(r"diphoton $p_T$ [GeV]"); axp.set_ylabel("normalized / bin")
    axp.legend(fontsize=8, frameon=False); axp.grid(True, which="both", alpha=0.15)

    # --- displacement ---
    dbins = np.logspace(-5, 0, 50)
    for ma in SIG_MASSES:
        b = sig[ma][1]; b = b[b > 0]
        if len(b):
            axd.hist(b, bins=dbins, density=True, histtype="step", lw=2, color=C_SIG[ma],
                     label=rf"ALP $m_a={ma}$ GeV ($g={GAGG_BENCH[ma]:.0e}$)")
    b = kl_b[kl_b > 0]
    if len(b):
        axd.hist(b, bins=dbins, density=True, histtype="step", lw=2.2, color="black", ls="--",
                 label=r"$K_L$")
    axd.axvline(1e-2, color="red", ls=":", lw=1.1); axd.axvline(1e-1, color="gray", ls=":", lw=1.1)
    axd.set_xscale("log"); axd.set_xlabel(r"reconstructed displacement $\tilde{b}$ [m]")
    axd.set_ylabel("normalized / bin"); axd.legend(fontsize=8, frameon=False)
    axd.grid(True, which="both", alpha=0.15)

    # --- text panel: selection + counts + yields ---
    axt.axis("off")
    info = [f"{args.analysis}   (config {'7' if cfg['era'] == 'run3' else '10'})",
            f"lumi = {cfg['lumi']} fb^-1",
            "", "SELECTION"]
    info += ["  " + s for s in selection_lines(cfg, cuts, act_cut)]
    sub = "" if args.max_kl == 0 else "   [SUBSAMPLE -- use --max-kl 0 for full yields]"
    info += ["", "K_L BACKGROUND" + sub,
             f"  rows read : {kl_read:,}",
             f"  survivors : {len(kl_pt):,}",
             f"  w/ 2-conv : {len(kl_b):,}   (enter displacement)",
             f"  yield     : {N_kl:.3e} events",
             "", "SIGNAL   (survivors / N_gen  ->  yield @ benchmark g)"]
    for ma in SIG_MASSES:
        info.append(f"  m={ma} (g={GAGG_BENCH[ma]:.1e}) : {len(sig[ma][0]):>4}/{N_GEN} -> {sig_yield(ma):.2e}")
    axt.text(0.0, 1.0, "\n".join(info), va="top", ha="left", family="monospace",
             fontsize=9, transform=axt.transAxes)

    act_txt = "no activity cut" if act_cut is None else rf"activity $\leq$ {ACT_CUT:g} GeV"
    suffix  = "_no_activity_cut" if act_cut is None else ""
    fig.suptitle(f"{args.analysis}: signal vs $K_L$ -- full selection, {act_txt}", fontsize=13)
    fig.tight_layout()
    out = args.out or f"plots/sig_vs_bkg_{args.analysis}{suffix}.png"
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""
K_L -> gamma gamma background for the HL-LHC L1 data-scouting analysis.

Selection mirrors analyze_configurable_v2.py with config10.json (era=phase2, mode=scouting):
    mask_eta  : both photons in |eta| in [1.4, 3.0]  (HGCAL)
    mask_vbf  : False  -> NO VBF factor (scouting records without trigger thresholds)
    mask_merge: delta_r_max = 0.3   (scouting merge = just dR <= dR_max, no conversion embedded)
    scouting block (always):
        - BOTH photons convert
        - separation_TRT       >= sep_cut  (1e-4 m)
        - displaced_vertex_TRT >= disp_cut (1e-2 m)   (track_resolution = 4e-5 m)

Input: kl_sample.csv (inclusive K_L; NO photon columns -> each K_L is decayed to gg analytically).
Unlike Run 3, the displacement here is an EXPLICIT selection cut (no data-driven sidebands at L1).
"""
import time
import numpy as np
import pandas as pd
import kl_common as kl

# --- selection constants (config10) ---
ETA_LO, ETA_HI = 1.4, 3.0      # mask_eta.eta_range (HGCAL)
DELTA_R_MAX    = 0.3           # mask_merge.delta_r_max
SEP_CUT        = 1e-4          # phase2_cuts.sep_cut          [m]
DISP_CUT       = 1e-2          # phase2_cuts.disp_cut         [m]
TRACK_RES      = 4e-5          # phase2_cuts.track_resolution [m]
LUMI_FB        = 3000.0        # Phase-2 integrated luminosity (250/fb first-year point reported too)

# --- soft-sample normalization (w_soft = physical K_L->gg events represented by one row) ---
SIGMA_INEL_PB  = 78.585e9      # inelastic pp cross section [pb] (78.585 mb)
N_SOFT         = 2e6           # generated inelastic events (all K_L, both hemispheres)
SAMPLE_CSV     = "data/kl_sample.csv"

def cutflow_phase2(df, P_conv, rng, tick):
    """Apply config10's full scouting chain to the soft sample; return the (produced, in-tracker,
       eta+merge+2conv, +separation+displacement) event yields at LUMI_FB. phi_parent = 0
       (fixed by decay_kl_to_gg -- only relative photon angles enter the cuts)."""
    pt, eta_kl = df["pt"].to_numpy(), df["eta"].to_numpy()
    p     = pt * np.cosh(eta_kl)                                           # |p| of the K_L
    theta = 2.0 * np.arctan(np.exp(-eta_kl))
    tick(f"kinematics done ({len(df):,} rows)")

    w_soft  = SIGMA_INEL_PB / N_SOFT * LUMI_FB * kl.FB_TO_PB_INV * kl.BR_KL_GG
    f_intrk = kl.f_in_tracker(p, eta_kl)                                   # in-tracker decay (always on)
    tick(f"in-tracker weight done (mean f_in_tracker = {f_intrk.mean():.3e})")

    decay_dist = kl.sample_decay_distance(p, eta_kl, rng)
    e1, f1, e2, f2 = kl.decay_kl_to_gg(p, theta, rng)                      # soft sample has no photons
    tick("decayed K_L -> gg + sampled decay positions")

    acc = ((np.abs(e1) >= ETA_LO) & (np.abs(e1) <= ETA_HI) &
           (np.abs(e2) >= ETA_LO) & (np.abs(e2) <= ETA_HI))                # mask_eta (HGCAL)
    dR = kl.delta_R_disp(e1, e2, f1, f2, decay_dist)
    pass_merge = (dR <= DELTA_R_MAX)                                       # scouting merge
    pconv2 = kl.conv_prob_disp(P_conv, e1, decay_dist) * kl.conv_prob_disp(P_conv, e2, decay_dist)  # BOTH convert, from decay point
    keep = acc & pass_merge & (pconv2 > 0)
    tick(f"eta+merge+conv: {int(keep.sum()):,} rows reach the scouting cuts")

    l1, l2 = kl.sample_track_length(e1, decay_dist, rng), kl.sample_track_length(e2, decay_dist, rng)
    sep, disp = np.zeros(len(df)), np.zeros(len(df))
    sep[keep]  = kl.separation_TRT(e1[keep], e2[keep], eta_kl[keep], f1[keep], f2[keep], 0.0, decay_dist[keep])
    disp[keep] = kl.displaced_vertex_TRT(eta_kl[keep], f1[keep], f2[keep], 0.0, l1[keep], l2[keep], decay_dist[keep], TRACK_RES)
    pass_scout = (sep >= SEP_CUT) & (disp >= DISP_CUT)
    tick("separation + displacement done")

    base = w_soft * f_intrk
    return (w_soft * len(df), base.sum(),
            (base * acc * pass_merge * pconv2).sum(),
            (base * acc * pass_merge * pconv2 * pass_scout).sum())

def main():
    t0 = time.time()
    def tick(msg): print(f"  [{time.time()-t0:6.1f}s] {msg}", flush=True)

    rng = np.random.default_rng(0)
    tick(f"reading {SAMPLE_CSV} ...")
    df = pd.read_csv(SAMPLE_CSV)
    tick(f"loaded {len(df):,} rows")
    P_conv = kl.make_pconv("phase2", ETA_HI)

    n_prod, n_intrk, n_conv, n_scout = cutflow_phase2(df, P_conv, rng, tick)
    tick("cut flow complete")
    stages = [("produced (Br incl.)",          n_prod),
              ("x f_in_tracker",               n_intrk),
              ("x eta & merge & 2conv",        n_conv),
              ("x separation & displacement",  n_scout)]

    print("\n=== Phase-2 scouting : K_L -> gamma gamma background cut-flow ===")
    kl.print_yields(stages, LUMI_FB)
    print(f"first-year (250/fb): N_final = {n_scout * 250.0 / LUMI_FB:.3e}")
    print("No VBF factor (mask_vbf = False in config10).")
    print("NOTE: all K_L have m_gg = m_KL = 0.498 GeV -> background localized at m_a ~ 0.5 GeV.")

if __name__ == "__main__":
    main()

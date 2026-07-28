"""
K_L -> gamma gamma background for the Run 3 VBF-parking analysis.

Selection mirrors analyze_configurable_v2.py with config7.json (era=run3, mode=parking):
    mask_eta  : both photons in |eta| in [0, 2.5]
    mask_vbf  : ALREADY baked into kl_vbf_sample.csv -- it was dumped only for VBF-passing
                events and w_hard normalizes by N_HARD = ALL generated hard events, so the VBF
                rate is carried by the weight. Do NOT re-apply mask_vbf or any eps_VBF factor.
    mask_merge: delta_r_max = 0.3 ; conversion required ONLY when dR < ECAL cell (use_tracks)

Input: kl_vbf_sample.csv (VBF-triggered K_L + the two K_L->gg photons, decayed at dump time).
Displacement is a FIT discriminant in Run 3 (2D mass-displacement fit), NOT a selection cut,
so it does not appear as a suppression factor here.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import kl_common as kl

_BKG = Path(__file__).resolve().parent.parent    # tracking_ALPs/bkg (data/ lives here)

# --- selection constants (config7 + analyze_configurable_v2.py defaults) ---
ETA_LO, ETA_HI = 0.0, 2.5      # mask_eta.eta_range
DELTA_R_MAX    = 0.3           # mask_merge.delta_r_max
ECAL_CELL_SIZE = 0.025         # analyze_configurable_v2.py default
LUMI_FB        = 312.0         # Run 3 parking integrated luminosity

# --- hard-sample normalization (w_hard = physical VBF K_L->gg events represented by one row) ---
SIGMA_HARD_PB  = 3.6413e-3 * 1e9   # sigma_hard(pThat>80) [pb]
N_HARD         = 20e6              # ALL generated hard events (carries the VBF rate; NOT the row count)
SAMPLE_CSV     = str(_BKG / "data" / "kl_vbf_sample.csv")

def cutflow_run3(df, P_conv, rng, tick):
    """Apply config7's non-VBF cut chain to the hard sample; return the (produced, in-tracker,
       final) event yields at LUMI_FB. Conversion enters only the very-merged bin (dR < cell)."""
    kl_pt, kl_eta = df["kl_pt"].to_numpy(), df["kl_eta"].to_numpy()
    e1, f1 = df["g1_eta"].to_numpy(), df["g1_phi"].to_numpy()
    e2, f2 = df["g2_eta"].to_numpy(), df["g2_phi"].to_numpy()
    p = kl_pt * np.cosh(kl_eta)                                            # |p| of the K_L
    tick(f"kinematics done ({len(df):,} rows)")

    w_hard  = SIGMA_HARD_PB / N_HARD * LUMI_FB * kl.FB_TO_PB_INV * kl.BR_KL_GG
    f_intrk = kl.f_in_tracker(p, kl_eta)                                   # in-tracker decay (always on)
    tick(f"in-tracker weight done (mean f_in_tracker = {f_intrk.mean():.3e})")

    decay_dist = kl.sample_decay_distance(p, kl_eta, rng)
    tick("decay positions sampled")

    acc = ((np.abs(e1) >= ETA_LO) & (np.abs(e1) <= ETA_HI) &
           (np.abs(e2) >= ETA_LO) & (np.abs(e2) <= ETA_HI))                # mask_eta
    tick(f"eta cut: {int(acc.sum()):,}/{len(df):,} rows keep both photons")

    dR = kl.delta_R_disp(e1, e2, f1, f2, decay_dist)
    n_merged   = int((dR < ECAL_CELL_SIZE).sum())
    n_resolved = int(((dR >= ECAL_CELL_SIZE) & (dR < DELTA_R_MAX)).sum())
    tick(f"merge: {n_merged:,} very-merged (dR<{ECAL_CELL_SIZE}), {n_resolved:,} resolved (<{DELTA_R_MAX})")

    pconv2 = kl.conv_prob_disp(P_conv, e1, decay_dist) * kl.conv_prob_disp(P_conv, e2, decay_dist)  # both convert, from decay point
    contrib = np.where(dR < ECAL_CELL_SIZE, pconv2,                        # merged -> need tracks
              np.where(dR < DELTA_R_MAX,     1.0, 0.0))                     # resolved -> pass ; wide -> fail
    tick("conversion + merge weights done")

    return w_hard * len(df), (w_hard * f_intrk).sum(), (w_hard * f_intrk * acc * contrib).sum()

def main():
    t0 = time.time()
    def tick(msg): print(f"  [{time.time()-t0:6.1f}s] {msg}", flush=True)

    rng = np.random.default_rng(0)
    tick(f"reading {SAMPLE_CSV} ...")
    df = pd.read_csv(SAMPLE_CSV)
    tick(f"loaded {len(df):,} rows")
    P_conv = kl.make_pconv("run3", ETA_HI)

    n_prod, n_intrk, n_final = cutflow_run3(df, P_conv, rng, tick)
    tick("cut flow complete")
    stages = [("produced (VBF & Br incl.)", n_prod),
              ("x f_in_tracker",            n_intrk),
              ("x eta & merge & conv",      n_final)]

    print("\n=== Run 3 parking : K_L -> gamma gamma background cut-flow ===")
    kl.print_yields(stages, LUMI_FB)
    print("VBF already baked into kl_vbf_sample.csv (no mask_vbf, no eps_VBF).")
    print("NOTE: all K_L have m_gg = m_KL = 0.498 GeV, so this background only populates "
          "the m_a ~ 0.5 GeV region of the search.")

if __name__ == "__main__":
    main()

"""
Shared physics for the K_L -> gamma gamma background estimate.

Both `run3_parking.py` and `phase2_scouting.py` import from here, so the common
physics (FORESEE spectrum, K_L decay, conversion, in-tracker fraction) can never
drift between the two analyses. Analysis-specific *selections* live in the two
analysis scripts, each mirroring analyze_configurable_v2.py for its config.

Run from this directory:  python run3_parking.py   /   python phase2_scouting.py
"""
import sys
import numpy as np

# --- make the signal's module_VBF importable (same functions the signal uses) ---
TRACKING_ALPS = "/home/export/sdurgut/scratch/alps/tracking_ALPs"
FORESEE_ROOT  = "/home/export/sdurgut/scratch/alps/bkg/FORESEE"
sys.path.append(TRACKING_ALPS + "/analysis")

from module_VBF import (
    TRT_length,            # tracker path length [m] vs eta (CMS geometry)
    conv_prob_novec,       # P_conv(eta) from material budget (NOT std_conv_prob = its uncertainty!)
    load_conv_prob_table,  # populate the conversion table from a material-budget file
    R_ECAL,                # ECAL inner radius [m], for the displaced Delta_R
    separation_TRT       as _separation_TRT,       # scalar; vectorized over bins below
    displaced_vertex_TRT as _displaced_vertex_TRT, # scalar; vectorized over bins below
)

separation_TRT       = np.vectorize(_separation_TRT)
displaced_vertex_TRT = np.vectorize(_displaced_vertex_TRT)

# ----------------------------------------------------------------------
# constants (PDG)
# ----------------------------------------------------------------------
M_KL         = 0.497611     # GeV, K_L mass
CTAU_KL      = 15.34        # m,   K_L ctau  (-> ell = 4.6 km at p = 150 GeV)
BR_KL_GG     = 5.47e-4      # Br(K_L -> gamma gamma)
FB_TO_PB_INV = 1000.0       # 1 fb^-1 = 1000 pb^-1
SIGMA_INEL   = 78e9         # pb, inelastic pp xsec at 13.6 TeV (multiplicity cross-check only)

MATERIAL_BUDGET = {
    "run3":   TRACKING_ALPS + "/data/material_budget/phase1_material_budget.txt",
    "phase2": TRACKING_ALPS + "/data/material_budget/phase2_material_budget.txt",
}

# ----------------------------------------------------------------------
# factor (A): FORESEE forward K_L spectrum
# ----------------------------------------------------------------------
def load_kl_spectrum(generator="EPOSLHC", energy="13.6"):
    """Per-bin arrays (p [GeV], theta [rad], eta, sigma [pb]) for forward K_L production.
       sigma is the cross section per bin, FORWARD hemisphere only (x2 for both sides at yield stage)."""
    fname = f"{FORESEE_ROOT}/files/hadrons/{energy}TeV/{generator}/{generator}_{energy}TeV_130.txt"
    logth, logp, sigma = np.loadtxt(fname, comments="#", unpack=True)
    theta = 10.0**logth
    p     = 10.0**logp
    pt    = p * np.sin(theta)
    pz    = p * np.cos(theta)
    eta   = np.arcsinh(pz / pt)
    return p, theta, eta, sigma

# ----------------------------------------------------------------------
# K_L kinematics
# ----------------------------------------------------------------------
def kl_decay_length(p):
    """Lab-frame K_L decay length [m] for momentum p [GeV].  ell = (p/m) c*tau = 30.8 m * (p/GeV)."""
    return (p / M_KL) * CTAU_KL

def f_in_tracker(p, eta):
    """Factor (C): fraction of K_L that decay inside the tracker.
       Analytic expectation of the signal MC cut (analyze_configurable_v2.py:146)."""
    L = TRT_length(eta)                       # 0 outside tracker eta acceptance
    return 1.0 - np.exp(-L / kl_decay_length(p))

def decay_kl_to_gg(p, theta_kl, rng):
    """Isotropic two-body K_L -> gamma gamma, boosted to the lab. Returns eta1,phi1,eta2,phi2.
       (phi_KL fixed to 0; only |eta| and dR matter downstream.)"""
    n     = len(p)
    E     = np.sqrt(p**2 + M_KL**2)
    beta  = p / E
    gam   = E / M_KL
    Estar = M_KL / 2.0
    cth = rng.uniform(-1, 1, n)               # isotropic (spin-0)
    sth = np.sqrt(1 - cth**2)
    phi = rng.uniform(0, 2*np.pi, n)
    px_s, py_s, pz_s = Estar*sth*np.cos(phi), Estar*sth*np.sin(phi), Estar*cth
    def to_lab(pxs, pys, pzs):
        pz1  = gam * (pzs + beta*Estar)                       # boost along K_L direction
        px_l =  pxs*np.cos(theta_kl) + pz1*np.sin(theta_kl)   # rotate to lab K_L direction
        pz_l = -pxs*np.sin(theta_kl) + pz1*np.cos(theta_kl)
        pt_l = np.sqrt(px_l**2 + pys**2)
        return np.arcsinh(pz_l/pt_l), np.arctan2(pys, px_l)
    e1, f1 = to_lab( px_s,  py_s,  pz_s)
    e2, f2 = to_lab(-px_s, -py_s, -pz_s)      # 2nd photon back-to-back in rest frame
    return e1, f1, e2, f2

def sample_decay_distance(p, eta, rng):
    """Decay distance [m] from the IP, conditioned on decaying within the tracker
       (truncated exponential on [0, L]; ~ uniform since ell >> L for K_L)."""
    ell = kl_decay_length(p)
    L   = TRT_length(eta)
    U   = rng.uniform(0, 1, len(p))
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(L > 0, -ell*np.log(1 - U*(1 - np.exp(-L/ell))), 0.0)
    return d

# ----------------------------------------------------------------------
# factor (D): conversion, and displaced Delta_R
# ----------------------------------------------------------------------
def make_pconv(era, eta_max):
    """Load the material budget for `era` and return a safe vectorized P_conv(|eta|),
       returning 0 outside |eta| < eta_max. Uses conv_prob (probability), NOT std_conv_prob."""
    load_conv_prob_table(MATERIAL_BUDGET[era])
    _cp = np.vectorize(conv_prob_novec)
    def P_conv(eta_arr):
        a = np.abs(eta_arr)
        out = np.zeros_like(a, dtype=float)
        m = a < eta_max
        out[m] = _cp(a[m])
        return out
    return P_conv

def delta_R_disp(eta1, eta2, phi1, phi2, decay_dist):
    """Vectorized copy of the signal's Delta_R (module_VBF.py:828): displaced-vertex-corrected dR.
       decay_dist is the K_L decay position [m] (sampled), which plays the role of l_a for signal."""
    dR = np.sqrt((np.abs(phi1 - phi2))**2 + (np.abs(eta1 - eta2))**2)
    return np.minimum((R_ECAL - decay_dist) / R_ECAL, 1.0) * dR

def conv_prob_disp(P_conv, eta_gamma, decay_dist):
    """P_conv(eta) scaled by the tracker fraction left past the K_L decay point (paper Eq 4.2):
       only material beyond the (sampled) decay position is available for conversion."""
    L = TRT_length(eta_gamma)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.clip(1.0 - decay_dist / L, 0.0, 1.0)
    return P_conv(eta_gamma) * np.where(L > 0, frac, 0.0)

def sample_track_length(eta_gamma, decay_dist, rng):
    """Post-conversion track length [m]: conversion point drawn uniformly in [decay_dist, L_tracker]."""
    L = TRT_length(eta_gamma)
    U = rng.uniform(0, 1, len(eta_gamma))
    return np.where(L > decay_dist, U * (L - decay_dist), 0.0)

# ----------------------------------------------------------------------
# yields + cut-flow
# ----------------------------------------------------------------------
def n_events(sigma_pb_forward, lumi_fb):
    """Forward-hemisphere cross section [pb] -> event count in lumi_fb fb^-1 (x2 for both hemispheres)."""
    return 2.0 * sigma_pb_forward * lumi_fb * FB_TO_PB_INV

def print_yields(stages, lumi_fb):
    """stages = list of (label, N_events at lumi_fb). For the sample-based cut flows, where the
       weight already is an event yield (not a cross section). Prints step + cumulative factors."""
    hdr = f"{'stage':<32}{'N('+str(lumi_fb)+'/fb)':>16}{'step':>12}{'cumul':>12}"
    print(hdr)
    print("-" * len(hdr))
    n0 = stages[0][1]
    prev = n0
    for label, n in stages:
        step = n/prev if prev else 0.0
        cum  = n/n0 if n0 else 0.0
        print(f"{label:<32}{n:>16.3e}{step:>12.3e}{cum:>12.3e}")
        prev = n


def print_cutflow(stages, lumis=(312, 3000)):
    """stages = list of (label, sigma_forward_pb). Prints step/cumulative factors and event yields."""
    hdr = f"{'stage':<30}{'sigma_fwd[pb]':>15}{'step':>12}{'cumul':>12}"
    for L in lumis:
        hdr += f"{'N('+str(L)+'/fb)':>14}"
    print(hdr)
    print("-" * len(hdr))
    sig0 = stages[0][1]
    prev = sig0
    for label, sig in stages:
        step = sig/prev if prev else 0.0
        cum  = sig/sig0 if sig0 else 0.0
        row  = f"{label:<30}{sig:>15.3e}{step:>12.3e}{cum:>12.3e}"
        for L in lumis:
            row += f"{n_events(sig, L):>14.3e}"
        print(row)
        prev = sig

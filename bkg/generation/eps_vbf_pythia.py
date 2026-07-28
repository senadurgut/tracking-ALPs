"""
eps_VBF for the Run 3 K_L -> gamma gamma background, from a truth-level Pythia8 run.

WHAT THIS COMPUTES
------------------
The Run 3 VBF-parking trigger cuts on JETS (lead pT>105, sub pT>40, mjj>720, |deta|>3),
which FORESEE cannot give us (its tables are inclusive kaon spectra, no jets). We want

    eps_VBF = P( event fires the VBF trigger | event contains a forward K_L ).

Kaon production is soft and ~uncorrelated with the hard VBF jets, so by Bayes over
collisions weighted by K_L multiplicity:

    eps_VBF = P(VBF) * <n_KL>_VBF / <n_KL>_inel
            = (sigma_pass / sigma_inel) * (<n_KL>_VBF / <n_KL>_inel)

HardQCD(pThat>80) gives sigma_pass + <n>_VBF; SoftQCD gives sigma_inel + baseline <n>_inel.
Same-event VBF fake only; pileup accidental is separate/data-driven.

DUMPS
-----
--dump-kl (SoftQCD): one row per final K_L: (job, evt, pt, eta, phi, E, m, xProd, yProd, zProd).
    The unbiased inclusive K_L sample (FORESEE analog + baseline normalization).
--dump-vbf-kl (HardQCD): the BACKGROUND sample -- for each VBF-passing event, one row per
    final K_L carrying the two VBF jets, the K_L, and its two K_L->gg decay photons.
    Pythia leaves K_L stable (long-lived), so the gg decay is done analytically here
    (isotropic two-body, forced gg) -- everything upstream (jets, K_L momentum) is Pythia.

PARALLEL USE (SLURM): each task runs ONE mode with its own seed, writes --out-json; merge
with merge_eps_vbf.py.  INTERACTIVE: --mode both runs both and prints the full result.
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
import pythia8

# Pythia needs its XML data dir. Env activation sets PYTHIA8DATA, but a direct
# `python` call (e.g. from SLURM) does not -> derive it from the interpreter prefix.
XMLDIR = os.environ.get("PYTHIA8DATA") or os.path.join(sys.prefix, "share", "Pythia8", "xmldoc")

# --- VBF trigger cuts (config7 / analyze_configurable_v2.py) ---
LEAD_PT_CUT = 105.0   # GeV
SUB_PT_CUT  = 40.0    # GeV
MJJ_CUT     = 720.0   # GeV
DETA_CUT    = 3.0
# --- jet definition (CMS AK4) ---
JET_R       = 0.4
JET_PTMIN   = 0.0     # GeV, no jet-pT floor for now (VBF cuts on the 2 leading jets do the work)
JET_ETAMAX  = 4.7     # CMS jet acceptance incl. HF
# --- K_L counting ---
KL_ETA_MAX  = None    # None => no eta cut for now (count ALL K_L)
KL_ID       = 130     # K_L^0
BR_KL_GG    = 5.47e-4 # PDG Br(K_L -> gamma gamma)
MB_TO_PB    = 1.0e9   # 1 mb = 1e9 pb
# --- isolation discriminant (hadronic activity around the diphoton) ---
ISO_R       = 0.3     # cone dR around the K_L (= diphoton) direction
ISO_PTMIN   = 0.0     # GeV, min constituent pT summed (0 => all hadrons)


def make_pythia(process_lines, seed, ecm=13600.0):
    """Configure a Pythia instance for a given process (list of readString lines)."""
    p = pythia8.Pythia(XMLDIR, False)      # XMLDIR -> data files; False -> no startup banner
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eCM = {ecm}")
    for line in process_lines:
        p.readString(line)
    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {seed}")
    p.readString("Print:quiet = on")
    p.readString("Next:numberShowEvent = 0")
    p.readString("Next:numberShowInfo = 0")
    p.readString("Next:numberShowProcess = 0")
    p.init()
    return p


def count_KL(event):
    """Number of final-state K_L (id 130). Applies |eta|<KL_ETA_MAX only if it is set."""
    n = 0
    for prt in event:
        if prt.isFinal() and prt.id() == KL_ID:
            if KL_ETA_MAX is None or abs(prt.eta()) < KL_ETA_MAX:
                n += 1
    return n


def hadronic_iso(event, i_kl):
    """Sum pT of final-state hadrons within dR < ISO_R of the K_L at index i_kl (excluding the
       K_L itself; its gg decay photons are not hadrons, so they drop out automatically)."""
    eta0, phi0 = event[i_kl].eta(), event[i_kl].phi()
    iso = 0.0
    for j in range(event.size()):
        if j == i_kl:
            continue
        prt = event[j]
        if not (prt.isFinal() and prt.isHadron()) or prt.pT() < ISO_PTMIN:
            continue
        dphi = (prt.phi() - phi0 + np.pi) % (2.0 * np.pi) - np.pi
        if (prt.eta() - eta0)**2 + dphi**2 < ISO_R * ISO_R:
            iso += prt.pT()
    return iso


def kl_values(event):
    """(pt, eta, phi, E, m, xProd, yProd, zProd, iso) for each final-state K_L. Vertex in mm;
       iso = hadronic pT in a dR<ISO_R cone around the K_L (isolation discriminant)."""
    rows = []
    for i in range(event.size()):
        prt = event[i]
        if prt.isFinal() and prt.id() == KL_ID:
            if KL_ETA_MAX is None or abs(prt.eta()) < KL_ETA_MAX:
                rows.append((prt.pT(), prt.eta(), prt.phi(), prt.e(), prt.m(),
                             prt.xProd(), prt.yProd(), prt.zProd(),
                             hadronic_iso(event, i)))
    return rows


def decay_kl_photons(prt, rng):
    """Isotropic two-body K_L -> gamma gamma (forced), boosted to the lab along the real K_L.
       Returns ((pt1,eta1,phi1), (pt2,eta2,phi2)). Pythia leaves K_L stable, so we supply this."""
    M = prt.m()
    E_K = prt.e()
    pK = np.array([prt.px(), prt.py(), prt.pz()])
    beta = pK / E_K
    b2 = float(beta.dot(beta))
    gamma = E_K / M
    Estar = M / 2.0
    cth = rng.uniform(-1.0, 1.0)                       # isotropic (spin-0)
    sth = np.sqrt(1.0 - cth * cth)
    ph = rng.uniform(0.0, 2.0 * np.pi)
    p1s = Estar * np.array([sth * np.cos(ph), sth * np.sin(ph), cth])
    out = []
    for ps in (p1s, -p1s):                             # two photons back-to-back in rest frame
        bp = float(beta.dot(ps))
        fac = ((gamma - 1.0) * bp / b2 + gamma * Estar) if b2 > 0 else gamma * Estar
        plab = ps + fac * beta                         # general Lorentz boost
        pt = float(np.hypot(plab[0], plab[1]))
        eta = float(np.arcsinh(plab[2] / pt)) if pt > 0 else float(np.sign(plab[2]) * 99.0)
        phi = float(np.arctan2(plab[1], plab[0]))
        out.append((pt, eta, phi))
    return out[0], out[1]


def leading_dijet(sj):
    """(p0, p1, lead_pt, sub_pt, mjj, deta) for the 2 hardest jets, or None if <2 jets."""
    if sj.sizeJet() < 2:
        return None
    p0, p1 = sj.p(0), sj.p(1)
    return p0, p1, sj.pT(0), sj.pT(1), (p0 + p1).mCalc(), abs(p0.eta() - p1.eta())


def passes_vbf(dj):
    """Apply the 4 VBF cuts to a leading_dijet() tuple."""
    if dj is None:
        return False
    _, _, lead, sub, mjj, deta = dj
    return lead > LEAD_PT_CUT and sub > SUB_PT_CUT and mjj > MJJ_CUT and deta > DETA_CUT


def run(process_lines, n_events, seed, apply_vbf,
        dump_writer=None, vbf_writer=None, job_id=0, progress=0):
    """Generate n_events; return a dict of raw counters.
       dump_writer   -> soft inclusive K_L rows.
       vbf_writer    -> VBF-passing-event rows (jets + K_L + gg photons)."""
    p = make_pythia(process_lines, seed)
    sj = pythia8.SlowJet(-1, JET_R, JET_PTMIN, JET_ETAMAX, 2, 1)  # -1 = anti-kt, select=2 visible
    rng = np.random.default_rng(seed)      # for the analytic K_L->gg decays in the VBF dump
    n_acc = 0
    n_pass = 0
    sum_nKL_all = 0
    sum_nKL_pass = 0
    for i in range(n_events):
        if progress and i and i % progress == 0:
            print(f"  ... {i}/{n_events} events (n_pass={n_pass})", flush=True)
        if not p.next():
            continue
        if dump_writer is not None:
            rows = kl_values(p.event)
            nKL = len(rows)
            for r in rows:
                dump_writer.writerow((job_id, n_acc) + r)
        else:
            nKL = count_KL(p.event)
        sum_nKL_all += nKL
        if apply_vbf:
            sj.analyze(p.event)
            dj = leading_dijet(sj)
            if passes_vbf(dj):
                n_pass += 1
                sum_nKL_pass += nKL
                if vbf_writer is not None:
                    p0, p1, lead, sub, mjj, deta = dj
                    jet = (lead, p0.eta(), p0.phi(), p0.mCalc(),
                           sub,  p1.eta(), p1.phi(), p1.mCalc(), mjj, deta)
                    for k in range(p.event.size()):
                        prt = p.event[k]
                        if prt.isFinal() and prt.id() == KL_ID:
                            if KL_ETA_MAX is not None and abs(prt.eta()) >= KL_ETA_MAX:
                                continue
                            g1, g2 = decay_kl_photons(prt, rng)
                            vbf_writer.writerow((job_id, n_acc) + jet +
                                                (prt.pT(), prt.eta(), prt.phi(), prt.e()) + g1 + g2 +
                                                (hadronic_iso(p.event, k),))
        n_acc += 1
    p.stat()
    return {
        "sigma_gen_mb": p.infoPython().sigmaGen(),
        "n_acc":        n_acc,
        "n_pass":       n_pass,
        "sum_nKL_all":  sum_nKL_all,
        "sum_nKL_pass": sum_nKL_pass,
    }


def combine(hard_jobs, soft_jobs):
    """Aggregate lists of per-job counter dicts into the final numbers.
       Cross sections combined as event-count-weighted means (min-variance for MC)."""
    ngh = sum(j["n_acc"] for j in hard_jobs)
    nph = sum(j["n_pass"] for j in hard_jobs)
    sig_hard = sum(j["sigma_gen_mb"] * j["n_acc"] for j in hard_jobs) / ngh if ngh else 0.0
    sigma_pass = sum(j["sigma_gen_mb"] * j["n_pass"] for j in hard_jobs) / ngh if ngh else 0.0
    nKL_vbf = (sum(j["sum_nKL_pass"] for j in hard_jobs) / nph) if nph else float("nan")
    ngs = sum(j["n_acc"] for j in soft_jobs)
    sigma_inel = sum(j["sigma_gen_mb"] * j["n_acc"] for j in soft_jobs) / ngs if ngs else 0.0
    nKL_inel = (sum(j["sum_nKL_all"] for j in soft_jobs) / ngs) if ngs else float("nan")
    mult_ratio = nKL_vbf / nKL_inel if nKL_inel else float("nan")
    p_vbf = sigma_pass / sigma_inel if sigma_inel else float("nan")
    eps_vbf = p_vbf * mult_ratio
    rel_err = (nph ** -0.5) if nph else float("nan")
    return {
        "n_hard": ngh, "n_pass": nph, "n_soft": ngs,
        "sigma_hard_mb": sig_hard, "sigma_pass_mb": sigma_pass, "sigma_inel_mb": sigma_inel,
        "nKL_vbf": nKL_vbf, "nKL_inel": nKL_inel, "mult_ratio": mult_ratio,
        "p_vbf": p_vbf, "eps_vbf": eps_vbf, "eps_vbf_relerr": rel_err,
    }


def print_report(res):
    eta_note = "all eta" if KL_ETA_MAX is None else f"|eta|<{KL_ETA_MAX}"
    print("\n=== inputs ===")
    print(f"  hard events              : {res['n_hard']}   (VBF-passing: {res['n_pass']})")
    print(f"  soft events              : {res['n_soft']}")
    print(f"  sigma_pass               : {res['sigma_pass_mb']:.4e} mb")
    print(f"  sigma_inel               : {res['sigma_inel_mb']:.4e} mb")
    print(f"  <n_KL>_inel ({eta_note})  : {res['nKL_inel']:.4f}")
    print(f"  <n_KL>_VBF               : {res['nKL_vbf']:.4f}")

    print(f"\n=== baseline N = L x sigma x Br(K_L->gg)  ({eta_note}, no cuts) ===")
    for L in (312, 3000):
        N_KLgg = res["sigma_inel_mb"] * MB_TO_PB * L * 1000.0 * res["nKL_inel"] * BR_KL_GG
        print(f"  L={L:>4} fb^-1 : N(K_L->gg)={N_KLgg:.3e}   N(VBF K_L->gg)={N_KLgg*res['eps_vbf']:.3e}")

    print("\n=== eps_VBF ===")
    print(f"  multiplicity ratio <n>_VBF/<n>_inel   = {res['mult_ratio']:.3f}")
    print(f"  P(VBF) = sigma_pass/sigma_inel        = {res['p_vbf']:.4e}")
    print(f"  eps_VBF (same-event)                  = {res['eps_vbf']:.4e}"
          f"  (+- {res['eps_vbf_relerr']*100:.1f}% stat)")
    print("\nNOTE: same-event VBF fake only; pileup accidental is separate/data-driven.")


def _open_writer(path, header):
    fh = open(path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(header)
    return w, fh


SOFT_HEADER = ["job", "evt", "pt", "eta", "phi", "E", "m", "xProd_mm", "yProd_mm", "zProd_mm",
               "iso_had_pt"]
VBF_HEADER = ["job", "evt",
              "j1_pt", "j1_eta", "j1_phi", "j1_m",
              "j2_pt", "j2_eta", "j2_phi", "j2_m", "mjj", "deta",
              "kl_pt", "kl_eta", "kl_phi", "kl_E",
              "g1_pt", "g1_eta", "g1_phi", "g2_pt", "g2_eta", "g2_phi",
              "kl_iso_had_pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "hard", "soft"], default="both")
    ap.add_argument("--n-events", type=int, default=50000, help="events (single-mode)")
    ap.add_argument("--n-hard", type=int, default=30000, help="HardQCD events (both mode)")
    ap.add_argument("--n-soft", type=int, default=30000, help="SoftQCD events (both mode)")
    ap.add_argument("--pthatmin", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-json", type=str, default=None, help="write raw counters (parallel merge)")
    ap.add_argument("--dump-kl", type=str, default=None,
                    help="SoftQCD: CSV of every final K_L (inclusive sample).")
    ap.add_argument("--dump-vbf-kl", type=str, default=None,
                    help="HardQCD: CSV of VBF-passing events (jets + K_L + gg photons) = the background sample.")
    ap.add_argument("--progress", type=int, default=0, help="heartbeat every N events (0=off).")
    args = ap.parse_args()

    hard_lines = ["HardQCD:all = on", f"PhaseSpace:pTHatMin = {args.pthatmin}"]
    soft_lines = ["SoftQCD:inelastic = on"]

    if args.mode == "hard":
        vbf_w, vbf_fh = (_open_writer(args.dump_vbf_kl, VBF_HEADER) if args.dump_vbf_kl else (None, None))
        res = run(hard_lines, args.n_events, args.seed, apply_vbf=True,
                  vbf_writer=vbf_w, job_id=args.seed, progress=args.progress)
        if vbf_fh:
            vbf_fh.close()
        res.update(mode="hard", seed=args.seed, pthatmin=args.pthatmin)
        print(f"hard seed={args.seed}: n_acc={res['n_acc']} n_pass={res['n_pass']} "
              f"sigma_gen={res['sigma_gen_mb']:.4e} mb")

    elif args.mode == "soft":
        w, fh = (_open_writer(args.dump_kl, SOFT_HEADER) if args.dump_kl else (None, None))
        res = run(soft_lines, args.n_events, args.seed, apply_vbf=False,
                  dump_writer=w, job_id=args.seed, progress=args.progress)
        if fh:
            fh.close()
        res.update(mode="soft", seed=args.seed)
        print(f"soft seed={args.seed}: n_acc={res['n_acc']} "
              f"sigma_gen={res['sigma_gen_mb']:.4e} mb sum_nKL={res['sum_nKL_all']}")

    else:  # both -> full interactive result
        vbf_w, vbf_fh = (_open_writer(args.dump_vbf_kl, VBF_HEADER) if args.dump_vbf_kl else (None, None))
        hard = run(hard_lines, args.n_hard, args.seed, apply_vbf=True,
                   vbf_writer=vbf_w, job_id=args.seed, progress=args.progress)
        if vbf_fh:
            vbf_fh.close()
        w, fh = (_open_writer(args.dump_kl, SOFT_HEADER) if args.dump_kl else (None, None))
        soft = run(soft_lines, args.n_soft, args.seed + 1, apply_vbf=False,
                   dump_writer=w, job_id=args.seed + 1, progress=args.progress)
        if fh:
            fh.close()
        res = combine([hard], [soft])
        print_report(res)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

"""
Aggregate the per-task JSON counters from a parallel eps_VBF run (SLURM array)
into the final eps_VBF and the baseline N = L x sigma x Br.

Usage:  python merge_eps_vbf.py OUTDIR/json
        (OUTDIR/json holds hard_*.json and soft_*.json written by eps_vbf_pythia.py)

Kept independent of pythia8 so it runs on a login node with plain python.
"""
import glob
import json
import os
import sys

BR_KL_GG = 5.47e-4
MB_TO_PB = 1.0e9


def load(json_dir):
    hard, soft = [], []
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not files:
        sys.exit(f"no *.json found in {json_dir}")
    for fn in files:
        with open(fn) as f:
            j = json.load(f)
        (hard if j.get("mode") == "hard" else soft).append(j)
    return hard, soft, len(files)


def combine(hard_jobs, soft_jobs):
    ngh = sum(j["n_acc"] for j in hard_jobs)
    nph = sum(j["n_pass"] for j in hard_jobs)
    sig_hard   = sum(j["sigma_gen_mb"] * j["n_acc"] for j in hard_jobs) / ngh if ngh else 0.0
    sigma_pass = sum(j["sigma_gen_mb"] * j["n_pass"] for j in hard_jobs) / ngh if ngh else 0.0
    nKL_vbf = (sum(j["sum_nKL_pass"] for j in hard_jobs) / nph) if nph else float("nan")

    ngs = sum(j["n_acc"] for j in soft_jobs)
    sigma_inel = sum(j["sigma_gen_mb"] * j["n_acc"] for j in soft_jobs) / ngs if ngs else 0.0
    nKL_inel = (sum(j["sum_nKL_all"] for j in soft_jobs) / ngs) if ngs else float("nan")

    mult_ratio = nKL_vbf / nKL_inel if nKL_inel else float("nan")
    p_vbf = sigma_pass / sigma_inel if sigma_inel else float("nan")
    eps_vbf = p_vbf * mult_ratio
    rel_err = (nph ** -0.5) if nph else float("nan")
    return dict(n_hard=ngh, n_pass=nph, n_soft=ngs, sigma_hard_mb=sig_hard,
                sigma_pass_mb=sigma_pass, sigma_inel_mb=sigma_inel, nKL_vbf=nKL_vbf,
                nKL_inel=nKL_inel, mult_ratio=mult_ratio, p_vbf=p_vbf,
                eps_vbf=eps_vbf, eps_vbf_relerr=rel_err)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python merge_eps_vbf.py OUTDIR/json")
    hard, soft, nfiles = load(sys.argv[1])
    res = combine(hard, soft)

    print(f"merged {nfiles} json files : {len(hard)} hard + {len(soft)} soft tasks")
    print("\n=== inputs (all tasks combined) ===")
    print(f"  hard events              : {res['n_hard']:,}   (VBF-passing: {res['n_pass']:,})")
    print(f"  soft events              : {res['n_soft']:,}")
    print(f"  sigma(HardQCD)           : {res['sigma_hard_mb']:.4e} mb")
    print(f"  sigma_pass               : {res['sigma_pass_mb']:.4e} mb")
    print(f"  sigma_inel               : {res['sigma_inel_mb']:.4e} mb")
    print(f"  <n_KL>_inel              : {res['nKL_inel']:.4f}")
    print(f"  <n_KL>_VBF               : {res['nKL_vbf']:.4f}")

    print("\n=== baseline N = L x sigma x Br(K_L->gg)  (all eta, no cuts) ===")
    for L in (312, 3000):
        N_coll = res["sigma_inel_mb"] * MB_TO_PB * L * 1000.0
        N_KL   = N_coll * res["nKL_inel"]
        N_KLgg = N_KL * BR_KL_GG
        print(f"  L={L:>4} fb^-1 : N_coll={N_coll:.3e}  N_KL={N_KL:.3e}  N(K_L->gg)={N_KLgg:.3e}")

    print("\n=== N(VBF K_L->gg) = N(K_L->gg) x eps_VBF  (production level, all eta, no cuts) ===")
    for L in (312, 3000):
        N_KLgg     = res["sigma_inel_mb"] * MB_TO_PB * L * 1000.0 * res["nKL_inel"] * BR_KL_GG
        N_vbf_KLgg = N_KLgg * res["eps_vbf"]
        print(f"  L={L:>4} fb^-1 : N(VBF K_L->gg) = {N_vbf_KLgg:.3e}")

    print("\n=== eps_VBF ===")
    print(f"  multiplicity ratio <n>_VBF/<n>_inel   = {res['mult_ratio']:.3f}")
    print(f"  P(VBF) = sigma_pass/sigma_inel        = {res['p_vbf']:.4e}")
    print(f"  eps_VBF (same-event)                  = {res['eps_vbf']:.4e}"
          f"  (+- {res['eps_vbf_relerr']*100:.2f}% stat)")
    print("\nNOTE: same-event VBF fake only; pileup accidental is separate/data-driven.")


if __name__ == "__main__":
    main()

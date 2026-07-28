# K_L -> gamma gamma background

Residual K_L -> gamma gamma background for the CMS tracking-ALP search, computed on the
**same cut flow** as the signal (config7 = Run 3 parking, config10 = Phase-2 scouting), from
Pythia8 K_L samples. Shares the signal physics via `../analysis/module_VBF.py`.

## Layout
```
bkg/
  analysis/     kl_common.py   shared physics (decay, conversion, in-tracker, TRT wrappers)
                run3_parking.py    Run 3 (config7) cut-flow yield on the VBF/hard sample
                phase2_scouting.py Phase-2 (config10) cut-flow yield on the inclusive/soft sample
  generation/   eps_vbf_pythia.py  Pythia8 generator (soft + VBF K_L, + hadronic-isolation column)
                merge_eps_vbf.py   merge per-task JSON counters -> eps_VBF
                concat_kl_csv.sh   concatenate per-task CSV chunks (strips repeated headers)
  plotting/     plot_kl_iso.py     hadronic-activity (isolation) distributions, run3 vs phase2
                plot_sig_vs_bkg.py signal vs K_L in mass/pT/displacement, full selection + activity cut
                alp_kl_analysis.py consolidated plotter (supersedes the notebooks)
  slurm/        submit_*.slurm  (WORKDIR = this dir; scripts called as <subdir>/<name>.py)
  notebooks/    exploratory .ipynb + displacement.py (reference)
  data/     (gitignored) kl_sample.csv, kl_vbf_sample.csv, out/, *.npy
  archive/  (gitignored) old FORESEE-spectrum code + session artifacts
```
Scripts anchor their data paths to this `bkg/` dir via `__file__`, so they can be launched
from any working directory.

## Pipeline  (env python `/home/export/sdurgut/micromamba/envs/foresee310/bin/python`)
1. **Generate** (SLURM): `sbatch slurm/submit_eps_vbf.slurm` -> per-task CSV/JSON in `data/out/`.
   Concatenate: `generation/concat_kl_csv.sh data/out/vbfcsv data/kl_vbf_sample.csv` (and the soft one).
2. **Cut-flow yields**: `sbatch slurm/submit_run3.slurm`, `sbatch slurm/submit_phase2.slurm`.
3. **Plots**: `sbatch slurm/submit_sig_vs_bkg.slurm` (mass/pT/displacement),
   `python plotting/plot_kl_iso.py` (isolation).

## Current results (validated cut flows)
- Run 3 parking:  N_KL ~ 3.96e5 @ 312 fb^-1
- Phase-2 scouting: N_KL ~ 2.18e10 @ 3000 fb^-1  (1.82e9 @ 250 fb^-1)
- Baselines match the cross-checks (1.93e8 / 5.4e13 @ 312). K_L lives only at m_gg = 0.498 GeV.
- Hadronic-activity (isolation) cut: modest handle for Run 3 (~factor 2-3), weak for Phase-2.

## Normalization
- w_hard = (sigma_hard_pb / N_HARD) * L * 1000 * Br,  sigma_hard_pb = 3.6413e-3*1e9, **N_HARD = 20e6**
  (all generated hard events -> VBF baked in; do NOT re-apply mask_vbf/eps_VBF).
- w_soft = (78.585e9 / 2e6) * L * 1000 * Br,  Br(K_L->gg) = 5.47e-4.

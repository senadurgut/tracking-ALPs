# Plot configs for `plot_sig_vs_bkg.py`

Each JSON here drives one signal-vs-K_L plot: every cut is an independent **on/off toggle with a
value**. Copy a `*_nominal.json`, flip toggles, and run the plotter to get a new plot. The config
schema extends `configs/config7.json`; the plotter (`bkg/plotting/plot_sig_vs_bkg.py`) reads it via
`load_cuts` (a self-contained mirror of `analyze_configurable_v2.resolve_cuts` + the extra toggles).

## Run
```
cd tracking_ALPs/bkg
PY=/home/export/sdurgut/micromamba/envs/foresee310/bin/python
$PY plotting/plot_sig_vs_bkg.py --analysis run3                       # shortcut -> run3_nominal.json
$PY plotting/plot_sig_vs_bkg.py --config configs/run3_novbf.json --max-kl 0
```
Output: `bkg/plots/sig_vs_bkg_<configstem>.png` (one per config; distinct name, never overwrites).
`--max-kl 0` reads every K_L row (exact yields; stream in chunks, add `--chunk 100000` if RAM tight);
a positive `--max-kl` subsamples the K_L (fast, shapes only — the panel flags `[SUBSAMPLE]`).

## Schema (all keys optional except `era`)
| key | meaning | default if absent |
|---|---|---|
| `era` | `run3` or `phase2` — fixes the K_L sample + normalization | required |
| `analysis_mode` | `parking` / `scouting` — only sets the convert/sep/disp defaults | `parking` |
| `lumi` | integrated lumi [fb^-1] for the yields | era default (312 / 3000) |
| `tracker_decay_cut` `{value}` | require K_L/ALP to decay inside the tracker | `true` |
| `mask_eta` `{value, eta_range}` | both photons in `eta_range` | off, `[0,0]` |
| `mask_vbf` `{value, lead_pt, sub_pt, mjj, deta}` | VBF dijet cut (**signal only**; see caveats) | off |
| `mask_merge` `{value, delta_r_max}` | merge dR cut (run3: `<cell` needs both-convert if `use_tracks`) | on, 0.3 |
| `use_tracks` (top-level bool) | run3 merged bin recovered via conversion | `true` |
| `mask_convert` `{value}` | require both photons to convert | `analysis_mode=='scouting'` |
| `mask_separation` `{value}` | `separation_TRT >= sep_cut` (needs tracks) | `analysis_mode=='scouting'` |
| `mask_displacement` `{value}` | `displaced_vertex_TRT >= disp_cut` (needs tracks) | `analysis_mode=='scouting'` |
| `phase2_cuts` `{sep_cut, disp_cut, track_resolution}` | values for sep/disp + b~ track resolution | 5e-4 / 1e-1 / 4e-5 |
| `activity_cut` `{value, cut_value}` | hadronic activity `<= cut_value` (**K_L only**) | off |
| `alp_pT_cut` `{value, cut_value}` | diphoton `pT > cut_value` | off |
| `signal` `{masses, gagg}` | signal masses + per-mass benchmark coupling | `[0.3,0.5,1.0]` / bench |

An unextended `configs/config7.json` / `config10.json` reproduces the original selection (the
convert/sep/disp toggles default from `analysis_mode`).

## Caveats (inherent to the samples, not bugs)
- **`era` fixes the K_L sample.** run3 → VBF-triggered *hard* sample (VBF is **baked in**); turning
  `mask_vbf` off affects **only the signal**, the K_L keeps VBF. phase2 → inclusive *soft* sample.
- **`activity_cut` affects only the K_L.** The signal MC is parton-level, so its hadronic activity
  is ~0 by construction (always passes).
- separation/displacement need converted-photon tracks, so turning either on auto-applies both-convert.

## Example configs here
- `run3_nominal.json` / `phase2_nominal.json` — reproduce config7 / config10 (the `--analysis` shortcuts).
- `run3_novbf.json` — VBF off (signal); shows the VBF handle's effect on the signal pT shape.
- `run3_pt60.json` — diphoton `pT > 60 GeV` on (the extra K_L-suppressing handle).
- `phase2_no_disp.json` — displacement cut off (keep separation), to isolate its effect.

# VBF ALP → γγ Analysis Code

Analysis code for studying axion-like particle (ALP) production via vector-boson fusion (VBF) at the LHC, where the ALP decays to a collimated pair of photons inside the tracking detector.

The code was developed for the ATLAS detector geometry but is designed to be adaptable to other detectors by changing a small set of geometry constants.

---

## Physics overview

The process studied is:

```
pp → qqγγ   (via  qq → qq + a,  a → γγ)
```

The ALP has a finite lifetime and decays at a displaced vertex inside the tracker.  The two photons from the decay are highly collimated (small ΔR) and each may convert to an e⁺e⁻ pair in the tracker material.  The analysis selects events where:

1. The diphoton system has high transverse momentum (`pT > pT_cut`).
2. **Both** photons convert inside the TRT (transition radiation tracker).
3. The two conversion tracks are spatially resolved (separation > `sep_resolution`).
4. The reconstructed conversion vertex is significantly displaced from the IP (impact parameter > `vertex_displacement`).
5. The two photons are collimated enough to appear as a single isolated photon in the ECAL (ΔR < `DeltaR_max`).

The output is the number of signal events passing all cuts as a function of the ALP mass `m_a` and coupling `g_{aγγ}`, which is used to derive sensitivity contours in the `(m_a, g_{aγγ})` parameter space.

---

## Repository structure

```
├── module_VBF.py              # Core physics library (detector sim, kinematics)
├── Analyze_madgraph_output.py # Main analysis script (reads data, applies cuts, writes results)
├── lhe_to_csv.py              # Preprocessing: converts Madgraph LHE → CSV
├── Make_parameter_space_plots.ipynb  # Notebook: sensitivity contour plots
├── Plot_distributions.ipynb          # Notebook: kinematic distribution plots
├── data/                      # Place your converted CSV files here
│   └── <ma_name>.csv
└── results/                   # Output CSVs (created automatically)
    ├── params.csv
    └── results_vbf_*.csv
```

---

## Installation

```bash
git clone <this-repo>
cd <repo>
pip install numpy scipy sympy pandas matplotlib
```

Python ≥ 3.8 is required. No other non-standard dependencies are needed.

### MadGraph (required for Step 1)

This repository does **not** generate events by itself. Install MadGraph5_aMC@NLO separately, for example:

```bash
cd /path/to/ALP
mkdir -p tools
wget -O tools/MG5_aMC_v3.5.13.tar.gz https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.5.13.tar.gz
tar -xzf tools/MG5_aMC_v3.5.13.tar.gz -C tools
```

You also need a Fortran compiler (`gfortran`) available in your `PATH` for event generation.

Install `gfortran` (examples):

```bash
# macOS (Homebrew)
brew install gcc

# Ubuntu/Debian
sudo apt-get install gfortran

# verify
which gfortran && gfortran --version
```

Quick check:

```bash
cd /path/to/ALP/tools/MG5_aMC_v3_5_13
./bin/mg5_aMC --help
```

---

## Quick start

### Step 1 — Run Madgraph

Events were generated at parton level with **MadGraph5_aMC@NLO 3.5.13**
and the `ALP-h_UFO` model. The matrix-element process is electroweak
VBF-like ALP production with two jets, followed by the ALP decay to two
photons:

```text
import model ALP-h_UFO --modelname
define j = u c d s b u~ c~ d~ s~ b~ g
generate p p > j j ax QCD=0, ax > a a
output CMS
```
`QCD=0` selects the electroweak production contribution. Because the decay
`ax > a a` is included in the matrix element, the LHE event record contains
the ALP (PDG ID 36) as an intermediate particle with status 2 and the two
photons (PDG ID 22) as final-state particles with status 1.

The principal run-card settings are:

| Setting | Value |
|---------|-------|
| `run_tag` | `tag_1` |
| `nevents` | 10,000 unweighted events per run |
| `iseed` | 0 (MadGraph assigns the seed automatically) |
| Beam particles | proton–proton |
| `ebeam1`, `ebeam2` | 6800 GeV each (`sqrt(s) = 13.6 TeV`) |
| `pdlabel` | `nn23lo1` |
| `lhaid` | 230000 |
| Renormalization/factorization scales | dynamic (`fixed_ren_scale = False`, `fixed_fac_scale = False`) |
| `dynamical_scale_choice` | -1 |
| `scalefact` | 1.0 |
| `event_norm` | `average` |
| `gridpack` | `False` |
| `nhel` | 0 |
| `sde_strategy` | 1 |
| `bwcutoff` | 15.0 |
| `ptj`, `pta` | 0.0 GeV |
| `etaj` | 5.0 |
| `etaa` | 0.0 |
| `drjj`, `draa`, `draj` | 0.0 |
| `mmjj`, `mmaa` | 0.0 GeV |
| `xptj`, `xpta` | 0.0 GeV |
| `xetamin`, `deltaeta` | 0.0 |
| `maxjetflavor` | 4 |
| `cut_decays` | `False` |
| Systematics | enabled with \(\mu_R,\mu_F = 0.5,1,2\) and PDF error sets |

No generator-level transverse-momentum, photon-isolation, pair-mass, or
minimum-\(\Delta R\) cuts are imposed by this card. In addition,
`cut_decays = False` means generic cuts are not applied to particles from the
specified ALP decay chain. The only finite rapidity setting is `etaj = 5.0`.

The ALP parameters are set in `param_card.dat`:

```text
Block mass
   36 <m_a in GeV>  # Max

Block effcouplings
    5 1.000000e+00  # gaa
    7 1.000000e+02  # fs

DECAY 36 Auto       # Wax
```

The width must be recomputed after changing the mass. `DECAY 36 Auto` lets
MadGraph calculate the corresponding ALP width at generation time.
Note that a reference value of coupling is used for each mass so the cross section must be scaled after generation, according to the equation: \n
$$\sigma(pp\to a\,jj) \approx 180~\mathrm{pb}
\left(\frac{\gagg}{10^{-2}\GeV^{-1}}\right)^{2}$$ \n 


Cards for a single mass point are provided as an example inside /model/Cards.

#### Mass grid and event batches

The generated ALP mass points are:

```text
0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0, 100.00 GeV
```


### Step 2 — Convert LHE files to CSV

```bash
python lhe_to_csv.py Events/run_01/unweighted_events.lhe.gz data/0_1000GeV.csv
```

This produces `data/0_1000GeV.csv` in the format expected by the analysis.
Use `--max-events N` to limit the number of events for testing.

**Filename convention:** CSV filenames must match the output of `ma_to_name` in
`Analyze_madgraph_output.py`, which formats masses to exactly 4 decimal places
with dots replaced by underscores:

| m_a (GeV) | Filename stem |
|-----------|--------------|
| 0.01 | `0_0100GeV` |
| 0.1  | `0_1000GeV` |
| 1.0  | `1_0000GeV` |
| 10.0 | `10_0000GeV` |

**Output format:** Each event occupies exactly 7 semicolon-delimited rows (no header).
Each row contains `E,px,py,pz` (comma-separated, in GeV).
The layout assumed by `lhe_to_csv.py` is:

| Row | Particle |
|-----|----------|
| 0 | Incoming quark 1 |
| 1 | Incoming quark 2 |
| 2 | VBF jet 1 |
| 3 | VBF jet 2 |
| 4 | **ALP** |
| 5 | **Photon 1** (from a → γγ) |
| 6 | **Photon 2** (from a → γγ) |

> **Verify the row ordering before a full run.**
> The row indices are set by `ROW_ALP`, `ROW_G1`, `ROW_G2` at the top of
> `module_VBF.py` (defaults: 4, 5, 6).  If your CSV was produced by a
> different script, these may differ.  Use the built-in helper to check:
>
> ```python
> from module_VBF import print_first_event
> print_first_event('0_1000GeV', data_dir='data')
> ```
>
> The ALP row should have the highest energy among rows 4-6 and satisfy
> `E ~= sqrt(px^2+py^2+pz^2+ma^2)`.  If the ordering looks wrong, adjust
> `ROW_ALP`, `ROW_G1`, `ROW_G2` in `module_VBF.py` accordingly.

### Step 3 — Run the analysis

The analysis is config-driven. `analysis/analyze_configurable_v2.py` is the
single source of truth for the physics; `analysis/analyze_configurable_parallel.py`
imports its logic and adds per-mass / SLURM-array support. The mass grid and cut
parameters live in the config (`ma_list`, `gagg_list`, etc.).

Full sweep (all masses, one process):

```bash
python analysis/analyze_configurable_v2.py --config configs/config8.json
```

Parallel over masses (one SLURM array task per mass), then merge the parts:

```bash
sbatch --array=0-30 --export=ALL,CONFIG=configs/config8.json scripts/slurm/run_array.sbatch
python scripts/analysis_helpers/merge_parts.py --config configs/config8.json
```

Set `--array` to `len(ma_list) - 1`. Output goes to `results/run_<stem>/results_<stem>.csv`;
if that dir already exists a `_NN` suffix is added (e.g. `run_config8_01`). For a
suffixed run, point the merge at the same dir printed under `run-dir:` in the log:

```bash
python3 scripts/analysis_helpers/merge_parts.py --config configs/config8.json --run-dir results/run_config8_01
```

### Step 4 — Make plots

Open `Make_parameter_space_plots.ipynb` in Jupyter and update the file paths at
the top of the notebook to point to your results CSVs.

---

## Configuring detector parameters

All detector geometry lives in the top section of `module_VBF.py`.  To adapt
the code for a detector other than ATLAS, edit the following blocks:

### TRT / tracker geometry

```python
# TRT barrel
z_max_TRT        = 0.72    # m – half-length of barrel active volume
R_min_TRT        = 0.56    # m – inner radius
R_max_TRT        = 1.08    # m – outer radius

# TRT endcap
z_min_TRT_endcap = 0.827   # m
z_max_TRT_endcap = 2.774   # m
R_min_TRT_endcap = 0.617   # m
R_max_TRT_endcap = 1.106   # m

# ECAL
R_ECAL = 1.8               # m – effective ECAL radius for ΔR correction
```

### Photon conversion fractions

The `conv_fr` array in `module_VBF.py` encodes the photon conversion
probability as a function of |η|.  The current values come from the ATLAS
measurement in [arXiv:1810.05087](https://arxiv.org/abs/1810.05087).

Replace this array with data from your own detector to adapt the simulation.
The format is:

```python
conv_fr = np.array([
    [eta_bin_edge_1, eta_bin_edge_2, ..., np.inf],  # upper |η| bin edges
    [f_total_1, f_total_2, ...],                    # fraction of all photons reco'd as converted
    [f_fake_1,  f_fake_2,  ...],                    # fake-conversion rate (true unconverted → reco converted)
    [f_reco_1,  f_reco_2,  ...],                    # true-conversion efficiency
])
```

### Cut parameters

Edit the block at the top of `Analyze_madgraph_output.py`:

```python
pT_cut_value         = 150     # GeV  – minimum diphoton pT
TRT_track_resolution = 2.0e-4  # m    – tracker angular resolution
vertex_displacement  = 1.0e-1  # m    – minimum displaced-vertex impact parameter
TRT_sep_resolution   = 5.0e-4  # m    – minimum track separation
DeltaR_max           = np.sqrt(0.025**2 + 0.0245**2)  # ECAL isolation cone
```

---

## Production cross-sections

The Madgraph cross-sections for VBF ALP production are **not** stored in this
repository (they depend on your PDF choice and Madgraph version).  To reproduce
the sensitivity plots in `Make_parameter_space_plots.ipynb`, you need to
provide:

- `xsec_gagg1e2_list`: production cross-section (pb) at `g_{aγγ} = 0.01 GeV^{-1}`
  for each ALP mass in `ma_list`.  The cross-section scales as `g_{aγγ}²`.

Update the corresponding list at the top of the notebook with values from your
own Madgraph runs.

---

## Existing constraints data

The `Make_parameter_space_plots.ipynb` notebook overlays existing experimental
limits from LEP, CDF, ATLAS Pb+Pb, CMS Pb+Pb, Belle-II, and NA64.  These are
read from CSV/text files under `data/existing_constraints/`.  Providing these
files is optional — comment out the relevant cells in the notebook if you do
not have them.

---

## Citation

If you use this code, please cite the paper for which it was originally
developed:

> [Add your paper reference here]

---

## License

[Add your license here]

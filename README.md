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

Quick check:

```bash
cd /path/to/ALP/tools/MG5_aMC_v3_5_13
./bin/mg5_aMC --help
```

---

## Quick start

### Step 1 — Run Madgraph

Generate VBF→ALP→γγ events for each ALP mass using your Madgraph UFO model.
The code expects the ALP PDG ID to be `9000005` (edit `ALP_PDG_ID` in `lhe_to_csv.py` if yours differs).

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

The default mass grid is 32 points log-spaced from 0.01 to 10 GeV
(`np.logspace(-2, 1, num=32)`), corresponding to indices 0–31.

```bash
# Analyse a single mass, e.g. index 9
python Analyze_madgraph_output.py 9
```

Results are written to `results/results_vbf_*.csv`, and a `results/params.csv`
file is written (overwriting any previous one) recording the cut parameters
used. All masses in a batch must be run with the same parameters.
To run all masses in parallel on a cluster (e.g. with Slurm):

```bash
for i in $(seq 0 31); do sbatch --wrap="python Analyze_madgraph_output.py $i"; done
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

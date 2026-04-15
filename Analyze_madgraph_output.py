"""
Analyze_madgraph_output.py
==========================
Command-line script that reads pre-processed Madgraph VBF→ALP→γγ events,
applies a cascade of detector selection cuts, and writes the per-cut event
counts to CSV files in ``results/``.

Usage
-----
    python Analyze_madgraph_output.py <mass_index>

where <mass_index> is an integer index into ``ma_list`` (0-based).

Example
-------
    # Analyse events for m_a = 0.1 GeV  (index 9 in ma_list)
    python Analyze_madgraph_output.py 9

Output
------
The following files are written inside the ``results/`` directory:

  params.csv                   – cut parameters used in this run (overwritten each run)
  results_vbf_total.csv        – all events passing pT cut with 2 converted photons
  results_vbf_separated.csv    – additionally requiring track separation > sep_resolution
  results_vbf_displaced.csv    – additionally requiring impact parameter > vertex_displacement
  results_vbf_isolated.csv     – additionally requiring Delta-R < DeltaR_max

The four results CSVs have one header row of g_agg values, then one row per
ALP mass with the event counts for each g_agg point.  The params CSV is
overwritten on every run — all masses in a batch should be run with the same
cut parameters.

Input data
----------
Events for each ALP mass are read from::

    data/<ma_name>.csv

where <ma_name> is a string like ``'01GeV'`` for m_a = 0.1 GeV.
See ``lhe_to_csv.py`` for generating these files from Madgraph LHE output.

Directory layout expected
-------------------------
    project_root/
    ├── Analyze_madgraph_output.py   (this script)
    ├── module_VBF.py
    ├── lhe_to_csv.py
    ├── data/                        # input CSVs from lhe_to_csv.py
    │   └── <ma_name>.csv
    └── results/                     # output CSVs (created automatically)
"""

################################################
## Load packages
################################################

import numpy as np
import pandas as pd
import os
import sys

from module_VBF import (
    read_data,
    raw_to_events,
    calculate_separations_2converted_displaced_isolated,
)

################################################
## ALP mass grid
## ─────────────────────────────────────────────
## Add or remove masses here to match your Madgraph runs.
## ma_name must match the filename stem in data/<ma_name>.csv.
################################################

ma_list= [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 5., 10.]
def ma_to_name(ma):
    """Convert an ALP mass (GeV, float) to the filename stub used in data/."""
    return f"{ma:.4f}GeV".replace('.', '_')

# Parse the mass index from the command line
ma_value = ma_list[round(float(sys.argv[1]))]
ma_name  = ma_to_name(ma_value)

################################################
## g_agg coupling grid
## ─────────────────────────────────────────────
## 32 values log-spaced from 1e-7 to 1e-2 GeV^{-1}.
## Adjust if you want a different range or density.
################################################

gagg_list = np.logspace(-7, -2, num=32)

################################################
## Number of Monte Carlo events
## ─────────────────────────────────────────────
## num_events: how many events to read from the CSV (set to match your sample).
## max_events: legacy variable (not used in current analysis logic).
################################################

num_events = 1_000_000

################################################
## Cut parameters
## ─────────────────────────────────────────────
## Edit these to match your analysis scenario.
################################################

# Minimum ALP (= diphoton) transverse momentum (GeV).
# Because the two photons are very collimated, the pT cut is applied on the ALP.
pT_cut_value = 150    # GeV

# Tracker angular resolution for determining photon track directions (metres).
TRT_track_resolution = 2.0e-4  # m = 0.2 mm

# Minimum displaced-vertex impact parameter required to tag a decay as displaced (metres).
vertex_displacement  = 1.0e-1  # m = 10 cm

# Minimum track separation required to resolve the two photon tracks (metres).
TRT_sep_resolution = 5.0e-4  # m = 0.5 mm

# Maximum Delta-R between the two photons for the pair to pass the ECAL isolation criterion.
# Uncomment the desired definition:
DeltaR_max = np.sqrt(0.025**2 + 0.0245**2)   # ECAL cell size (~0.035)
# DeltaR_max = np.sqrt(0.075**2 + 0.123**2)  # ECAL L1 granularity (~0.14)



################################################
## Make sure the output directory exists
################################################

os.makedirs('results', exist_ok=True)

################################################
## Write parameters to file
## ─────────────────────────────────────────────
## A params.csv is written (overwriting any previous one) each time the
## script runs, so the results folder always documents the settings used.
## All masses in a batch must be run with the same parameters.
################################################

params = {
    'run_kind':            'analysis_run',
    'pT_cut':              pT_cut_value,
    'track_res':           TRT_track_resolution,
    'sep_res':             TRT_sep_resolution,
    'vertex_displacement': vertex_displacement,
    'Delta_R':             DeltaR_max,
}
pd.DataFrame.from_dict(params, orient='index').to_csv(
    'results/params.csv', header=False
)

################################################
## Analysis pipeline
################################################

print(f'Reading data for m_a = {ma_value} GeV  ({ma_name}) ...')
raw_events = read_data(ma_name, num=num_events, data_dir='data')

print('Converting to event objects ...')
events = raw_to_events(raw_events, gaggs=gagg_list, ma=ma_value)
del raw_events   # free memory

print('Calculating separations ...')
(separations,
 average_etas,
 impact_parameters,
 delta_Rs,
 delta_etas,
 ltracks,
 _event_indices) = calculate_separations_2converted_displaced_isolated(
    events,
    gaggs=gagg_list,
    pTcut=pT_cut_value,
    track_resolution=TRT_track_resolution,
    check=True,          # also return ltracks and event indices for potential debugging
)

# ── Step 1: remove entries flagged as outside TRT acceptance (value = -1) ──
# Note: ltracks is not filtered here — it stores 2 values per event (one per
# photon) so its shape differs from the other arrays. It is available via
# _event_indices above if needed for debugging.
average_etas      = [average_etas[i][separations[i] != -1]      for i in range(len(separations))]
impact_parameters = [impact_parameters[i][separations[i] != -1] for i in range(len(separations))]
delta_Rs          = [delta_Rs[i][separations[i] != -1]          for i in range(len(separations))]
delta_etas        = [delta_etas[i][separations[i] != -1]        for i in range(len(separations))]
separations       = [x[x != -1] for x in separations]

# ── Step 2: require track separation > TRT_sep_resolution ──
mask_sep                    = [separations[i] > TRT_sep_resolution      for i in range(len(separations))]
average_etas_separated      = [average_etas[i][mask_sep[i]]      for i in range(len(separations))]
impact_parameters_separated = [impact_parameters[i][mask_sep[i]] for i in range(len(separations))]
delta_Rs_separated          = [delta_Rs[i][mask_sep[i]]          for i in range(len(separations))]
delta_etas_separated        = [delta_etas[i][mask_sep[i]]        for i in range(len(separations))]
separations_separated       = [separations[i][mask_sep[i]]       for i in range(len(separations))]

# ── Step 3: require displaced vertex > vertex_displacement ──
mask_disp                   = [impact_parameters_separated[i] > vertex_displacement for i in range(len(separations_separated))]
average_etas_displaced      = [average_etas_separated[i][mask_disp[i]]      for i in range(len(separations_separated))]
separations_displaced       = [separations_separated[i][mask_disp[i]]       for i in range(len(separations_separated))]
delta_Rs_displaced          = [delta_Rs_separated[i][mask_disp[i]]          for i in range(len(separations_separated))]
delta_etas_displaced        = [delta_etas_separated[i][mask_disp[i]]        for i in range(len(separations_separated))]
impact_parameters_displaced = [impact_parameters_separated[i][mask_disp[i]] for i in range(len(separations_separated))]

# ── Step 4: require Delta-R < DeltaR_max (ECAL isolation) ──
mask_DR                    = [delta_Rs_displaced[i] < DeltaR_max for i in range(len(delta_Rs_displaced))]
average_etas_isolated      = [average_etas_displaced[i][mask_DR[i]]      for i in range(len(delta_Rs_displaced))]
separations_isolated       = [separations_displaced[i][mask_DR[i]]       for i in range(len(delta_Rs_displaced))]
impact_parameters_isolated = [impact_parameters_displaced[i][mask_DR[i]] for i in range(len(delta_Rs_displaced))]
delta_etas_isolated        = [delta_etas_displaced[i][mask_DR[i]]        for i in range(len(delta_Rs_displaced))]
delta_Rs_isolated          = [delta_Rs_displaced[i][mask_DR[i]]          for i in range(len(delta_Rs_displaced))]

# Uncomment the block below to apply a Delta-eta cut instead of Delta-R:
# mask_Deta                    = [delta_etas_displaced[i] < DeltaR_max for i in range(len(delta_etas_displaced))]
# average_etas_isolated        = [average_etas_displaced[i][mask_Deta[i]]      for i in range(len(delta_etas_displaced))]
# separations_isolated         = [separations_displaced[i][mask_Deta[i]]       for i in range(len(delta_etas_displaced))]
# impact_parameters_isolated   = [impact_parameters_displaced[i][mask_Deta[i]] for i in range(len(delta_etas_displaced))]
# delta_Rs_isolated            = [delta_Rs_displaced[i][mask_Deta[i]]          for i in range(len(delta_etas_displaced))]
# delta_etas_isolated          = [delta_etas_displaced[i][mask_Deta[i]]        for i in range(len(delta_etas_displaced))]

################################################
## Write results to CSV
################################################

base = 'results/results_vbf'

# Write a header row with g_agg values the first time this mass grid is used
for suffix in ('_total', '_separated', '_displaced', '_isolated'):
    path = base + suffix + '.csv'
    if not os.path.exists(path):
        pd.DataFrame({'g_agg': gagg_list}).T.to_csv(path, mode='w', index=True, header=False)

def _write_row(path, label, counts):
    pd.DataFrame({label: counts}).T.to_csv(path, mode='a', index=True, header=False)

print('Writing results ...')
_write_row(base + '_total.csv',     ma_name, np.array([len(x) for x in separations]))
_write_row(base + '_separated.csv', ma_name, np.array([len(x) for x in separations_separated]))
_write_row(base + '_displaced.csv', ma_name, np.array([len(x) for x in separations_displaced]))
_write_row(base + '_isolated.csv',  ma_name, np.array([len(x) for x in separations_isolated]))

print(f'Done. Results written to {base}_*.csv')

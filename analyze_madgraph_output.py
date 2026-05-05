"""
Analyze_madgraph_output.py
==========================
Command-line script that reads pre-processed Madgraph VBF→ALP→γγ events,
applies a cascade of detector selection cuts, and writes the per-cut event
counts to CSV files in ``results/``.

Usage
-----
    python Analyze_madgraph_output.py <mass | index> [--results-file STEM]

``mass`` can be:

  - A MadGraph-style token with ``p`` as the decimal point, e.g. ``0p01`` for
    0.01 GeV (matches merged CSV stems via the usual four-digit formatting).
  - Optionally the full merged stem: ``merged_ma_0p0100GeV``.
  - Or an integer **index** into ``ma_list`` (0-based), e.g. ``9`` for 0.1 GeV
    on the default grid — only used when the argument does not look like a mass
    token (no ``p`` and not starting with ``merged_ma_``).

CSVs are always written under the results directory: ``TRACKING_ALP_RESULTS_DIR``
if set, else ``<repo>/results``.

Example
-------
    # By index (0.1 GeV is index 9 on the default ma_list)
    python Analyze_madgraph_output.py 9

    # By mass token (0.01 GeV → reads merged_ma_0p0100GeV.csv)
    python Analyze_madgraph_output.py 0p01

    python Analyze_madgraph_output.py 9 --results-file my_scan

Output
------
Files are written under the results directory (see Usage):

  params.csv                   – cut parameters (created once if missing; unchanged on later runs)

  <stem>_total.csv             – default stem ``results_vbf``; all events passing pT cut with 2 converted photons
  <stem>_separated.csv         – additionally requiring track separation > sep_resolution
  <stem>_displaced.csv         – additionally requiring impact parameter > vertex_displacement
  <stem>_isolated.csv          – additionally requiring Delta-R < DeltaR_max

The four results CSVs have one header row of g_agg values, then one row per
ALP mass appended for each run (existing rows are not cleared).  ``params.csv``
is written only if it does not already exist so later mass points do not
clobber it.  Use the same cut parameters for every mass in a batch.  Use a
different ``--results-file`` stem to keep parallel scans separate.

Input data
----------
Events for each ALP mass are read from merged per-mass CSVs (same naming as
``merge_cmsrun3_csvs_by_mass.sh``)::

    <tracking-ALPs>/data/cmsrun3-csvs-merged/merged_ma_<mass>GeV.csv

where ``<mass>`` uses the MadGraph-style token (e.g. ``0p1000`` for 0.1 GeV).
Those files are built from per-run CSVs produced by ``lhe_to_csv.py``.

Directory layout expected
-------------------------
    tracking-ALPs/
    ├── Analyze_madgraph_output.py   (this script)
    ├── module_VBF.py
    ├── data/cmsrun3-csvs-merged/merged_ma_*GeV.csv
    └── results/                     # output CSVs (created automatically)

Monte Carlo (conversion sampling)
---------------------------------
The merged CSV kinematics are fixed, but ``module_VBF.raw_to_events`` **samples**
whether each photon converts in the tracker and where (probabilities depend on
``g_agg``, decay length, etc.).  Before sampling, this script calls
``numpy.random.seed`` — **default seed is 42** — so repeated runs with the same
inputs give the same counts (same NumPy version/build assumed).  Override with
``--seed N`` if you want independent draws (e.g. parallel batch tasks).
"""

################################################
## Load packages
################################################

import argparse
import os
import time

import numpy as np
import pandas as pd

from module_VBF import (
    read_data,
    raw_to_events,
    calculate_separations_2converted_displaced_isolated,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    'CMSRUN3_CSVS_MERGED_DIR',
    os.path.join(SCRIPT_DIR, 'data', 'cmsrun3-csvs-merged'),
)
_DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

################################################
## ALP mass grid
## ─────────────────────────────────────────────
## Add or remove masses here to match your Madgraph runs.
## ma_name must match the stem of merged_ma_*GeV.csv under DATA_DIR.
################################################

#ma_list = np.logspace(-2,1, num=32)
ma_list = [0.01, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.1,
        0.2, 0.3, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 1.0, 2.0,
        3.0, 4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0]


def ma_to_name(ma):
    """Map m_a (GeV) to merged CSV stem: merged_ma_0p7000GeV (MG5-style)."""
    token = f'{ma:.4f}'.replace('.', 'p')
    return f'merged_ma_{token}GeV'


def _mass_token_to_gev(spec):
    """
    Parse a MadGraph-style mass string to float GeV.

    Accepts e.g. ``0p01``, ``10p0000``, or ``merged_ma_0p0100GeV`` (prefix /
    suffix optional, case-insensitive).
    """
    t = spec.strip()
    if not t:
        raise ValueError('empty mass token')
    low = t.lower()
    if low.startswith('merged_ma_'):
        t = t[len('merged_ma_'):]
        low = t.lower()
    if low.endswith('gev'):
        t = t[:-3]
        low = t.lower()
    if 'p' not in low:
        raise ValueError(f'mass token must contain "p" as decimal point, got {spec!r}')
    i = low.index('p')
    numeric = t[:i] + '.' + t[i + 1 :]
    return float(numeric)


def _is_mass_token_argument(spec):
    """True → interpret as MG mass token / merged stem; False → ``ma_list`` index."""
    sl = spec.strip().lower()
    return 'p' in sl or sl.startswith('merged_ma_')


def _mass_index(s):
    """Accept integer or float-like strings (same as legacy ``round(float(...))``)."""
    return int(round(float(s)))


def _results_file_stem(s):
    """
    Basename only (no directories): becomes ``<stem>_total.csv``, etc.
    A trailing ``.csv`` is stripped if present.
    """
    stem = s.strip()
    if not stem:
        raise argparse.ArgumentTypeError('empty results file stem')
    if os.sep in stem or (os.altsep and os.altsep in stem):
        raise argparse.ArgumentTypeError(
            'results file must be a basename only (no path); outputs always go to the results directory'
        )
    if stem.endswith('.csv'):
        stem = stem[:-4]
    if not stem:
        raise argparse.ArgumentTypeError('empty results file stem')
    return stem


def _parse_args():
    p = argparse.ArgumentParser(
        description='Run VBF→ALP→γγ cut flow on merged MadGraph CSVs and append counts to results CSVs.',
    )
    p.add_argument(
        'mass',
        help=(
            f'Mass token with p as decimal (e.g. 0p01), optional merged_ma_…GeV stem, '
            f'or integer index into ma_list (0 … {len(ma_list) - 1})'
        ),
    )
    p.add_argument(
        '-r', '--results-file',
        metavar='STEM',
        type=_results_file_stem,
        default='results_vbf',
        help=(
            'Basename stem for the four results CSVs (default: results_vbf). '
            'Writes STEM_total.csv, STEM_separated.csv, … under the results directory.'
        ),
    )
    p.add_argument(
        '--seed',
        metavar='N',
        type=int,
        default=42,
        help=(
            'numpy.random.seed(N) before photon conversion sampling (default: 42). '
            'Use a different N per parallel task if you need independent MC draws.'
        ),
    )
    return p.parse_args()


args = _parse_args()
_mass_arg = args.mass.strip()
if _is_mass_token_argument(_mass_arg):
    try:
        ma_value = _mass_token_to_gev(_mass_arg)
    except ValueError as exc:
        raise SystemExit(f'Invalid mass token {_mass_arg!r}: {exc}') from exc
    ma_name = ma_to_name(ma_value)
else:
    idx = _mass_index(_mass_arg)
    if idx < 0 or idx >= len(ma_list):
        raise SystemExit(f'mass index must be in [0, {len(ma_list) - 1}], got {idx}')
    ma_value = ma_list[idx]
    ma_name = ma_to_name(ma_value)

print(f'ma_value: {ma_value} GeV')
print(f'ma_name (CSV stem): {ma_name}')

RESULTS_FILE_STEM = args.results_file
RESULTS_DIR = os.environ.get('TRACKING_ALP_RESULTS_DIR', _DEFAULT_RESULTS_DIR)
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

num_events = 100_000

################################################
## Cut parameters
## ─────────────────────────────────────────────
## Edit these to match your analysis scenario.
################################################

# Minimum ALP (= diphoton) transverse momentum (GeV).
# Because the two photons are very collimated, the pT cut is applied on the ALP.
pT_cut_value = 150    # GeV

# Tracker angular resolution for determining photon track directions (metres).
TRT_track_resolution =3.0e-5  # m = 10 micrometers

# Minimum displaced-vertex impact parameter required to tag a decay as displaced (metres).
vertex_displacement  = 1.0e-1  # m = 10 


# Minimum track separation required to resolve the two photon tracks (metres).
TRT_sep_resolution = 4.0e-4  # m = 0.4 mm

# Maximum Delta-R between the two photons for the pair to pass the ECAL isolation criterion.
# Uncomment the desired definition:
#DeltaR_max = np.sqrt(0.025**2 + 0.0245**2)   # ECAL cell size (~0.035)
# DeltaR_max = np.sqrt(0.075**2 + 0.123**2)  # ECAL L1 granularity (~0.14)
DeltaR_max = np.sqrt(0.0174**2 + 0.0174**2)  # ≈ 0.0246 CMS ECAL barrel cell size


################################################
## Make sure the output directory exists
################################################

try:
    os.makedirs(RESULTS_DIR, exist_ok=True)
except PermissionError as e:
    if RESULTS_DIR != _DEFAULT_RESULTS_DIR:
        print(
            f"WARNING: Cannot create RESULTS_DIR={RESULTS_DIR!r} "
            f"(from TRACKING_ALP_RESULTS_DIR). Falling back to default "
            f"{_DEFAULT_RESULTS_DIR!r}. Error was: {e}"
        )
        RESULTS_DIR = _DEFAULT_RESULTS_DIR
        os.makedirs(RESULTS_DIR, exist_ok=True)
    else:
        raise

################################################
## Write parameters to file
## ─────────────────────────────────────────────
## params.csv is created once if missing. Later mass-index runs do not
## overwrite it (so a batch can accumulate results under one parameter record).
## To change cuts and start fresh: remove params.csv and the results CSVs for
## your stem (default ``results_vbf_*.csv``), or use a new ``--results-file`` stem.
## All masses in a batch should use the same cut values as recorded there.
################################################

params = {
    'run_kind':            'analysis_run',
    'pT_cut':              pT_cut_value,
    'track_res':           TRT_track_resolution,
    'sep_res':             TRT_sep_resolution,
    'vertex_displacement': vertex_displacement,
    'Delta_R':             DeltaR_max,
}
params_path = os.path.join(RESULTS_DIR, 'params.csv')
if not os.path.isfile(params_path):
    pd.DataFrame.from_dict(params, orient='index').to_csv(params_path, header=False)
    print(f'Wrote {params_path}')
else:
    print(f'Leaving existing {params_path} unchanged.')

################################################
## Analysis pipeline
################################################

csv_path = os.path.join(DATA_DIR, ma_name + '.csv')
if not os.path.isfile(csv_path):
    raise SystemExit(f'Missing input CSV: {csv_path}')

_t_pipeline_start = time.perf_counter()
print(f'Reading data for m_a = {ma_value} GeV  ({ma_name}) ...')
raw_events = read_data(ma_name, num=num_events, data_dir=DATA_DIR)
n_loaded = len(raw_events)

print('Converting to event objects ...')
np.random.seed(args.seed)
events = raw_to_events(raw_events, gaggs=gagg_list, ma=ma_value)
del raw_events   # free memory

print('Calculating separations ...')
(separations,
 _average_etas,
 impact_parameters,
 delta_Rs,
 delta_etas) = calculate_separations_2converted_displaced_isolated(
    events,
    gaggs=gagg_list,
    pTcut=pT_cut_value,
    track_resolution=TRT_track_resolution,
    check=False,
)
del _average_etas, delta_etas, events

# ── Cut cascade: only event counts are needed for CSV output, so we mask
# in-place per g_agg without retaining intermediate kinematic arrays.
# ── Step 1: TRT acceptance (separation != -1)
# ── Step 2: track separation > TRT_sep_resolution
# ── Step 3: impact parameter > vertex_displacement
# ── Step 4: Delta-R < DeltaR_max (or swap for Delta-eta; see commented block)
n_g = len(separations)
counts_total = np.empty(n_g, dtype=np.int64)
counts_separated = np.empty(n_g, dtype=np.int64)
counts_displaced = np.empty(n_g, dtype=np.int64)
counts_isolated = np.empty(n_g, dtype=np.int64)
for i in range(n_g):
    s = separations[i]
    ip = impact_parameters[i]
    dr = delta_Rs[i]
    m = s != -1
    s, ip, dr = s[m], ip[m], dr[m]
    counts_total[i] = s.size
    m = s > TRT_sep_resolution
    s, ip, dr = s[m], ip[m], dr[m]
    counts_separated[i] = s.size
    m = ip > vertex_displacement
    s, ip, dr = s[m], ip[m], dr[m]
    counts_displaced[i] = s.size
    m = dr < DeltaR_max
    counts_isolated[i] = int(m.sum())

# Delta-eta isolation instead of Delta-R:
# for i in range(n_g):
#     ...
#     deta = delta_etas[i]
#     s, ip, deta = s[m], ip[m], deta[m]
#     ...
#     m = deta < DeltaR_max
#     counts_isolated[i] = int(m.sum())

################################################
## Write results to CSV
################################################

base = os.path.join(RESULTS_DIR, RESULTS_FILE_STEM)

# Write a header row with g_agg values only when creating a new results file.
# Subsequent runs append one row per mass (mode 'a'); they do not replace the file.
for suffix in ('_total', '_separated', '_displaced', '_isolated'):
    path = base + suffix + '.csv'
    if not os.path.exists(path):
        pd.DataFrame({'g_agg': gagg_list}).T.to_csv(path, mode='w', index=True, header=False)

def _write_row(path, label, counts):
    pd.DataFrame({label: counts}).T.to_csv(path, mode='a', index=True, header=False)

print('Writing results ...')
_write_row(base + '_total.csv',     ma_name, counts_total.astype(float))
_write_row(base + '_separated.csv', ma_name, counts_separated.astype(float))
_write_row(base + '_displaced.csv', ma_name, counts_displaced.astype(float))
_write_row(base + '_isolated.csv',  ma_name, counts_isolated.astype(float))

_elapsed = time.perf_counter() - _t_pipeline_start
if n_loaded > 0:
    _ms_ev = _elapsed / n_loaded * 1e3
    print(
        f'Timing: {_elapsed:.2f} s wall time for {n_loaded} events '
        f'({_ms_ev:.1f} ms/event; requested cap {num_events}).'
    )
else:
    print(
        f'Timing: {_elapsed:.2f} s wall time; 0 events loaded '
        f'(requested cap {num_events}).'
    )
print(f'Done. Results written to {RESULTS_FILE_STEM}_*.csv under {RESULTS_DIR}')

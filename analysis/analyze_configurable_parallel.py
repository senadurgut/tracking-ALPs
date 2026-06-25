"""
Per-mass / SLURM-array-friendly wrapper around analyze_configurable_v2.py.

The physics is NOT defined here.  Config resolution and the per-event cut chain
are imported from ``analyze_configurable_v2`` (``resolve_cuts`` + ``count_passing``),
so this script always stays in lock-step with v2 — update the cuts in v2 and the
per-mass workflow picks the change up automatically.

Modes
-----
* Without ``--mass-index``: drop-in equivalent of analyze_configurable_v2.py — it
  loops over every mass in ``ma_list`` and writes a single ``results_<stem>.csv``.
* With ``--mass-index N``: processes ONLY ``ma_list[N]`` and writes that mass's row
  to ``parts/row_ma<NNN>.csv`` (no header).  This is the form used by the SLURM
  array job.  Combine the parts into the standard single CSV with
  ``analysis/merge_parts.py``.

``np.random.seed(args.seed)`` is reset at the top of each mass iteration, so any
single mass produces the same counts whether it runs alone (array mode) or as part
of the full sweep (serial mode), for the same ``--seed``.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from module_VBF import read_data, raw_to_events
from scripts.analysis_helpers.file_helpers import ma_to_name
from analyze_configurable_v2 import (
    _load_config,
    _run_dir,
    resolve_cuts,
    resolved_summary,
    count_passing,
)


_DEFAULT_RESULTS_DIR = 'results'


def _parse_args():
    p = argparse.ArgumentParser(
        description='VBF→ALP→γγ cut-chain analysis (per-mass / SLURM-array wrapper around v2).',
    )
    p.add_argument('--config', required=True,
                   help='Config file (era, analysis_mode, masks, ma_list, gagg_list).')
    p.add_argument('--num-events', type=int, default=100000)
    p.add_argument('--data-dir', type=str, default='data/cmsrun3-csvs')
    p.add_argument('--results-dir', type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--mass-index', type=int, default=None,
                   help='If set, process only ma_list[mass_index] and write the row to '
                        'parts/row_ma<NNN>.csv (no header).  Intended for SLURM array jobs.  '
                        'Leave unset to run all masses sequentially as a single CSV.')
    p.add_argument('--run-dir', default=None,
                   help='Output dir; if unset, auto-resolved to run_<stem>[_NN].')

    return p.parse_args()


def main():
    args = _parse_args()
    config = _load_config(args.config)
    config_stem = os.path.splitext(os.path.basename(args.config))[0]

    ma_list   = config['ma_list']
    gagg_list = config['gagg_list']
    era       = config['era']

    cuts = resolve_cuts(config)

    run_dir = args.run_dir or _run_dir(args.results_dir, config_stem)
    os.makedirs(run_dir, exist_ok=True)
    stem = os.path.basename(run_dir).removeprefix('run_')

    if args.mass_index is not None:
        if not (0 <= args.mass_index < len(ma_list)):
            raise ValueError(f'--mass-index must be in [0, {len(ma_list) - 1}], got {args.mass_index}')
        masses_to_process = [(args.mass_index, ma_list[args.mass_index])]
        parts_dir = os.path.join(run_dir, 'parts')
        os.makedirs(parts_dir, exist_ok=True)
    else:
        masses_to_process = list(enumerate(ma_list))
        results_path = os.path.join(run_dir, f'results_{stem}.csv')
        header = ['g_agg'] + [str(x) for x in gagg_list]
        pd.DataFrame(columns=header).to_csv(results_path, index=False)

    if args.mass_index is None or args.mass_index == 0:
        params = {
            'cli': vars(args),
            'config': config,
            'resolved': resolved_summary(cuts),
        }
        params_path = os.path.join(run_dir, f'params_{stem}.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

    for i_mass, ma in masses_to_process:
        ma_name = ma_to_name(ma)
        raw_events = read_data(ma_name, num=args.num_events, data_dir=args.data_dir)
        n_loaded = len(raw_events)
        print(f'[ma_index={i_mass}] Loaded {n_loaded} events for m_a = {ma} GeV')
        np.random.seed(args.seed)
        events = raw_to_events(raw_events, gaggs=gagg_list, ma=ma, era=era)
        del raw_events

        counts = count_passing(events, gagg_list, cuts)

        row = [f'{ma}GeV'.replace('.', '_')] + counts
        if args.mass_index is None:
            pd.DataFrame([row], columns=header).to_csv(results_path, mode='a', index=False, header=False)
        else:
            part_path = os.path.join(parts_dir, f'row_ma{i_mass:03d}.csv')
            pd.DataFrame([row]).to_csv(part_path, index=False, header=False)
            print(f'[ma_index={i_mass}] Wrote {part_path}')


if __name__ == '__main__':
    main()

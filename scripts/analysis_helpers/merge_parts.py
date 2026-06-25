"""
Stitch per-mass part files written by ``analyze_configurable_parallel.py``
(in SLURM array mode) into the single ``results_<stem>.csv`` format that
``analyze_configurable.py`` produces.

Reads:
    <results-dir>/run_<stem>/parts/row_ma<NNN>.csv  (one row per file, no header)

Writes:
    <results-dir>/run_<stem>/results_<stem>.csv     (header + one row per mass)

Usage:
    python analysis/merge_parts.py --config configs/config1.json
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
import yaml


def _load_config(path):
    with open(path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        return json.load(f)


def _parse_args():
    p = argparse.ArgumentParser(
        description='Merge per-mass row files into the single results CSV.',
    )
    p.add_argument('--config', required=True,
                   help='Same config file passed to analyze_configurable_parallel.py.')
    p.add_argument('--results-dir', default='results',
                   help='Top-level results directory (default: results).')
    p.add_argument('--run-dir', default=None,
                   help='Run dir to merge; defaults to <results-dir>/run_<stem>.')
    p.add_argument('--strict', action='store_true',
                   help='Fail if any mass index in ma_list is missing a part file.')
    return p.parse_args()


def main():
    args = _parse_args()
    config = _load_config(args.config)
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    gagg_list = config['gagg_list']
    ma_list = config['ma_list']

    run_dir = args.run_dir or os.path.join(args.results_dir, f'run_{config_stem}')
    stem = os.path.basename(run_dir.rstrip('/')).removeprefix('run_')
    parts_dir = os.path.join(run_dir, 'parts')
    if not os.path.isdir(parts_dir):
        raise SystemExit(f'No parts directory found at {parts_dir}.  '
                         f'Did the array job run?')

    part_paths = sorted(glob.glob(os.path.join(parts_dir, 'row_ma*.csv')))
    if not part_paths:
        raise SystemExit(f'No part files matching row_ma*.csv in {parts_dir}.')

    found_indices = set()
    rows = []
    for path in part_paths:
        m = re.search(r'row_ma(\d+)\.csv$', os.path.basename(path))
        if not m:
            print(f'WARN: skipping unrecognised filename {path}')
            continue
        idx = int(m.group(1))
        found_indices.add(idx)
        df = pd.read_csv(path, header=None)
        if len(df) != 1:
            raise SystemExit(f'{path} should contain exactly 1 row, got {len(df)}.')
        rows.append((idx, df.iloc[0].tolist()))

    expected = set(range(len(ma_list)))
    missing = sorted(expected - found_indices)
    extra = sorted(found_indices - expected)
    if missing:
        msg = f'Missing part files for mass indices: {missing}'
        if args.strict:
            raise SystemExit(msg)
        print(f'WARN: {msg}  (re-submit those array indices, or rerun with --strict to fail).')
    if extra:
        print(f'WARN: part files for unknown mass indices {extra} are present and will be included.')

    rows.sort(key=lambda x: x[0])
    out_rows = [r for _, r in rows]

    header = ['g_agg'] + [str(x) for x in gagg_list]
    out_path = os.path.join(run_dir, f'results_{stem}.csv')
    pd.DataFrame(columns=header).to_csv(out_path, index=False)
    pd.DataFrame(out_rows).to_csv(out_path, mode='a', index=False, header=False)
    print(f'Wrote {out_path}  ({len(out_rows)} rows)')


if __name__ == '__main__':
    main()

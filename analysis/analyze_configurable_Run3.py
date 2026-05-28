
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from module_VBF import (
    read_data,
    raw_to_events,
    separation_TRT,
    displaced_vertex_TRT,
    Delta_R,
    TRT_length,
)
from scripts.analysis.file_helpers import ma_to_name
from scripts.analysis.kinematics_helpers import compute_mjj


_DEFAULT_RESULTS_DIR = 'results'


def _load_config(path):
    with open(path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        return json.load(f)


def _parse_args():
    p = argparse.ArgumentParser(
        description='VBF→ALP→γγ cut-chain analysis with configurable parameters.',
    )
    p.add_argument('--mass-config', default='configs/mass_configs/all.json')
    p.add_argument('--cut-config', default='configs/cut_configs/masks.json')
    p.add_argument('--other-params', default='configs/other_params/other_params.json')
    p.add_argument('--num-events', type=int, default=100000)
    p.add_argument('--data-dir', type=str, default='data/cmsrun3-csvs')
    p.add_argument('--results-dir', type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument('--seed', type=int, default=1234)
    return p.parse_args()


def main():
    args = _parse_args()
    mass_config = _load_config(args.mass_config)
    ma_list = mass_config['ma_list']
    cut_config = _load_config(args.cut_config)
    eta_mask_config = cut_config.get('mask_eta', {})
    vbf_mask_config = cut_config.get('mask_vbf', {})
    merge_mask_config = cut_config.get('mask_merge', {})
    alp_pT_cut_config = cut_config.get('alp_pT_cut', {})
    mask_eta = eta_mask_config.get('value', False)
    mask_vbf = vbf_mask_config.get('value', False)
    mask_merge = merge_mask_config.get('value', True)
    mask_pT = alp_pT_cut_config.get('value', True)
    eta_range = eta_mask_config.get('eta_range', [0.0, 0.0])
    leading_jet_pt_cut = vbf_mask_config.get('lead_pt', 0.0)
    sub_jet_pt_cut = vbf_mask_config.get('sub_pt', 0.0)
    mjj_cut = vbf_mask_config.get('mjj', 0.0)
    deta_cut = vbf_mask_config.get('deta', 0.0)

    if mask_merge:
        delta_r_max = merge_mask_config.get('delta_r_max', 0.3)
        delta_r_min = merge_mask_config.get('delta_r_min', 0.025)
    else:
        delta_r_min = 0
        delta_r_max = 0.3
    alp_pT_cut_value = alp_pT_cut_config.get('cut_value', 0.0)
    pT_cut = alp_pT_cut_value if mask_pT else 0.0
    other_params = _load_config(args.other_params)
    gagg_list = other_params['gagg_list']

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.results_dir, f'run_{run_id}')
    os.makedirs(run_dir, exist_ok=True)
    params = {
        'run_id': run_id,
        'cli': vars(args),
        'configs': {
            'mass': mass_config,
            'cut': cut_config,
            'other': other_params,
        },
        'resolved': {
            'mask_eta': mask_eta,
            'mask_vbf': mask_vbf,
            'mask_merge': mask_merge,
            'mask_pT': mask_pT,
            'eta_range': eta_range,
            'leading_jet_pt_cut': leading_jet_pt_cut,
            'sub_jet_pt_cut': sub_jet_pt_cut,
            'mjj_cut': mjj_cut,
            'deta_cut': deta_cut,
            'pT_cut': pT_cut,
            'delta_r_max': delta_r_max,
            'delta_r_min': delta_r_min,
        },
    }
    params_path = os.path.join(run_dir, f'params_{run_id}.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    results_path = os.path.join(run_dir, f'results_{run_id}.csv')
    header = ['g_agg'] + [str(x) for x in gagg_list]
    pd.DataFrame(columns=header).to_csv(results_path, index=False)

    for ma in ma_list:
        ma_name = ma_to_name(ma)
        raw_events = read_data(ma_name, num=args.num_events, data_dir=args.data_dir)
        n_loaded = len(raw_events)
        print(f'Loaded {n_loaded} events for m_a = {ma} GeV')
        np.random.seed(args.seed)
        events = raw_to_events(raw_events, gaggs=gagg_list, ma=ma)
        del raw_events

        counts = [0] * len(gagg_list)

        for i_g, g in enumerate(gagg_list):

            for i, ev in enumerate(events):
                passes_pT = (ev['a']['pt'] > pT_cut) if mask_pT else True # currently set to zero, can be changed later if needed 
                alp_inside_tracker = -ev['a']['l'][i_g] * np.log(np.random.uniform()) < TRT_length(ev['a']['eta'])

                if not (passes_pT and alp_inside_tracker):
                    continue

                eta1 = ev['g1']['eta']
                eta2 = ev['g2']['eta']
                eta_a = ev['a']['eta']
                phi1 = ev['g1']['phi']
                phi2 = ev['g2']['phi']
                phi_a = ev['a']['phi']
                l1 = ev['g1']['l_track'][i_g]
                l2 = ev['g2']['l_track'][i_g]
                l_a = ev['a']['l'][i_g]
                if mask_eta: # no need for modification
                    passes_eta_1 = (abs(eta1) >= eta_range[0]) & (abs(eta1) <= eta_range[1])
                    passes_eta_2 = (abs(eta2) >= eta_range[0]) & (abs(eta2) <= eta_range[1])
                    if not (passes_eta_1 & passes_eta_2):
                        continue
                if mask_vbf: # no need for modification
                    lead_jet_pt = max(ev['j1']['pt'], ev['j2']['pt'])
                    sub_jet_pt = min(ev['j1']['pt'], ev['j2']['pt'])
                    mjj = compute_mjj(ev)
                    passes_lead_jet_pt = lead_jet_pt > leading_jet_pt_cut
                    passes_sub_jet_pt = sub_jet_pt > sub_jet_pt_cut
                    passes_mjj = mjj > mjj_cut
                    passes_deta = abs(ev['j1']['eta'] - ev['j2']['eta']) > deta_cut
                    if not (passes_lead_jet_pt & passes_sub_jet_pt & passes_mjj & passes_deta):
                        continue
                if mask_merge:
                    delta_r=Delta_R(eta1, eta2, phi1, phi2, l_a)
                    passes_merge = delta_r > delta_r_min and delta_r < delta_r_max
                    if not passes_merge:
                        continue
                counts[i_g] += 1

        row = [f'{ma}GeV'.replace('.', '_')] + counts
        pd.DataFrame([row], columns=header).to_csv(results_path, mode='a', index=False, header=False)


if __name__ == '__main__':
    main()

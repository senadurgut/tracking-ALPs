
import argparse
import json
import os
import sys
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
    eta_region,
    hgcal_cell_size
)
from scripts.analysis_helpers.file_helpers import ma_to_name
from scripts.analysis_helpers.kinematics_helpers import compute_mjj


_DEFAULT_RESULTS_DIR = 'results'


def _load_config(path):
    with open(path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        return json.load(f)


def _run_dir(results_dir, stem):
    d = os.path.join(results_dir, f'run_{stem}')
    i = 0
    while os.path.exists(d):
        i += 1
        d = os.path.join(results_dir, f'run_{stem}_{i:02d}')
    return d


def resolve_cuts(config):
    """Flatten a merged config dict into the resolved cut parameters used by the analysis.

    This (together with ``count_passing``) is the single source of truth for the
    physics.  ``analyze_configurable_parallel.py`` imports both, so any change here
    is automatically used by the per-mass / SLURM-array workflow too.
    """
    eta_mask_config   = config.get('mask_eta', {})
    vbf_mask_config   = config.get('mask_vbf', {})
    merge_mask_config = config.get('mask_merge', {})
    alp_pT_cut_config = config.get('alp_pT_cut', {})
    phase2_cfg        = config.get('phase2_cuts', {})

    analysis_mode = config.get('analysis_mode', 'parking')
    if analysis_mode not in ('parking', 'scouting'):
        raise ValueError(f"analysis_mode must be 'parking' or 'scouting', got {analysis_mode!r}")

    mask_alp_pT = alp_pT_cut_config.get('value', True)
    mask_merge  = merge_mask_config.get('value', True)

    return {
        'era':           config['era'],
        'analysis_mode': analysis_mode,
        'use_tracks':    config.get('use_tracks', False),
        'mask_eta':      eta_mask_config.get('value', False),
        'mask_vbf':      vbf_mask_config.get('value', False),
        'mask_merge':    mask_merge,
        'mask_alp_pT':   mask_alp_pT,
        'eta_range':          eta_mask_config.get('eta_range', [0.0, 0.0]),
        'leading_jet_pt_cut': vbf_mask_config.get('lead_pt', 0.0),
        'sub_jet_pt_cut':     vbf_mask_config.get('sub_pt', 0.0),
        'mjj_cut':            vbf_mask_config.get('mjj', 0.0),
        'deta_cut':           vbf_mask_config.get('deta', 0.0),
        'sep_cut':          phase2_cfg.get('sep_cut', 5.0e-4),
        'disp_cut':         phase2_cfg.get('disp_cut', 1.0e-1),
        'track_resolution': phase2_cfg.get('track_resolution', 2.0e-4),
        'delta_r_max':   merge_mask_config.get('delta_r_max', 0.3) if mask_merge else 0.3,
        'ecal_cell_size': 0.025,
        'pT_cut':        alp_pT_cut_config.get('cut_value', 0.0) if mask_alp_pT else 0.0,
    }


def resolved_summary(cuts):
    """The 'resolved' block written into params_<stem>.json (scouting-only cuts nulled otherwise)."""
    scouting = cuts['analysis_mode'] == 'scouting'
    return {
        'era': cuts['era'],
        'analysis_mode': cuts['analysis_mode'],
        'use_tracks': cuts['use_tracks'],
        'mask_eta': cuts['mask_eta'],
        'mask_vbf': cuts['mask_vbf'],
        'mask_merge': cuts['mask_merge'],
        'mask_alp_pT': cuts['mask_alp_pT'],
        'eta_range': cuts['eta_range'],
        'leading_jet_pt_cut': cuts['leading_jet_pt_cut'],
        'sub_jet_pt_cut': cuts['sub_jet_pt_cut'],
        'mjj_cut': cuts['mjj_cut'],
        'deta_cut': cuts['deta_cut'],
        'pT_cut': cuts['pT_cut'],
        'delta_r_max': cuts['delta_r_max'],
        'sep_cut':          cuts['sep_cut']          if scouting else None,
        'disp_cut':         cuts['disp_cut']         if scouting else None,
        'track_resolution': cuts['track_resolution'] if scouting else None,
        'ecal_cell_size': cuts['ecal_cell_size'],
    }


def count_passing(events, gagg_list, cuts):
    """Run the per-event cut chain for one mass and return the per-g_agg pass counts.

    The single source of truth for the physics selection (run3/phase2, parking/
    scouting).  Assumes ``np.random.seed(...)`` has already been set by the caller
    (immediately before ``raw_to_events``) so the RNG stream is reproducible.
    """
    era             = cuts['era']
    analysis_mode   = cuts['analysis_mode']
    use_tracks      = cuts['use_tracks']
    mask_eta        = cuts['mask_eta']
    mask_vbf        = cuts['mask_vbf']
    mask_merge      = cuts['mask_merge']
    mask_alp_pT     = cuts['mask_alp_pT']
    eta_range       = cuts['eta_range']
    leading_jet_pt_cut = cuts['leading_jet_pt_cut']
    sub_jet_pt_cut  = cuts['sub_jet_pt_cut']
    mjj_cut         = cuts['mjj_cut']
    deta_cut        = cuts['deta_cut']
    sep_cut         = cuts['sep_cut']
    disp_cut        = cuts['disp_cut']
    track_resolution = cuts['track_resolution']
    delta_r_max     = cuts['delta_r_max']
    ecal_cell_size  = cuts['ecal_cell_size']
    pT_cut          = cuts['pT_cut']

    counts = [0] * len(gagg_list)

    for i_g, g in enumerate(gagg_list):
        for i, ev in enumerate(events):
            passes_pT = (ev['a']['pt'] > pT_cut) if mask_alp_pT else True # currently set to zero, can be changed later if needed 
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
            if mask_eta: # keep this for now
                passes_eta_1 = (abs(eta1) >= eta_range[0]) & (abs(eta1) <= eta_range[1]) # keep this for now
                passes_eta_2 = (abs(eta2) >= eta_range[0]) & (abs(eta2) <= eta_range[1]) # keep this for now
                if not (passes_eta_1 & passes_eta_2): # keep this for now
                    continue # keep this for now
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

                if era =='run3' and analysis_mode == 'parking':
                    cell_size = ecal_cell_size
                    if delta_r < cell_size:
                        if use_tracks:
                            both_conv   = ev['g1']['conv'][i_g] and ev['g2']['conv'][i_g]
                            passes_merge = both_conv 
                        else:
                            passes_merge = False
                    elif delta_r < delta_r_max:
                        passes_merge = True
                    else:
                        passes_merge = False 

                elif era =='phase2' and analysis_mode == 'parking':
                    if  eta_region(eta_a) == 'barrel':
                        cell_size = ecal_cell_size
                    elif eta_region(eta_a) == 'endcap':
                        cell_size = hgcal_cell_size(eta_a)
                    
                    if delta_r<=cell_size:
                        both_conv = ev['g1']['conv'][i_g] and ev['g2']['conv'][i_g]
                        passes_merge = both_conv 
                    elif delta_r <= delta_r_max:
                        passes_merge = True
                    else: 
                        passes_merge=False

                elif era == 'phase2' and analysis_mode == 'scouting':
                    if delta_r <= delta_r_max: 
                        passes_merge=True
                    else: 
                        passes_merge=False
                else:
                    raise ValueError(f"unsupported era/analysis_mode pair, check your config")
                    
                if not passes_merge: 
                    continue

            if analysis_mode=='scouting':
                both_conv   = ev['g1']['conv'][i_g] and ev['g2']['conv'][i_g]
                if not both_conv: 
                    continue 
                if separation_TRT(eta1, eta2, eta_a, phi1, phi2, phi_a, l_a) < sep_cut:
                    continue
                if displaced_vertex_TRT(eta_a, phi1, phi2, phi_a, l1, l2, l_a, track_resolution) < disp_cut:
                    continue

            counts[i_g] += 1

    return counts


def _parse_args():
    p = argparse.ArgumentParser(
        description='VBF→ALP→γγ cut-chain analysis with configurable parameters.',
    )
    p.add_argument('--config', required=True,
                   help='Config file (era, analysis_mode, masks, ma_list, gagg_list).')
    p.add_argument('--num-events', type=int, default=100000)
    p.add_argument('--data-dir', type=str, default='data/cmsrun3-csvs')
    p.add_argument('--results-dir', type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument('--seed', type=int, default=1234)

    return p.parse_args()


def main():
    args = _parse_args()
    config = _load_config(args.config)
    config_stem = os.path.splitext(os.path.basename(args.config))[0]

    ma_list   = config['ma_list']
    gagg_list = config['gagg_list']
    era       = config['era']

    cuts = resolve_cuts(config)

    run_dir = _run_dir(args.results_dir, config_stem)
    os.makedirs(run_dir, exist_ok=True)
    stem = os.path.basename(run_dir).removeprefix('run_')
    params = {
        'cli': vars(args),
        'config': config,
        'resolved': resolved_summary(cuts),
    }
    params_path = os.path.join(run_dir, f'params_{stem}.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    results_path = os.path.join(run_dir, f'results_{stem}.csv')
    header = ['g_agg'] + [str(x) for x in gagg_list]
    pd.DataFrame(columns=header).to_csv(results_path, index=False)

    for ma in ma_list:
        ma_name = ma_to_name(ma)
        raw_events = read_data(ma_name, num=args.num_events, data_dir=args.data_dir)
        n_loaded = len(raw_events)
        print(f'Loaded {n_loaded} events for m_a = {ma} GeV')
        np.random.seed(args.seed)
        events = raw_to_events(raw_events, gaggs=gagg_list, ma=ma, era=era)
        del raw_events

        counts = count_passing(events, gagg_list, cuts)

        row = [f'{ma}GeV'.replace('.', '_')] + counts
        pd.DataFrame([row], columns=header).to_csv(results_path, mode='a', index=False, header=False)


if __name__ == '__main__':
    main()

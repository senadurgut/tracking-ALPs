"""
module_VBF.py
=============
Physics utility module for analysing ALP (axion-like particle) production via
vector-boson fusion (VBF) at the LHC, with the ALP subsequently decaying to two
photons inside the tracking detector.

The module implements:
  - Data I/O: reading pre-processed Madgraph event CSVs.
  - Kinematics: 4-momentum parsing, decay-length calculation.
  - Detector simulation (ATLAS geometry by default):
      * Photon conversion probability as a function of eta, including finite
        ALP decay-length corrections (from ATLAS paper arXiv:1810.05087).
      * TRT tracker length as a function of eta.
      * Transverse separation of the two conversion tracks at the outer TRT wall.
      * Impact-parameter (displaced-vertex) significance.
      * Delta-R / Delta-eta between the two photons at the ECAL.
  - Batch processing: ``calculate_separations_2converted_displaced_isolated``
    sweeps over a grid of ALP–photon couplings g_agg.

Detector geometry
-----------------
All geometry constants at the top of this file correspond to the **ATLAS**
inner detector (TRT barrel + endcap) and ECAL.  If you are analysing a
different experiment, replace the values in the "Detector geometry" section
with your detector's parameters and supply your own photon-conversion
probability table (``conv_fr`` / ``std_conv_fr``).

Input data format
-----------------
Each event is stored as 7 consecutive rows in a semicolon-delimited CSV file.
Each row contains the four-momentum [E, px, py, pz] of one particle,
comma-separated:

  Row 0: incoming quark 1
  Row 1: incoming quark 2
  Row 2: VBF jet 1
  Row 3: VBF jet 2
  Row 4: ALP  (a)
  Row 5: photon 1 (g1) from a -> gg
  Row 6: photon 2 (g2) from a -> gg

Only rows 4–6 are read by ``read_data``; rows 0–3 are skipped.

CSV files must be placed in ``data/<run_name>.csv`` relative to the working
directory (or pass a full path to ``read_data``).  See ``lhe_to_csv.py`` for
the script that converts Madgraph LHE output to this format.
"""

################################################
## Load packages
################################################

import numpy as np
from numpy.random import uniform
from scipy.optimize import root_scalar
import scipy as scp
from sympy import *
import pandas as pd
import os
import copy
import sys

################################################
## Detector geometry  (Updated for CMS Run 3, April 9, 2026, based on https://cds.cern.ch/record/1129810/files/jinst8_08_s08004.pdf )
## ─────────────────────────────────────────────
## Replace these values if you use a different detector.
################################################

# TRT barrel
R_min_TRT        = 0.55   # m  – inner radius of barrel, needs to be decided if it's 0.2 or 0.55
R_max_TRT        = 1.10   # m  – outer radius of barrel
z_max_TRT        = 1.18   # m  – half-length of barrel active volume



# TRT endcap
z_min_TRT_endcap = 1.24  # m  – inner z-edge of endcap active volume
z_max_TRT_endcap = 2.82  # m  – outer z-edge of endcap active volume
R_min_TRT_endcap = 0.225  # m  – inner radius of endcap
R_max_TRT_endcap = 1.135  # m  – outer radius of endcap

# ECAL
R_ECAL = 1.29  # m  – CMS ECAL barrel inner face

# Derived pseudorapidity boundaries (do not edit)
eta_min           = 0.
eta_max           = -np.log(np.tan(0.5 * np.arctan(R_max_TRT / z_max_TRT)))
eta_min_endcap    = -np.log(np.tan(0.5 * np.arctan(R_max_TRT_endcap / z_min_TRT_endcap)))
eta_max_endcap    = -np.log(np.tan(0.5 * np.arctan(R_min_TRT_endcap / z_max_TRT_endcap)))
eta_corner_endcap = -np.log(np.tan(0.5 * np.arctan(R_max_TRT_endcap / z_max_TRT_endcap)))

################################################
## Photon conversion fractions  (ATLAS, arXiv:1810.05087) !STILL NEEDS TO BE UPDATED FOR CMS RUN 3!
## ─────────────────────────────────────────────
## conv_fr[0]  : upper edges of |eta| bins
## conv_fr[1]  : fraction of all photons reconstructed as converted
## conv_fr[2]  : fraction of true *unconverted* photons reco'd as converted (fake rate)
## conv_fr[3]  : fraction of true *converted*  photons reco'd as converted  (efficiency)
##
## If you use a different detector, replace these arrays with your own
## measurements.  The last column (eta -> inf) is a dummy "no-acceptance" bin.
################################################

conv_fr = np.array([
    [0.6,  1.37, 1.52, 1.81, 2.37, np.inf],   # |eta| upper bin edges
    [0.215, 0.309, 0.,  0.438, 0.536, 0.],     # total converted fraction
    [0.053, 0.036, 0.,  0.001, 0.003, 0.],     # fake-conversion rate
    [0.731, 0.708, 1.,  0.812, 0.544, 1.],     # true-conversion efficiency
])

std_conv_fr = np.array([
    [0.6,   1.37,  1.52, 1.81,  2.37,  np.inf],
    [0.014, 0.021, 0.,   0.031, 0.014, 0.],
    [0.007, 0.007, 0.,   0.009, 0.006, 0.],
    [0.040, 0.043, 0.,   0.052, 0.014, 0.],
])

################################################
## Helper: file existence check
################################################

def is_non_zero_file(fpath):
    """Return True if *fpath* exists and is non-empty."""
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

################################################
## Physics: decay length
################################################

def decay_length(ma, momentum, gagg):
    """
    Compute the lab-frame decay length of an ALP decaying to two photons.

    Parameters
    ----------
    ma : float
        ALP mass in GeV.
    momentum : float
        ALP 3-momentum magnitude in GeV.
    gagg : float
        ALP–photon coupling in GeV^{-1}.

    Returns
    -------
    float
        Decay length in metres.

    Notes
    -----
    The proper decay length is
        c*tau = 64*pi / (g_{agg}^2 * m_a^3)
    in natural units (GeV^{-1}).  The conversion factor
    1 GeV^{-1} = 1.973e-16 m is used.
    """
    gamma = momentum / ma
    ctau_iGeV = 64.0 * np.pi * gamma / (gagg**2 * ma**3)
    return ctau_iGeV * 1.0e-9 * 1.973e-7  # GeV^{-1} -> m

################################################
## Photon conversion probability
################################################

def conv_prob_novec(eta, kind='true'):
    """
    Return the photon conversion probability for a single pseudorapidity value.

    Parameters
    ----------
    eta : float
        Photon pseudorapidity.
    kind : {'true', 'total', 'fake', 'reco'}
        Which fraction to return:
        - 'true'  : probability that a photon truly converts in the tracker.
        - 'total' : fraction of all photons reconstructed as converted.
        - 'fake'  : fraction of unconverted photons misidentified as converted.
        - 'reco'  : reconstruction efficiency for genuinely converted photons.

    Returns
    -------
    float
    """
    i = np.searchsorted(conv_fr[0], abs(eta))
    if kind == 'true':
        return (conv_fr[1, i] - conv_fr[2, i]) / (conv_fr[3, i] - conv_fr[2, i])
    elif kind == 'total':
        return conv_fr[1, i]
    elif kind == 'fake':
        return conv_fr[2, i]
    elif kind == 'reco':
        return conv_fr[3, i]
    else:
        print('conv_prob_novec: unknown kind "{}"'.format(kind))
        return 0.

conv_prob = np.vectorize(conv_prob_novec)


def std_conv_prob_novec(eta, kind='true'):
    """
    Return the uncertainty on the photon conversion probability.

    Parameters mirror those of ``conv_prob_novec``.
    """
    i = np.searchsorted(conv_fr[0], abs(eta))
    if kind == 'true':
        denom = conv_fr[3, i] - conv_fr[2, i]
        return np.sqrt(
            (std_conv_fr[1, i] / denom)**2
            + (std_conv_fr[2, i] * (conv_fr[1, i] - conv_fr[3, i]) / denom**2)**2
            + (std_conv_fr[3, i] * (conv_fr[1, i] - conv_fr[2, i]) / denom**2)**2
        )
    elif kind == 'total':
        return std_conv_fr[1, i]
    elif kind == 'fake':
        return std_conv_fr[2, i]
    elif kind == 'reco':
        return std_conv_fr[3, i]
    else:
        print('std_conv_prob_novec: unknown kind "{}"'.format(kind))
        return 0.

std_conv_prob = np.vectorize(std_conv_prob_novec)

################################################
## Tracker geometry
################################################

def TRT_length_novec(eta):
    """
    Return the radial (barrel) or longitudinal (endcap) path length through
    the TRT active volume for a straight track at pseudorapidity *eta*.

    Returns 0 if *eta* is outside the TRT acceptance.

    Parameters
    ----------
    eta : float
        Track pseudorapidity (sign is taken into account via abs).

    Returns
    -------
    float
        Path length in metres.
    """
    theta = 2.0 * np.arctan(np.exp(-eta))
    aeta  = abs(eta)
    if aeta < eta_max:
        return R_max_TRT / np.sin(theta)
    elif eta_min_endcap < aeta < eta_corner_endcap:
        return R_max_TRT_endcap / np.sin(theta)
    elif eta_corner_endcap <= aeta < eta_max_endcap:
        return z_max_TRT_endcap / abs(np.cos(theta))
    else:
        return 0.

TRT_length = np.vectorize(TRT_length_novec)

################################################
## Conversion probability corrected for finite ALP lifetime
################################################

def conv_prob_finite_lifetime_novec(eta, ma, pa, gagg, kind='true'):
    """
    Compute the photon conversion probability for a photon coming from an ALP
    that decays at a displaced vertex, and sample the conversion point.

    Because the ALP decays at a distance *l* from the IP, the photon starts
    its trajectory from an off-origin point, reducing the effective tracker
    length available for conversion.

    Parameters
    ----------
    eta : float
        Photon pseudorapidity.
    ma : float
        ALP mass in GeV.
    pa : float
        ALP 3-momentum in GeV.
    gagg : float
        ALP–photon coupling in GeV^{-1}.
    kind : str
        Passed to ``conv_prob``; usually 'true'.

    Returns
    -------
    c_prob : float
        Effective conversion probability (scalar).
    conv : bool
        Whether the photon converts (Monte Carlo decision).
    track_length : float
        If converted, the track length inside the TRT in metres; else 0.
    """
    d_length   = decay_length(ma, pa, gagg)
    f_conv_true = conv_prob(eta, kind=kind)
    # Also account for reconstruction efficiency
    f_conv      = f_conv_true * np.sqrt(conv_prob(eta, kind='reco'))
    L_tracker   = TRT_length(eta)

    if L_tracker == 0.:
        c_prob = 0.
    else:
        c_prob = f_conv * (1.0 - (1.0 - np.exp(-L_tracker / d_length)) * d_length / L_tracker)

    random = uniform()
    conv   = random < c_prob

    if conv:
        random2 = uniform()
        cumulative = lambda x: (
            f_conv / c_prob
            * (1.0 - d_length / L_tracker * (
                1.0 - (1.0 - (L_tracker - x) / d_length) * np.exp(-x / d_length)
            ))
        )
        bracket_ok = (cumulative(0) - random2) * (cumulative(L_tracker) - random2) < 0
        if bracket_ok:
            result = root_scalar(lambda x: cumulative(x) - random2,
                                 bracket=[0, L_tracker], method='brentq')
            track_length = L_tracker - result.root
        else:
            track_length = 0.
    else:
        track_length = 0.

    return c_prob, conv, track_length

conv_prob_finite_lifetime = np.vectorize(conv_prob_finite_lifetime_novec)

################################################
## Data structures
################################################

# Template dictionaries used to build event arrays.
# These define the keys that each particle/event dict will have.

_particle_template = {
    "eta"     : np.array([]),   # pseudorapidity
    "phi"     : np.array([]),   # azimuthal angle (rad)
    "pt"      : np.array([]),   # transverse momentum (GeV)
    "p"       : np.array([]),   # 3-momentum magnitude (GeV)
    "E"       : np.array([]),   # energy (GeV)
    "l"       : np.array([]),   # ALP decay length per g_agg value (m) [ALP only]
    "p_conv"  : np.array([]),   # conversion probability per g_agg [photons only]
    "conv"    : np.array([]),   # bool: did photon convert? per g_agg [photons only]
    "l_track" : np.array([]),   # available track length per g_agg (m) [photons only]
}

_event_template = {
    "a"  : dict(_particle_template),   # ALP
    "g1" : dict(_particle_template),   # photon 1
    "g2" : dict(_particle_template),   # photon 2
}

_raw_event_template = {
    "a"  : np.array([]),   # 4-momentum [E, px, py, pz]
    "g1" : np.array([]),
    "g2" : np.array([]),
}

################################################
## I/O: reading Madgraph CSV files
################################################

# ── Row-index map for the 7-row-per-event CSV format ─────────────────────────
# Each event occupies exactly ROWS_PER_EVENT consecutive rows in the file.
# Only the three rows below are actually used; the others are skipped.
#
# *** If your CSV was produced by a different script and has a different
#     particle ordering, change ROW_ALP, ROW_G1, ROW_G2 to match. ***
#
# To verify the ordering: inspect the first event in your CSV manually and
# check that row ROW_ALP has the ALP energy (usually the largest of the three)
# and that rows ROW_G1/ROW_G2 have the two photon four-momenta.
# The helper function ``print_first_event`` below can assist with this.

ROWS_PER_EVENT = 7     # total rows written per event (including skipped particles)
ROW_ALP        = 4     # row index of the ALP  four-momentum [E, px, py, pz]
ROW_G1         = 5     # row index of photon 1 four-momentum
ROW_G2         = 6     # row index of photon 2 four-momentum

# Expected content of skipped rows (for documentation only — not enforced):
#   Row 0: incoming quark 1
#   Row 1: incoming quark 2
#   Row 2: VBF jet 1
#   Row 3: VBF jet 2


def print_first_event(run_name, data_dir='data'):
    """
    Print the raw four-momenta of the first event in a CSV file.

    Use this to verify that ROW_ALP / ROW_G1 / ROW_G2 are set correctly
    before running the full analysis.  The ALP row should have the highest
    energy among rows 4-6, and energy should equal sqrt(|p|^2 + m_a^2).

    Parameters
    ----------
    run_name : str
        Run name (filename stem), as passed to ``read_data``.
    data_dir : str, optional
        Directory containing the CSV files.  Default ``'data'``.
    """
    filepath = os.path.join(data_dir, run_name + '.csv')
    temp     = pd.read_csv(filepath, sep=';', header=None, nrows=ROWS_PER_EVENT)
    labels   = [
        'row 0 (expected: incoming q1)',
        'row 1 (expected: incoming q2)',
        'row 2 (expected: VBF jet 1) ',
        'row 3 (expected: VBF jet 2) ',
        'row 4 (expected: ALP)       ',
        'row 5 (expected: photon 1)  ',
        'row 6 (expected: photon 2)  ',
    ]
    print(f'First event in {filepath}:')
    print(f'  {"":35s}  {"E":>12s}  {"px":>12s}  {"py":>12s}  {"pz":>12s}')
    for j in range(ROWS_PER_EVENT):
        vals = np.fromstring(temp.values[j, 0], sep=',')
        tag  = ' ← ALP' if j == ROW_ALP else (' ← γ1' if j == ROW_G1 else (' ← γ2' if j == ROW_G2 else ''))
        print(f'  {labels[j]}  {vals[0]:12.4f}  {vals[1]:12.4f}  {vals[2]:12.4f}  {vals[3]:12.4f}{tag}')


def read_data(run_name, num=10000, data_dir='data'):
    """
    Read pre-processed Madgraph events from a CSV file.

    The CSV file must contain exactly ``ROWS_PER_EVENT`` (default 7) rows per
    event with no header.  Each row holds the four-momentum
    ``[E, px, py, pz]`` of one particle, comma-separated, and rows are
    separated by semicolons.

    The rows that are actually read are controlled by the module-level
    constants ``ROW_ALP``, ``ROW_G1``, ``ROW_G2`` (defaults: 4, 5, 6).
    If your CSV has a different ordering, change those constants.
    Call ``print_first_event`` to verify the ordering before running a
    full analysis.

    Default row layout (produced by ``lhe_to_csv.py``):

        Row 0: incoming quark 1   (skipped)
        Row 1: incoming quark 2   (skipped)
        Row 2: VBF jet 1          (skipped)
        Row 3: VBF jet 2          (skipped)
        Row 4: ALP                → stored as 'a'
        Row 5: photon 1           → stored as 'g1'
        Row 6: photon 2           → stored as 'g2'

    Parameters
    ----------
    run_name : str
        Name of the run, used to locate ``<data_dir>/<run_name>.csv``.
        Example: ``'01GeV'`` for m_a = 0.1 GeV.
    num : int, optional
        Maximum number of events to load.  Default 10 000.
    data_dir : str, optional
        Directory containing the CSV files.  Default ``'data'``.

    Returns
    -------
    list of dict
        Each element corresponds to one event and has keys ``'a'``, ``'g1'``,
        ``'g2'``, each holding a NumPy array ``[E, px, py, pz]``.
    """
    filepath = os.path.join(data_dir, run_name + '.csv')
    temp     = pd.read_csv(filepath, sep=';', header=None, nrows=ROWS_PER_EVENT * num)
    length   = min(len(temp) // ROWS_PER_EVENT, num)

    raw_events = [dict(_raw_event_template) for _ in range(length)]
    for i in range(length):
        rows = [np.fromstring(temp.values[ROWS_PER_EVENT * i + j, 0], sep=',')
                for j in range(ROWS_PER_EVENT)]
        raw_events[i]["a"]  = rows[ROW_ALP]
        raw_events[i]["g1"] = rows[ROW_G1]
        raw_events[i]["g2"] = rows[ROW_G2]
    return raw_events


def raw_to_events(raw_events, gaggs, ma):
    """
    Convert a list of raw four-momentum arrays into fully processed event dicts.

    For each event the function:
    * Computes kinematic variables (eta, phi, pT, |p|, E) for the ALP and
      both photons.
    * For the ALP: computes the decay length for each value in *gaggs*.
    * For each photon: samples the conversion probability and conversion point
      for each value in *gaggs*, accounting for the finite ALP decay length.

    Parameters
    ----------
    raw_events : list of dict
        Output of ``read_data``.
    gaggs : array-like of float
        Grid of ALP–photon couplings g_{agg} in GeV^{-1} to evaluate.
    ma : float
        ALP mass in GeV.

    Returns
    -------
    list of dict
        Each element has keys 'a', 'g1', 'g2'.  Each particle sub-dict
        contains arrays indexed by event and/or g_{agg} grid point.
    """
    events = []
    for raw in raw_events:
        ev = copy.deepcopy(_event_template)
        for ptcl in ('a', 'g1', 'g2'):
            mom  = raw[ptcl]
            p    = np.sqrt(mom[1]**2 + mom[2]**2 + mom[3]**2)
            pt   = np.sqrt(mom[1]**2 + mom[2]**2)
            eta  = np.arctanh(mom[3] / p)
            phi  = np.arctan2(mom[2], mom[1])
            ev[ptcl]['eta'] = eta
            ev[ptcl]['phi'] = phi
            ev[ptcl]['pt']  = pt
            ev[ptcl]['p']   = p
            ev[ptcl]['E']   = mom[0]
            if ptcl == 'a':
                ev[ptcl]['l'] = np.array([decay_length(ma, p, g) for g in gaggs])
            else:
                result = np.array([
                    conv_prob_finite_lifetime(eta, ma, ev['a']['p'], g)
                    for g in gaggs
                ])
                ev[ptcl]['p_conv'],  \
                ev[ptcl]['conv'],    \
                ev[ptcl]['l_track'] = result.T
        events.append(ev)
    return events

################################################
## Physics calculations
################################################

def separation_TRT(eta1, eta2, eta_a, phi1, phi2, phi_a, alp_decay_length):
    """
    Compute the transverse separation (in metres) between the two photon
    conversion tracks at the outer wall of the TRT barrel/endcap.

    The separation is estimated geometrically: the angular opening between
    each photon track and the ALP direction is converted into a transverse
    distance at the outer TRT radius, corrected for the ALP decay vertex
    position.

    Parameters
    ----------
    eta1, eta2 : float
        Pseudorapidities of photon 1 and photon 2.
    eta_a : float
        Pseudorapidity of the ALP.
    phi1, phi2 : float
        Azimuthal angles of the two photons (rad).
    phi_a : float
        Azimuthal angle of the ALP (rad).
    alp_decay_length : float
        ALP decay length in metres (used to find the effective radius at
        which tracks start).

    Returns
    -------
    float
        Transverse separation in metres, or -1 if outside TRT acceptance.
    """
    # Sort photons by |eta| so that etas[1] is the more forward photon
    etas = np.array(sorted([eta1, eta2], key=abs))
    phis = np.array([x for _, x in sorted(zip([abs(eta1), abs(eta2)], [phi1, phi2]))])
    thetas = 2.0 * np.arctan(np.exp(-etas))
    theta_a = 2.0 * np.arctan(np.exp(-eta_a))

    aeta1 = abs(etas[1])
    if aeta1 < eta_max:
        rho = R_max_TRT / abs(np.sin(theta_a)) - alp_decay_length
    elif eta_min_endcap < aeta1 < eta_corner_endcap:
        rho = R_max_TRT_endcap / abs(np.sin(theta_a)) - alp_decay_length
    elif eta_corner_endcap <= aeta1 < eta_max_endcap:
        rho = z_max_TRT_endcap / abs(np.cos(theta_a)) - alp_decay_length
    else:
        rho = -1.

    if rho < 0:
        return -1.

    cos1 = (np.sin(thetas[0]) * np.sin(theta_a) * np.cos(phis[0] - phi_a)
            + np.cos(thetas[0]) * np.cos(theta_a))
    cos2 = (np.sin(thetas[1]) * np.sin(theta_a) * np.cos(phis[1] - phi_a)
            + np.cos(thetas[1]) * np.cos(theta_a))
    return rho * (abs(np.tan(np.arccos(cos1))) + abs(np.tan(np.arccos(cos2))))


def displaced_vertex_TRT_one_photon(eta_a, phi_phot, phi_a,
                                     l_track, alp_decay_length, track_resolution):
    """
    Estimate the displaced-vertex impact parameter (metres) for a single
    converted photon track, given the finite track angular resolution.

    Parameters
    ----------
    eta_a : float
        ALP pseudorapidity (used to get TRT length along ALP direction).
    phi_phot : float
        Azimuthal angle of the photon (rad).
    phi_a : float
        Azimuthal angle of the ALP (rad).
    l_track : float
        Remaining track length available after conversion point (m).
    alp_decay_length : float
        ALP decay length in metres.
    track_resolution : float
        Spatial resolution of the tracker for the photon direction (m).

    Returns
    -------
    d : float
        Estimated displacement of the reconstructed vertex from the IP (m).
    beta : float
        Opening angle between the photon track and the ALP direction after
        subtracting the angular resolution (rad).
    """
    L_TRT  = TRT_length(eta_a)
    theta_1 = abs(phi_phot - phi_a)
    alpha_1 = np.arctan(track_resolution / l_track)
    beta_1  = theta_1 - alpha_1

    if beta_1 < 0:
        return 0., beta_1

    a_1 = track_resolution / (2.0 * np.sin(theta_1))
    c_1 = L_TRT - alp_decay_length - a_1

    if c_1 < 0:
        return 0., beta_1

    # Quadratic to find the reconstructed vertex position along the track
    c1 = 1.
    c2 = -2.0 * L_TRT * (1.0 + 4.0 * np.tan(theta_1)**2 * c_1 / L_TRT) / (1.0 + 4.0 * np.tan(theta_1)**2)
    c3 = 4.0 * np.tan(theta_1)**2 / (1.0 + 4.0 * np.tan(theta_1)**2)
    coeffs = np.array([c1, c2, c3])

    if np.any(np.isnan(coeffs)):
        return 0., beta_1

    s_1 = min(np.real(np.roots(coeffs)))
    h_1 = np.tan(theta_1) * (c_1 - s_1)
    d_1 = L_TRT - s_1 - h_1 / np.tan(beta_1)
    return d_1, beta_1


def displaced_vertex_TRT(eta_a, phi1, phi2, phi_a,
                          l_track1, l_track2, alp_decay_length, track_resolution):
    """
    Estimate the displaced-vertex impact parameter (metres) combining the two
    photon tracks from an ALP decay.

    Parameters
    ----------
    eta_a : float
        ALP pseudorapidity.
    phi1, phi2 : float
        Azimuthal angles of photon 1 and photon 2 (rad).
    phi_a : float
        Azimuthal angle of the ALP (rad).
    l_track1, l_track2 : float
        Available track lengths for the two photons (m).
    alp_decay_length : float
        ALP decay length in metres.
    track_resolution : float
        Spatial resolution of the tracker (m).

    Returns
    -------
    float
        Combined impact parameter estimate in metres.
    """
    d_1, beta_1 = displaced_vertex_TRT_one_photon(
        eta_a, phi1, phi_a, l_track1, alp_decay_length, track_resolution)
    d_2, beta_2 = displaced_vertex_TRT_one_photon(
        eta_a, phi2, phi_a, l_track2, alp_decay_length, track_resolution)

    if d_1 <= 0 and d_2 <= 0:
        return 0.
    if d_1 <= 0:
        return d_2 * np.sin(beta_2)
    if d_2 <= 0:
        return d_1 * np.sin(beta_1)

    # Both tracks valid: combine geometrically
    ds    = sorted([d_1, d_2], key=abs)
    betas = [beta_1, beta_2] if abs(d_1) <= abs(d_2) else [beta_2, beta_1]
    x     = np.sin(betas[1]) / (np.sin(betas[0]) + np.sin(betas[1])) * (ds[1] - ds[0])
    h     = x * np.sin(beta_1)
    return np.sqrt(h**2 + (ds[0] + x)**2)


def Delta_R(eta1, eta2, phi1, phi2, alp_decay_length):
    """
    Compute the effective Delta-R between the two photons at the ECAL,
    corrected for the displaced ALP decay vertex.

    For a decay at distance *l* from the IP, the angular separation shrinks
    by a factor (R_ECAL - l) / R_ECAL.

    Parameters
    ----------
    eta1, eta2 : float
        Photon pseudorapidities.
    phi1, phi2 : float
        Photon azimuthal angles (rad).
    alp_decay_length : float
        ALP decay length in metres.

    Returns
    -------
    float
        Effective Delta-R (dimensionless).
    """
    Delta_phi = abs(phi1 - phi2)
    Delta_eta = abs(eta1 - eta2)
    D_R       = np.sqrt(Delta_phi**2 + Delta_eta**2)
    scale     = min((R_ECAL - alp_decay_length) / R_ECAL, 1.0)
    return scale * D_R

################################################
## Batch analysis
################################################

def calculate_separations_2converted_displaced_isolated(
        events, gaggs, pTcut=0., track_resolution=1.0e-4, check=False):
    """
    For each coupling in *gaggs*, select events where:
      - The ALP has pT > *pTcut* (GeV), and
      - Both photons convert inside the TRT,

    and compute for each such event:
      - Transverse track separation at the outer TRT wall (m).
      - Average photon pseudorapidity.
      - Displaced-vertex impact parameter (m).
      - Delta-R at the ECAL (corrected for displacement).
      - Delta-eta at the ECAL (displacement-corrected, phi=0 for both photons).

    Parameters
    ----------
    events : list of dict
        Output of ``raw_to_events``.
    gaggs : array-like of float
        Grid of couplings g_{agg} (GeV^{-1}).
    pTcut : float, optional
        Minimum ALP transverse momentum in GeV.  Default 0.
    track_resolution : float, optional
        Tracker spatial resolution in metres.  Default 1e-4 m = 0.1 mm.
    check : bool, optional
        If True, also return per-photon track lengths and event indices for
        debugging.  Default False.

    Returns
    -------
    separations : list of ndarray
        One array per coupling value; each element is a track separation (m).
        Events outside TRT acceptance are flagged with -1 and should be masked
        before applying a separation cut.
    average_etas : list of ndarray
        Mean pseudorapidity of the two photons for each selected event.
    impact_parameters : list of ndarray
        Displaced-vertex impact parameter (m) for each selected event.
    delta_Rs : list of ndarray
        Effective Delta-R at the ECAL for each selected event.
    delta_etas : list of ndarray
        Effective Delta-eta at the ECAL (phi=0 for both photons).

    If *check* is True, two additional lists are returned:
    ltrack_list : list of ndarray
        Track lengths [l_track1, l_track2] per selected event per coupling.
    i_list : list of ndarray
        Original event indices for each selected event.
    """
    seps_list    = []
    av_eta_list  = []
    imp_list     = []
    DR_list      = []
    Deta_list    = []
    ltrack_list  = []
    i_list       = []

    for i_g, g in enumerate(gaggs):
        seps      = []
        av_eta    = []
        imp_params = []
        delta_rs  = []
        delta_etas= []
        ltracks   = []
        i_vals    = []

        for i, ev in enumerate(events):
            passes_pT   = ev['a']['pt'] > pTcut
            both_conv   = ev['g1']['conv'][i_g] and ev['g2']['conv'][i_g]
            if not (passes_pT and both_conv):
                continue

            eta1  = ev['g1']['eta'];  eta2  = ev['g2']['eta'];  eta_a = ev['a']['eta']
            phi1  = ev['g1']['phi'];  phi2  = ev['g2']['phi'];  phi_a = ev['a']['phi']
            l1    = ev['g1']['l_track'][i_g]
            l2    = ev['g2']['l_track'][i_g]
            l_a   = ev['a']['l'][i_g]

            seps.append(separation_TRT(eta1, eta2, eta_a, phi1, phi2, phi_a, l_a))
            av_eta.append((eta1 + eta2) / 2.0)
            imp_params.append(displaced_vertex_TRT(eta_a, phi1, phi2, phi_a,
                                                    l1, l2, l_a, track_resolution))
            delta_rs.append(Delta_R(eta1, eta2, phi1, phi2, l_a))
            delta_etas.append(Delta_R(eta1, eta2, 0., 0., l_a))
            ltracks.extend([l1, l2])
            i_vals.append(i)

        seps_list.append(np.array(seps))
        av_eta_list.append(np.array(av_eta))
        imp_list.append(np.array(imp_params))
        DR_list.append(np.array(delta_rs))
        Deta_list.append(np.array(delta_etas))
        ltrack_list.append(np.array(ltracks))
        i_list.append(np.array(i_vals, dtype=int))

    if check:
        return seps_list, av_eta_list, imp_list, DR_list, Deta_list, ltrack_list, i_list
    return seps_list, av_eta_list, imp_list, DR_list, Deta_list


def calculate_splittings(events, gaggs, pTcut=0., B_TRT=2.):
    """
    Compute the average e⁺e⁻ splitting from photon conversions for each event.

    For each event where both photons convert, the splitting is estimated as
    the mean over the two photons of l_track^2 / r_curv, where r_curv is the
    cyclotron radius of the conversion pair in the magnetic field B_TRT.

    Parameters
    ----------
    events : list of dict
        Output of ``raw_to_events``.
    gaggs : array-like of float
        Grid of couplings in GeV^{-1}.
    pTcut : float, optional
        Minimum ALP pT in GeV.  Default 0.
    B_TRT : float, optional
        Magnetic field strength in the TRT in Tesla.  Default 2 T (ATLAS solenoid).

    Returns
    -------
    list of ndarray
        One array per coupling value; each element is a splitting value (m).
    """
    splits_list = []
    for i_g in range(len(gaggs)):
        splits = []
        for ev in events:
            if ev['a']['pt'] > pTcut and ev['g1']['conv'][i_g] and ev['g2']['conv'][i_g]:
                pts    = np.array([ev['g1']['pt'], ev['g2']['pt']])
                ltrs   = np.array([ev['g1']['l_track'][i_g], ev['g2']['l_track'][i_g]])
                r_curv = pts / 2.0 / (0.3 * B_TRT)   # cyclotron radius (m)
                splits.append(np.mean(ltrs**2 / r_curv))
        splits_list.append(np.array(splits))
    return splits_list

"""
module_VBF.py
=============
Physics utility module for analysing ALP (axion-like particle) production via
vector-boson fusion (VBF) at the LHC, with the ALP subsequently decaying to two
photons inside the tracking detector.

The module implements:
  - Data I/O: reading pre-processed Madgraph event CSVs (jets + ALP + photons).
  - Kinematics: 4-momentum parsing, decay-length calculation.
  - Detector simulation (CMS Run 3 geometry):
      * Photon conversion probability as a function of eta, including finite
        ALP decay-length corrections.
      * TRT-equivalent tracker path length as a function of eta.
      * Transverse separation of the two conversion tracks at the outer
        tracker wall.
      * Impact-parameter (displaced-vertex) significance.
      * Delta-R between the two photons at the ECAL.

Detector geometry
-----------------
Geometry constants in the "Detector geometry" section correspond to the
**CMS Run 3** silicon tracker barrel + endcap and ECAL inner face.  If you
are analysing a different experiment, replace those values and update the
photon-conversion table (``conv_fr`` / ``std_conv_fr``).

Photon conversion table
-----------------------
``conv_fr`` is a 2-row array: row 0 is the upper edges of |η| bins and row 1
is the true conversion fraction in that bin.  Only the 'true' kind is
supported (the ATLAS-paper-style fake/reco unfolding is not used).

Input data format
-----------------
Each event is stored as 7 consecutive rows in a semicolon-delimited CSV file.
Each row contains the four-momentum [E, px, py, pz] of one particle,
comma-separated:

  Row 0: incoming quark 1   (skipped)
  Row 1: incoming quark 2   (skipped)
  Row 2: VBF jet 1          → stored as 'j1'
  Row 3: VBF jet 2          → stored as 'j2'
  Row 4: ALP  (a)           → stored as 'a'
  Row 5: photon 1 (g1)      → stored as 'g1'
  Row 6: photon 2 (g2)      → stored as 'g2'

CSV files must be placed in ``data/<run_name>.csv`` relative to the working
directory (or pass a full path to ``read_data``).  See ``lhe_to_csv.py`` for
the script that converts Madgraph LHE output to this format.
"""

################################################
## Load packages
################################################

import numpy as np
from numpy.random import uniform
import pandas as pd
import os

################################################
## Detector geometry  (Updated for CMS Run 3, April 9, 2026, based on https://cds.cern.ch/record/1129810/files/jinst8_08_s08004.pdf )
## ─────────────────────────────────────────────
## Replace these values if you use a different detector.
################################################

# Tracker barrel
#R_min_TRT        = 0.55   # m  – inner radius of barrel  (not used in current calculations)
#R_max_TRT        = 1.10   # m  – outer radius of barrel
#z_max_TRT        = 1.18   # m  – half-length of barrel active volume
#
#
#
## TRT endcap
#z_min_TRT_endcap = 1.24  # m  – inner z-edge of endcap active volume
#z_max_TRT_endcap = 2.82  # m  – outer z-edge of endcap active volume
#R_min_TRT_endcap = 0.225  # m  – inner radius of endcap
#R_max_TRT_endcap = 1.135  # m  – outer radius of endcap
#
## ECAL
#R_ECAL = 1.29  # m  – CMS ECAL barrel inner face


## Detector geometry — CMS Phase-2 (HL-LHC)
## Source: CMS Phase-2 Tracker TDR, CERN-LHCC-2017-009
## ─────────────────────────────────────────────

# Outer Tracker barrel  (≡ "TRT barrel" role)
R_min_TRT        = 0.230   # m  – inner radius of OT barrel (T5 layer, ~23 cm)
R_max_TRT        = 1.100   # m  – outer radius of OT barrel (T6 layer, ~110 cm)
z_max_TRT        = 1.200   # m  – half-length of OT barrel active volume

# Outer Tracker endcap  (≡ "TRT endcap" role)
z_min_TRT_endcap = 1.30    # m  – inner z-edge of OT endcap active volume
z_max_TRT_endcap = 2.80    # m  – outer z-edge of OT endcap active volume
R_min_TRT_endcap = 0.200   # m  – inner radius of OT endcap
R_max_TRT_endcap = 1.100   # m  – outer radius of OT endcap

# ECAL (unchanged in Phase-2)
R_ECAL = 1.29              # m  – CMS ECAL barrel inner face

# Derived pseudorapidity boundaries (do not edit)
eta_min           = 0.
eta_max           = -np.log(np.tan(0.5 * np.arctan(R_max_TRT / z_max_TRT)))
eta_min_endcap    = -np.log(np.tan(0.5 * np.arctan(R_max_TRT_endcap / z_min_TRT_endcap)))
eta_max_endcap    = -np.log(np.tan(0.5 * np.arctan(R_min_TRT_endcap / z_max_TRT_endcap)))
eta_corner_endcap = -np.log(np.tan(0.5 * np.arctan(R_max_TRT_endcap / z_max_TRT_endcap)))

################################################
## Photon conversion fractions  (CMS Run 3, placeholder values)
## ─────────────────────────────────────────────
## conv_fr[0]  : upper edges of |eta| bins
## conv_fr[1]  : true converted fraction in that |eta| bin
##
## NOTE: the 0.4 entries are a flat placeholder pending real CMS Run 3
## measurements.  The last column (eta -> inf) is a dummy "no-acceptance" bin.
################################################

conv_fr = np.array([
    [0.6,  1.37, 1.52, 1.81, 2.37, np.inf],   # |eta| upper bin edges
    [0.4,  0.4,  0.4,  0.4,  0.4,  0.0],      # true converted fraction (placeholder)
])

std_conv_fr = np.array([
    [0.6,  1.37, 1.52, 1.81, 2.37, np.inf],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

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


def decay_length_batch(ma, momentum, gaggs):
    """Same physics as ``decay_length`` but vectorized over *gaggs* (1-D array)."""
    gaggs = np.asarray(gaggs, dtype=np.float64)
    gamma = momentum / ma
    ctau_iGeV = 64.0 * np.pi * gamma / (gaggs * gaggs * ma**3)
    return ctau_iGeV * 1.0e-9 * 1.973e-7


def _invert_cumulative_bisect(random2, d_length, L_tracker, f_conv, c_prob, maxiter=80):
    """
    Solve cumulative(x) == random2 for x in (0, L_tracker), with *cumulative*
    defined as in ``conv_prob_finite_lifetime_novec``.  Uses bisection only
    (no SciPy) — this is the hot path for converted photons.
    """
    fc_over_cp = f_conv / c_prob
    dl = d_length
    Lt = L_tracker

    def F(x):
        return fc_over_cp * (
            1.0 - dl / Lt * (
                1.0 - (1.0 - (Lt - x) / dl) * np.exp(-x / dl)
            )
        ) - random2

    a, b = 0.0, Lt
    fa, fb = F(a), F(b)
    if not (fa * fb < 0):
        return None

    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = F(m)
        if abs(fm) <= 1e-14 * max(1.0, abs(random2)):
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

################################################
## Photon conversion probability
################################################

def conv_prob_novec(eta):
    """
    Return the true photon conversion probability for a single pseudorapidity
    value, looked up from the ``conv_fr`` table.

    Parameters
    ----------
    eta : float
        Photon pseudorapidity.

    Returns
    -------
    float
        True converted fraction in the |η| bin containing *eta*.
    """
    i = np.searchsorted(conv_fr[0], abs(eta))
    return conv_fr[1, i]


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

def conv_prob_finite_lifetime_novec(eta, ma, pa, gagg):
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

    Returns
    -------
    c_prob : float
        Effective conversion probability (scalar).
    conv : bool
        Whether the photon converts (Monte Carlo decision).
    track_length : float
        If converted, the track length inside the tracker in metres; else 0.
    """
    alp_decay_length = decay_length(ma, pa, gagg)
    f_conv = conv_prob_novec(eta)
    L_tracker = TRT_length_novec(eta)

    if L_tracker == 0.:
        c_prob = 0.0
    else:
        c_prob = f_conv * (
            1.0 - (1.0 - np.exp(-L_tracker / alp_decay_length)) * alp_decay_length / L_tracker
        )

    random = uniform()
    conv = random < c_prob

    if conv:
        random2 = uniform()
        c0 = f_conv / c_prob * (
            1.0 - alp_decay_length / L_tracker * (
                1.0 - (1.0 - L_tracker / alp_decay_length) * 1.0
            )
        )
        cL = f_conv / c_prob * (
            1.0 - alp_decay_length / L_tracker * (
                1.0 - (1.0 - 0.0 / alp_decay_length) * np.exp(-L_tracker / alp_decay_length)
            )
        )
        bracket_ok = (c0 - random2) * (cL - random2) < 0
        if bracket_ok:
            root = _invert_cumulative_bisect(random2, alp_decay_length, L_tracker, f_conv, c_prob)
            if root is not None:
                track_length = L_tracker - root
            else:
                track_length = 0.0
        else:
            track_length = 0.0
    else:
        track_length = 0.0

    return c_prob, conv, track_length


def conv_prob_finite_lifetime_batch(eta, ma, pa, gaggs):
    """
    Same Monte Carlo as ``conv_prob_finite_lifetime_novec`` for each coupling,
    with identical RNG consumption order as a scalar loop over *gaggs*
    (two ``uniform()`` calls per coupling when the photon converts).
    Returns ``p_conv``, ``conv``, ``l_track`` with shape ``(len(gaggs),)``.
    """
    gaggs = np.asarray(gaggs, dtype=np.float64)
    n = gaggs.shape[0]
    p_out = np.empty(n, dtype=np.float64)
    c_out = np.empty(n, dtype=bool)
    l_out = np.empty(n, dtype=np.float64)

    f_conv = conv_prob_novec(eta)
    L_tracker = TRT_length_novec(eta)

    if L_tracker == 0.0:
        p_out.fill(0.0)
        c_out.fill(False)
        l_out.fill(0.0)
        return p_out, c_out, l_out

    d_lengths = decay_length_batch(ma, pa, gaggs)
    exp_term = np.exp(-L_tracker / d_lengths)
    p_out[:] = f_conv * (
        1.0 - (1.0 - exp_term) * d_lengths / L_tracker
    )

    for i in range(n):
        d_length = d_lengths[i]
        c_prob = p_out[i]
        random = uniform()
        conv = random < c_prob
        c_out[i] = conv
        if not conv:
            l_out[i] = 0.0
            continue
        random2 = uniform()
        c0 = f_conv / c_prob * (
            1.0 - d_length / L_tracker * (
                1.0 - (1.0 - L_tracker / d_length) * 1.0
            )
        )
        cL = f_conv / c_prob * (
            1.0 - d_length / L_tracker * (
                1.0 - (1.0 - 0.0 / d_length) * np.exp(-L_tracker / d_length)
            )
        )
        if (c0 - random2) * (cL - random2) < 0:
            root = _invert_cumulative_bisect(random2, d_length, L_tracker, f_conv, c_prob)
            l_out[i] = (L_tracker - root) if root is not None else 0.0
        else:
            l_out[i] = 0.0

    return p_out, c_out, l_out

################################################
## I/O: reading Madgraph CSV files
################################################

# ── Row-index map for the 7-row-per-event CSV format ─────────────────────────
# Each event occupies exactly ROWS_PER_EVENT consecutive rows in the file.
# Rows 0–1 (incoming quarks) are skipped; rows 2–6 are stored on the event
# dict produced by ``read_data`` / ``raw_to_events``.
#
# *** If your CSV was produced by a different script and has a different
#     particle ordering, change ROW_J1/ROW_J2/ROW_ALP/ROW_G1/ROW_G2 to match.
#     Use ``print_first_event`` to verify the ordering. ***

ROWS_PER_EVENT = 7     # total rows written per event (including skipped particles)
ROW_J1         = 2     # row index of VBF jet 1 four-momentum   → 'j1'
ROW_J2         = 3     # row index of VBF jet 2 four-momentum   → 'j2'
ROW_ALP        = 4     # row index of the ALP  four-momentum    → 'a'
ROW_G1         = 5     # row index of photon 1 four-momentum    → 'g1'
ROW_G2         = 6     # row index of photon 2 four-momentum    → 'g2'

# Skipped rows (for documentation only — not enforced):
#   Row 0: incoming quark 1
#   Row 1: incoming quark 2


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
    constants ``ROW_J1``, ``ROW_J2``, ``ROW_ALP``, ``ROW_G1``, ``ROW_G2``
    (defaults: 2, 3, 4, 5, 6).  If your CSV has a different ordering, change
    those constants.  Call ``print_first_event`` to verify the ordering
    before running a full analysis.

    Default row layout (produced by ``lhe_to_csv.py``):

        Row 0: incoming quark 1   (skipped)
        Row 1: incoming quark 2   (skipped)
        Row 2: VBF jet 1          → stored as 'j1'
        Row 3: VBF jet 2          → stored as 'j2'
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
        Each element corresponds to one event and has keys ``'j1'``, ``'j2'``,
        ``'a'``, ``'g1'``, ``'g2'``, each holding a NumPy array
        ``[E, px, py, pz]``.
    """
    filepath = os.path.join(data_dir, run_name + '.csv')
    temp = pd.read_csv(filepath, sep=';', header=None, nrows=ROWS_PER_EVENT * num)
    length = min(len(temp) // ROWS_PER_EVENT, num)
    if length == 0:
        return []

    nrows = length * ROWS_PER_EVENT
    coords = (
        temp.iloc[:nrows, 0]
        .str.split(',', expand=True)
        .to_numpy(dtype=np.float64)
        .reshape(length, ROWS_PER_EVENT, 4)
    )
    raw_events = []
    for i in range(length):
        raw_events.append({
            'a': coords[i, ROW_ALP],
            'g1': coords[i, ROW_G1],
            'g2': coords[i, ROW_G2],
            'j1': coords[i, ROW_J1],
            'j2': coords[i, ROW_J2],
        })
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
        Each element has keys ``'a'``, ``'g1'``, ``'g2'``, ``'j1'``, ``'j2'``.
        Each particle sub-dict contains scalar kinematics (eta, phi, pt, p, E,
        px, py, pz); the ALP entry additionally has ``'l'`` (decay length per
        g_agg), and the photon entries have ``'p_conv'``, ``'conv'``, and
        ``'l_track'`` arrays indexed by g_agg grid point.
    """
    gaggs_arr = np.asarray(gaggs, dtype=np.float64)
    events = []
    for raw in raw_events:
        ev = {'a': {}, 'g1': {}, 'g2': {}, 'j1': {}, 'j2': {}}
        for ptcl in ('a', 'g1', 'g2', 'j1', 'j2'):
            mom = raw[ptcl]
            p = np.sqrt(mom[1]**2 + mom[2]**2 + mom[3]**2)
            pt = np.sqrt(mom[1]**2 + mom[2]**2)
            eta = np.arctanh(mom[3] / p)
            phi = np.arctan2(mom[2], mom[1])
            ev[ptcl]['eta'] = eta
            ev[ptcl]['phi'] = phi
            ev[ptcl]['pt'] = pt
            ev[ptcl]['p'] = p
            ev[ptcl]['E'] = mom[0]
            ev[ptcl]['px'] = mom[1]
            ev[ptcl]['py'] = mom[2]
            ev[ptcl]['pz'] = mom[3]

            if ptcl == 'a':
                ev[ptcl]['l'] = decay_length_batch(ma, p, gaggs_arr)
            elif ptcl in ('g1', 'g2'):
                pc, cv, lt = conv_prob_finite_lifetime_batch(
                    eta, ma, ev['a']['p'], gaggs_arr
                )
                ev[ptcl]['p_conv'] = pc
                ev[ptcl]['conv'] = cv
                ev[ptcl]['l_track'] = lt
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

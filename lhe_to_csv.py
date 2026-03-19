"""
lhe_to_csv.py
=============
Convert a Madgraph LHE (Les Houches Event) file to the semicolon/comma CSV
format expected by ``module_VBF.read_data``.

The script reads the LHE file, extracts the four-momenta of the seven final-
and initial-state particles from each event block, and writes them to a CSV
file where:

  - Rows are separated by semicolons  (one row per particle per event).
  - Within each row, the four values [E, px, py, pz] are comma-separated.
  - There is NO header row.

The particle order written per event is:

    Row 0: incoming parton 1        (status = -1)
    Row 1: incoming parton 2        (status = -1)
    Row 2: VBF jet 1                (status =  1, not ALP/photon)
    Row 3: VBF jet 2                (status =  1, not ALP/photon)
    Row 4: ALP  (PDG ID = 36)  (status =  1)
    Row 5: photon 1 (PDG ID = 22)   (status =  1)
    Row 6: photon 2 (PDG ID = 22)   (status =  1)

This matches the format that ``read_data`` in ``module_VBF.py`` expects.

Usage
-----
    python lhe_to_csv.py <input.lhe> <output.csv> [--max-events N]

Arguments
---------
input.lhe    : path to the Madgraph LHE file (may be gzip-compressed: .lhe.gz)
output.csv   : path to write the output CSV
--max-events : (optional) stop after this many events (useful for testing)

Example
-------
    python lhe_to_csv.py Events/run_01/unweighted_events.lhe.gz data/01GeV.csv
    python lhe_to_csv.py Events/run_01/unweighted_events.lhe     data/01GeV.csv --max-events 10000

Notes
-----
- The script expects exactly 2 photons (PDG 22) and 1 ALP (PDG 36) among
  the final-state particles.  Events that do not match this topology are
  skipped with a warning.
- The ALP PDG ID (36) can be changed via the ``ALP_PDG_ID`` constant
  below if your UFO model uses a different code.
- The script is intentionally kept self-contained (stdlib + standard LHE
  parsing) so it can be run in any environment without extra dependencies.
"""

import sys
import os
import argparse
import gzip
from xml.etree import ElementTree as ET


# ── Change this if your UFO model uses a different PDG code for the ALP ──
ALP_PDG_ID = 36#9000005
PHOTON_PDG_ID = 22


def open_lhe(path):
    """Open an LHE file, transparently handling .lhe.gz."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')


def parse_lhe(lhe_path, csv_path, max_events=None):
    """
    Parse *lhe_path* and write the converted CSV to *csv_path*.

    Parameters
    ----------
    lhe_path  : str  – input LHE file path (.lhe or .lhe.gz)
    csv_path  : str  – output CSV file path
    max_events: int or None – stop after this many events
    """
    n_written = 0
    n_skipped = 0

    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

    with open_lhe(lhe_path) as lhe_fh, open(csv_path, 'w') as csv_fh:
        in_event  = False
        particles = []   # list of (status, pdg_id, E, px, py, pz)

        for raw_line in lhe_fh:
            line = raw_line.strip()

            if line == '<event>':
                in_event  = True
                particles = []
                continue

            if line == '</event>':
                in_event = False
                rows = _process_event(particles)
                if rows is None:
                    n_skipped += 1
                else:
                    for row in rows:
                        csv_fh.write(','.join(f'{v:.10e}' for v in row) + ';' + '\n')
                    n_written += 1
                    if max_events and n_written >= max_events:
                        break
                continue

            if in_event and line and not line.startswith('<') and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        pdg_id = int(parts[0])
                        status = int(parts[1])
                        px     = float(parts[6])
                        py     = float(parts[7])
                        pz     = float(parts[8])
                        E      = float(parts[9])
                        particles.append((status, pdg_id, E, px, py, pz))
                    except ValueError:
                        pass   # skip non-numeric lines (e.g. event-info header line)

    print(f'Written {n_written} events to {csv_path}  ({n_skipped} skipped)')


def _process_event(particles):
    """
    Extract the 7-row [E, px, py, pz] block for one event.

    Returns None if the event topology does not match the expected VBF→ALP→γγ.
    """
    incoming = [(E, px, py, pz)
                for status, pdg, E, px, py, pz in particles
                if status == -1]
    outgoing = [(pdg, E, px, py, pz)
                for status, pdg, E, px, py, pz in particles
                if (status == 1 or status ==2)]

    alp_rows     = [(E, px, py, pz) for pdg, E, px, py, pz in outgoing if pdg == ALP_PDG_ID]
    photon_rows  = [(E, px, py, pz) for pdg, E, px, py, pz in outgoing if pdg == PHOTON_PDG_ID]
    jet_rows     = [(E, px, py, pz) for pdg, E, px, py, pz in outgoing
                    if pdg not in (ALP_PDG_ID, PHOTON_PDG_ID)]

    # Validate topology
    if len(incoming)   != 2:
        print(f'Warning: expected 2 incoming partons, got {len(incoming)} — skipping event')
        return None
    if len(alp_rows)   != 1:
        print(f'Warning: expected 1 ALP (PDG {ALP_PDG_ID}), got {len(alp_rows)} — skipping event')
        return None
    if len(photon_rows) != 2:
        print(f'Warning: expected 2 photons, got {len(photon_rows)} — skipping event')
        return None
    if len(jet_rows)   != 2:
        print(f'Warning: expected 2 VBF jets, got {len(jet_rows)} — skipping event')
        return None

    rows = (
        list(incoming)     +   # rows 0-1: incoming quarks
        list(jet_rows)     +   # rows 2-3: VBF jets
        list(alp_rows)     +   # row  4:   ALP
        list(photon_rows)      # rows 5-6: photons
    )
    return rows   # each element is (E, px, py, pz)


def main():
    parser = argparse.ArgumentParser(
        description='Convert a Madgraph LHE file to the CSV format for module_VBF.')
    parser.add_argument('input',  help='Input LHE file (.lhe or .lhe.gz)')
    parser.add_argument('output', help='Output CSV file path  (e.g. data/01GeV.csv)')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Stop after this many events (default: all)')
    args = parser.parse_args()

    parse_lhe(args.input, args.output, max_events=args.max_events)


if __name__ == '__main__':
    main()

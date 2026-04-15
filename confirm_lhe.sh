#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please run via: source confirm_lhe.sh <path/to/unweighted_events.lhe[.gz]>"
  return 2 2>/dev/null || exit 2
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: source confirm_lhe.sh <path/to/unweighted_events.lhe[.gz]>"
  return 2 2>/dev/null || exit 2
fi

LHE_PATH="$1"
if [[ ! -f "$LHE_PATH" ]]; then
  echo "ERROR: file not found: $LHE_PATH"
  return 2 2>/dev/null || exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_OUT="${ROOT_DIR}/data/confirm_$(basename "$LHE_PATH" | sed 's/[^A-Za-z0-9_.-]/_/g').csv"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "LHE:  $LHE_PATH"
echo "CSV:  $CSV_OUT"
echo "Python: $PYTHON_BIN"

# Extract ALP mass from the LHE file (PDG 36). In LHE particle records,
# the mass appears in column 11 (0-index 10) on the particle line.
MA="$("$PYTHON_BIN" -c "
import gzip, sys
path = sys.argv[1]
op = gzip.open if path.endswith('.gz') else open
with op(path, 'rt') as f:
    in_event = False
    for raw in f:
        line = raw.strip()
        if line == '<event>':
            in_event = True
            continue
        if line == '</event>':
            in_event = False
            continue
        if not in_event or not line or line.startswith('<') or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            pdg = int(parts[0])
            status = int(parts[1])
        except ValueError:
            continue
        if pdg == 36:
            # prefer the decayed-resonance entry if present (status 2), otherwise take what we see
            mass = float(parts[10])
            print(mass)
            raise SystemExit(0)
print('0.0')
" "$LHE_PATH")"

echo "ma (from LHE): ${MA} GeV"

mkdir -p "${ROOT_DIR}/data"

echo "Running LHE -> CSV..."
"$PYTHON_BIN" "${ROOT_DIR}/scripts/lhe_to_csv.py" "$LHE_PATH" "$CSV_OUT"

echo "Running confirmations (ΔR + energy checks)..."
"$PYTHON_BIN" "${ROOT_DIR}/scripts/confirm_lhe.py" "$CSV_OUT" --ma "$MA"

echo "OK: confirmations passed."



import numpy as np
import scipy as scp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd 
import sys
import importlib.util
from pathlib import Path


#add the path to the tracking_ALPs directory
tracking_dir = Path(sys.argv[1])

spec = importlib.util.spec_from_file_location(
    "Analyze_madgraph_output",
    str(tracking_dir / "Analyze_madgraph_output.py"),
)
from pathlib import Path
EC_DIR = next(
    p / 'existing_constraints'
    for p in (Path.cwd(), *Path.cwd().parents)
    if (p / 'existing_constraints').is_dir()
)


### Parameters and constants ###
#l_ATLAS_run2_1 = 37 #fb^-1
#l_ATLAS_run2 = 139 #fb^-1
#l_ATLAS_run3 = 300 #fb^-1
#l_ATLAS_HL = 3 #ab^-1
l_CMS_run3 = 312  # fb^-1
l_CMS_HL = 3000 #fb^-1
l_CMS_HL_one_year = 250 #fb^-1

alpha_em = 1/137
sw2 = 0.223
cw2 = 1 - sw2
xsec_gagg1e2_list = 180*np.ones(56)

ma_list = [
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.1,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    2.0, 5.0, 10.0, 20, 30, 40, 50, 60, 70, 80, 100
]
n_a = 40000
def ma_to_name(ma):
    """Convert an ALP mass (GeV, float) to the filename stub used in data/."""
    return f"{ma:.4f}GeV".replace('.', 'p')
def n2g (n2g_na, na):
    rslt = n2g_na * na
    return rslt

#Read results, return event counts for each mass and gagg combination

def calculate_n2g(filename, lumi):
    """Read results from a CSV file and return event counts for each mass and gagg combination."""
    df = pd.read_csv(filename, index_col=0, header=None)
    gagg_list = df.values[0]
    n2g_na= df.values[1:]/n_a 

    #calculate production cross section
    prod_xsec_pp_a_13TeV_list =  np.array([xsec_gagg1e2_list * (x/1.0E-2)**2 for x in gagg_list])
    n_a_list = lumi/1.0E-3 * prod_xsec_pp_a_13TeV_list 
    n2g_result = np.array([[n2g(n2g_na[i, j], n_a_list[j, i]) for i in range(len(ma_list))] for j in range(len(gagg_list))])

    return n2g_result


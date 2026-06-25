def compute_mjj(ev):
    import numpy as np
    j1 = np.array([ev['j1']['E'], ev['j1']['px'], ev['j1']['py'], ev['j1']['pz']])
    j2 = np.array([ev['j2']['E'], ev['j2']['px'], ev['j2']['py'], ev['j2']['pz']])
    """Dijet invariant mass from two [E, px, py, pz] arrays."""
    s = j1 + j2   # 4-vector sum
    m2 = s[0]**2 - s[1]**2 - s[2]**2 - s[3]**2
    mjj = np.sqrt(max(m2, 0.0))
    return mjj
def ma_to_name(ma):
    token = f'{ma:.4f}'.replace('.', 'p')
    return f'grid_CMSRun3_vbf_ax_ma_{token}GeV'

def ma_to_mass_token(ma):
    token = f'{ma:.4f}'.replace('.', 'p')
    return f'ma_{token}GeV'
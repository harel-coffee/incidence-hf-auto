# PATHS
MLFLOW_URL = 'http://localhost:5000'
OUTPUT_PATH = './output'
DATA_MAIN_PATH = '~/data/homage_170314.csv'
DATA_FLEMENGHO_PATH = '~/data/homage_flemengho.csv'

# PLOTS
TIME_POINT_PREDICTION = 5 * 365
FONT = 'Calibri'
STRIPES_OPACITY = 0.4

COLORS = {'gb': '#FF7F00', 'coxnet': '#4194D9', 'stacking': '#D941C3', 'pcp_hf': '#2BC07B'}


def get_color(name):
    return COLORS.get(name)


METHODS_TITLE = {
    'gb': 'Gradient\nBoosting (GB)', 'coxnet': "CoxNet (CN)", 'pcp_hf': "PCP-HF", 'stacking': "Stacking (ST)"
}

# MLFLOW IDs TO SHOW

GROUPS_LCO_LIMITED = {
    'gb': '52e729f7-0ae3-40a4-b639-8f9f3eaa3a36',
    'coxnet': 'bbcbe7fa-7499-4464-a21e-bb7c6c571886',
}

GROUPS_LCO = {
    'stacking': '5e37865c-7821-4726-a015-56447f4b172d',
    'coxnet': 'bbcbe7fa-7499-4464-a21e-bb7c6c571886',
    'coxnet_ns': 'de06404b-c3c4-45e1-a641-e7bde8860f85',
    'gb': '52e729f7-0ae3-40a4-b639-8f9f3eaa3a36',
    'gb_s': 'a8c4b6a6-c83c-411c-972c-2783595b9a97',
    'pcp_hf': '787ec10f-784e-41dc-8753-7a63dce91fc9',
    # 'dcm': 'eaa42764-afc1-465c-9cd0-1aaf4ce321fc',
    # 'xgb': '11645e13-dd1a-4727-b0a9-b468bf7dc3cd',  # 'stacking2': '79d9a6546a8d4e29ab0db4a0f0e3bc30',
}

GROUPS_LM = {
    'coxnet': '1748b866-20a3-43a2-ac1b-93e41daaca2c',
    'gb': '9de3c9e9-2cb1-4216-b7bb-c395beb5cc81',
}

GROUPS_10_fold = {
    'gb': '7fef3844-51ff-4f7a-a6b7-96030e5733aa',
    'coxnet': '03240b7d-b0bd-4285-839f-dc82a55189ec',
}

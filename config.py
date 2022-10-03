# PATHS
MLFLOW_URL = 'http://localhost:5000'
OUTPUT_PATH = './output'
DATA_MAIN_PATH = '~/data/homage_170314.csv'
DATA_FLEMENGHO_PATH = '~/data/homage_flemengho.csv'

# PLOTS
TIME_POINT_PREDICTION = 5 * 365
FONT = 'Calibri'
STRIPES_OPACITY = 0.4

COLORS = {'gb': '#FF7F00', 'coxnet': '#4194D9', 'stacking': '#D941C3', 'pcp_hf': '#2BC07B', 'svm': '#AAAA00'}

COHORTS = ['PROSPER', 'HEALTHABC', 'PREDICTOR', 'HVC', 'ASCOT', 'FLEMENGHO']

METHODS_TITLE = {
    'gb': 'Gradient\nBoosting (GB)', 'coxnet': "CoxNet (CN)", 'pcp_hf': "PCP-HF", 'stacking': "Stacking (ST)"
}

# MLFlow group_id's

GROUPS_LCO_SELECTED = {
    'gb': '52e729f7-0ae3-40a4-b639-8f9f3eaa3a36',
    'coxnet': 'bbcbe7fa-7499-4464-a21e-bb7c6c571886',
}

GROUPS_LCO = {
    **GROUPS_LCO_SELECTED,
    'stacking': '5e37865c-7821-4726-a015-56447f4b172d',
    'pcp_hf': '787ec10f-784e-41dc-8753-7a63dce91fc9',
    'svm': '50780dac-09d7-43b9-9d77-c0ea879b3fa4',
}

GROUPS_LM_SELECTED = {
    'coxnet': '1748b866-20a3-43a2-ac1b-93e41daaca2c',
    'gb': '9de3c9e9-2cb1-4216-b7bb-c395beb5cc81',
}

GROUPS_10_fold = {
    'gb': '7fef3844-51ff-4f7a-a6b7-96030e5733aa',
    'coxnet': '03240b7d-b0bd-4285-839f-dc82a55189ec',
}


def get_color(name):
    return COLORS.get(name)

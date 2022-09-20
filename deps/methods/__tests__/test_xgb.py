from math import exp

from pandas import DataFrame

from deps.methods.xgb import XGB
from hcve_lib.functional import mapl


def test_xgb():

    model = XGB()
    model.fit(
        DataFrame({'x': [10, 20]}),
        {'data': DataFrame({
            'tte': [50, 20],
            'label': [0, 1],
        })},
    )

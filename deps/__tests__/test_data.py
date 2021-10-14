import numpy as np
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from deps.data import fill_missing_pp


def test_fill_missing_pp():
    assert_frame_equal(
        fill_missing_pp(DataFrame({
            'PP': [50, None],
            'SBP': [140, 150],
            'DBP': [90, 80]
        })),
        DataFrame({
            'PP': [50., 70.],
            'SBP': [140, 150],
            'DBP': [90, 80]
        }),
    )

from math import log, exp

import numpy as np
from pandas import Series
from sklearn.base import BaseEstimator

from hcve_lib.custom_types import ExceptionValue


class PooledCohort(BaseEstimator):

    def fit(self, *args, **kwargs):
        pass
        return self

    def predict(self, X):
        X_ = X.copy()

        def predict_(row: Series) -> float:
            # TODO
            # if row['AGE'] < 30:
            #     return np.nan

            bsug_adj = row['GLU'] * 18.018
            row['CHOL'] = row['CHOL'] * 38.67
            row['HDL'] = row['HDL'] * 38.67
            bmi = row['BMI']
            sex = row['SEX']
            if 'qrs' in row:
                qrs = row['QRS']
            else:
                qrs = 94.03636286848551

            trt_ht = row['TRT_AH']
            if row['SMK'] == 0 or row['SMK'] == 2:
                csmk = 0
            elif row['SMK'] == 1:
                csmk = 1
            ln_age = log(row['AGE'])
            ln_age_sq = ln_age**2
            ln_tchol = log(row['CHOL'])
            ln_hchol = log(row['HDL'])
            ln_sbp = log(row['SBP'])
            hdm = row['DIABETES']
            ln_agesbp = ln_age * ln_sbp
            ln_agecsmk = ln_age * csmk
            ln_bsug = log(bsug_adj)
            ln_bmi = log(bmi)
            ln_agebmi = ln_age * ln_bmi
            ln_qrs = log(qrs)

            if (sex == 1) and (trt_ht == 0): coeff_sbp = 0.91
            if (sex == 1) and (trt_ht == 1): coeff_sbp = 1.03
            if (sex == 2) and (trt_ht == 0): coeff_sbp = 11.86
            if (sex == 2) and (trt_ht == 1): coeff_sbp = 12.95
            if (sex == 2) and (trt_ht == 0): coeff_agesbp = -2.73
            if (sex == 2) and (trt_ht == 1): coeff_agesbp = -2.96

            if (sex == 1) and (hdm == 0): coeff_bsug = 0.78
            if (sex == 1) and (hdm == 1): coeff_bsug = 0.90
            if (sex == 2) and (hdm == 0): coeff_bsug = 0.91
            if (sex == 2) and (hdm == 1): coeff_bsug = 1.04

            if sex == 1:
                IndSum = ln_age * 41.94 + ln_age_sq * (
                    -0.88
                ) + ln_sbp * coeff_sbp + csmk * 0.74 + ln_bsug * coeff_bsug + ln_tchol * 0.49 + ln_hchol * (
                    -0.44
                ) + ln_bmi * 37.2 + ln_agebmi * (-8.83) + ln_qrs * 0.63

                HF_risk = 100 * (1 - (0.98752**exp(IndSum - 171.5)))
            elif sex == 2:
                IndSum = ln_age * 20.55 + ln_sbp * coeff_sbp + ln_agesbp * coeff_agesbp + csmk * 11.02 + ln_agecsmk * (
                    -2.50
                ) + ln_bsug * coeff_bsug + ln_hchol * (-0.07) + ln_bmi * 1.33 + ln_qrs * 1.06
                HF_risk = 100 * (1 - (0.99348**exp(IndSum - 99.73)))

            return HF_risk

        return X.apply(predict_, axis=1)

import xgboost
from pandas import DataFrame
from sklearn.base import BaseEstimator

from hcve_lib.custom_types import Target
from hcve_lib.survival import survival_to_interval, get_event_probability, get_event_case_ratio
from hcve_lib.wrapped_sklearn import DFWrapped


class XGB(DFWrapped, BaseEstimator):
    model = None

    def __init__(
        self,
        silent=None,
        lambda_=None,
        objective=None,
        eval_metric=None,
        booster=None,
        alpha=None,
        max_depth=None,
        eta=None,
        gamma=None,
        grow_policy=None,
        sample_type=None,
        normalize_type=None,
        rate_drop=None,
        skip_drop=None,
        balance_event_rate: bool = False,
    ):
        self.silent = silent
        self.lambda_ = lambda_
        self.objective = objective
        self.eval_metric = eval_metric
        self.booster = booster
        self.alpha = alpha
        self.max_depth = max_depth
        self.eta = eta
        self.gamma = gamma
        self.grow_policy = grow_policy
        self.sample_type = sample_type
        self.normalize_type = normalize_type
        self.rate_drop = rate_drop
        self.skip_drop = skip_drop
        self.balance_event_rate = balance_event_rate

    def fit(self, X, y: Target, **kwargs):
        print(f'{X=}')
        print(f'{X.dtypes=}')
        self.save_fit_features(X)
        if self.balance_event_rate:
            case_ratio = get_event_case_ratio(y['data'])
            weight = y['data']['label'].apply(lambda label: case_ratio if label == 1 else 1)
        else:
            weight = None

        dtrain = xgboost.DMatrix(X, weight=weight)
        y_lower_bound, y_upper_bound = survival_to_interval(y)
        dtrain.set_float_info('label_lower_bound', y_lower_bound)
        dtrain.set_float_info('label_upper_bound', y_upper_bound)

        params = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': 'normal',
            'aft_loss_distribution_scale': 1.20,
            'silent': self.silent,
            'booster': self.booster,
            'lambda': self.lambda_,
            'alpha': self.alpha,
            'max_depth': self.max_depth,
            'eta': self.eta,
            'gamma': self.gamma,
            'grow_policy': self.grow_policy,
            'sample_type': self.sample_type,
            'normalize_type': self.normalize_type,
            'rate_drop': self.rate_drop,
            'skip_drop': self.skip_drop,
        }

        model = xgboost.train(params, dtrain, num_boost_round=5, evals=[(dtrain, 'train')])

        self.model = model

        return self

    def predict(self, X, **kwargs):
        self.save_fit_features(X)
        dtest = xgboost.DMatrix(X)
        y_pred = -self.model.predict(dtest)
        return y_pred

    def predict_survival(self, *args, **kwargs):
        raise NotImplementedError()

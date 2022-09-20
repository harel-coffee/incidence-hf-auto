from pandas import DataFrame
from plotly import express as px
from plotly.graph_objs import Figure
import plotly.graph_objects as go

from config import METHODS_TITLE, COLORS, get_color
from deps.constants import RANDOM_STATE
from deps.memory import memory
from hcve_lib.custom_types import Prediction
from hcve_lib.evaluation_functions import average_group_scores, merge_standardize_prediction, merge_predictions, \
    compute_metrics_prediction
from hcve_lib.functional import reject_none, t
from hcve_lib.metrics import BootstrappedMetric
from hcve_lib.metrics import statistic_from_bootstrap
from hcve_lib.tracking import load_group_results
from hcve_lib.utils import transpose_list, get_first_entry
from hcve_lib.visualisation import setup_plotly_style
from functools import cache


def get_endpoint_analysis_cohort(
    name, y, group_id, metrics, iterations=50, standardize: bool = True, return_summary: bool = False
) -> Prediction:
    merged_prediction = get_endpoint_analysis_prediction(group_id, standardize)
    metrics_ = compute_metrics_prediction(
        list(
            map(
                lambda metric: BootstrappedMetric(
                    metric,
                    RANDOM_STATE,
                    iterations=iterations,
                    return_summary=return_summary,
                ),
                metrics,
            )
        ),
        y,
        merged_prediction
    )
    return metrics_


def get_endpoint_analysis_result(groups, standardize):
    return {
        cohort_name: get_endpoint_analysis_prediction(group_id, standardize)
        for cohort_name,
        group_id in groups.items()
    }


load_group_results_cached = memory.cache(load_group_results)


def get_endpoint_analysis_prediction(group_id, standardize=False):
    group = load_group_results_cached(group_id, load_models=True)
    result = average_group_scores(group)
    merge_prediction_fn = (merge_standardize_prediction if standardize else merge_predictions)
    merged_prediction = merge_prediction_fn(result)
    return merged_prediction


def run_roc_analysis(groups, *args, **kwargs):
    fig = Figure(layout=dict(
        title='ROC',
        xaxis_title='FPR',
        yaxis_title='TPR',
        width=800,
        height=600,
    ))

    for name, group_id in groups.items():
        metrics_ = get_endpoint_analysis_cohort(name, group_id, fig, *args, **kwargs)
        tpr, fpr, threshold = list(metrics_.values())[0]
        plot_data = DataFrame({
            'TPR': tpr,
            'FPR': fpr,  # 'threshold': threshold,
        })
        fig.add_scatter(x=tpr, y=fpr, name=f'{name} (ROC/AUC {list(metrics_.values())[1]:.3f})')

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


def get_pr_analysis(groups, y, metrics, iterations=50, standardize=False, return_summary: bool = False):
    return {
        name: get_endpoint_analysis_cohort(
            name,
            y=y,
            group_id=group_id,
            metrics=metrics,
            iterations=iterations,
            standardize=standardize,
            return_summary=return_summary,
        )
        for name,
        group_id in groups.items()
    }


def run_pr_analysis_ci(groups, y, metrics, standardize, iterations=50):
    pr = get_pr_analysis(groups, y=y, metrics=metrics, standardize=standardize, iterations=iterations)
    pr_ci = get_pr_analysis_ci(pr)
    return plot_pr_ci(pr_ci)


def get_pr_analysis_ci(pr):
    return {
        cohort_name: get_curve_analysis_ci_from_bootstrap(get_first_entry(pr_bootstrapped))
        for cohort_name,
        pr_bootstrapped in pr.items()
    }


get_pr_analysis_cache = cache(get_pr_analysis_ci)


def get_curve_analysis_ci_from_bootstrap(curves, level=0.95):
    precision, recall, thresholds = transpose_list(curves)
    thresholds_ci = get_curve_ci_from_bootstrap(thresholds)
    precision_ci = get_curve_ci_from_bootstrap(precision)
    recall_ci = get_curve_ci_from_bootstrap(recall)
    return precision_ci, recall_ci, thresholds_ci


def get_curve_ci_from_bootstrap(curve):
    return [statistic_from_bootstrap(list(reject_none(points))) for points in transpose_list(curve)]


def plot_pr_ci(pr_ci_out):

    # In 10-year incident HF prediction, stacking method, combining the PCP-HF score, GB, CoxNet and Gaussian
    # mixture, achieved the best discrimination in precision-recall analysis (Figure 4), with PR/AUC 0.652 (95% CI
    # 0.618-0.682). GB scored second, with PR/AUC of 0.497 (0.468-0.527). CoxNet and PCP-HF performed similarly with
    # PR/AUC of 0.460 (0.435-0.491) and 0.459 (0.428-0.486), respectively. Precision-recall analysis is more
    # sensitive to differences in false positives and thus better captures the practical aspect of clinical decision
    # making.
    METHODS_TITLE = {
        'stacking': '<b>Stacking (ST)</b><br>PR/AUC: 0.652 (0.618-0.682)',
        'gb': "<b>Gradient Boosting (GB)</b><br>PR/AUC: 0.497 (0.468-0.527)",
        'coxnet': "<b>CoxNet (CN)</b><br>PR/AUC: 0.460 (0.435-0.491)",
        'pcp_hf': "<b>PCP-HF</b><br>PR/AUC: 0.459 (0.418-0.486)",
    }

    method_names = []
    precision_mean = []
    precision_ci_low = []
    precision_ci_high = []
    recall_mean = []

    for method_name, (precision_curve_ci, recall_curve_ci, thresholds) in pr_ci_out.items():
        n_points = len(precision_curve_ci)
        method_names += [method_name] * n_points
        recall_mean += [point['mean'] for point in recall_curve_ci]
        precision_ci_low += [point['ci'][0] for point in precision_curve_ci]
        precision_ci_high += [point['ci'][1] for point in precision_curve_ci]
        precision_mean += [point['mean'] for point in precision_curve_ci]

    pr_ci_df = DataFrame(
        {
            'method_name': method_names,
            'method_title': [METHODS_TITLE.get(m, m) for m in method_names],
            'precision_mean': precision_mean,
            'precision_ci_low': precision_ci_low,
            'precision_ci_high': precision_ci_high,
            'recall_mean': recall_mean,
        }
    ).sort_values('recall_mean')

    fig = Figure()
    # fig = px.line(
    #     pr_ci_df,
    #     x='recall_mean',
    #     y='precision_mean',
    #     color=[COLORS[m] for m in pr_ci_df['method_name']],
    # )
    setup_plotly_style(fig)

    for method_name, pr_ci_df_method in pr_ci_df.groupby('method_name'):
        fig.add_trace(
            go.Scatter(
                x=pr_ci_df_method['recall_mean'],
                y=pr_ci_df_method['precision_ci_high'],
                mode='lines',
                line=dict(color=get_color(method_name), width=0.1, shape='spline', smoothing=1.3),
                name=pr_ci_df_method['method_title'].iloc[0],
                showlegend=False,
                hovertemplate=''
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pr_ci_df_method['recall_mean'],
                y=pr_ci_df_method['precision_mean'],
                mode='lines',
                line=dict(color=get_color(method_name)),
                fill='tonexty',
                name=pr_ci_df_method['method_title'].iloc[0],
            )
        )
        fig.add_trace(
            go.Scatter(
                name=pr_ci_df_method['method_title'].iloc[0],
                x=pr_ci_df_method['recall_mean'],
                y=pr_ci_df_method['precision_ci_low'],
                mode='lines',
                line=dict(color=get_color(method_name), width=0.1, shape='spline', smoothing=1.3),
                fill='tonexty',
                showlegend=False,
                hovertemplate=''
            )
        )

    fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', width=800, height=600, hovermode="x unified")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

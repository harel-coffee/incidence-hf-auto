from pandas import DataFrame
from plotly.graph_objs import Figure

from deps.constants import RANDOM_STATE
from hcve_lib.custom_types import Prediction
from hcve_lib.evaluation_functions import average_group_scores, merge_standardize_prediction, merge_predictions, \
    compute_metrics_prediction
from hcve_lib.metrics import BootstrappedMetric
from hcve_lib.tracking import load_group_results
from hcve_lib.visualisation import setup_plotly_style


def get_endpoint_analysis_cohort(
    name, y, group_id, metrics, sample_weight=None, standardize: bool = True
) -> Prediction:
    group = load_group_results(group_id, load_models=True)
    result = average_group_scores(group)
    merge_prediction_fn = (merge_standardize_prediction if standardize else merge_predictions)
    merged_prediction = merge_prediction_fn(result)
    metrics_ = compute_metrics_prediction(
        list(
            map(
                lambda metric: BootstrappedMetric(
                    metric,
                    RANDOM_STATE,
                    iterations=10,
                    return_summary=False,
                ),
                metrics,
            )
        ),
        y,
        merged_prediction
    )
    return metrics_


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


def get_pr_analysis(groups, y, *args, **kwargs):
    return {name: get_endpoint_analysis_cohort(name, y, group_id, *args, **kwargs) for name, group_id in groups.items()}


def plot_pr_analysis(groups, y):
    fig = Figure(
        layout=dict(
            title='Precision-recall',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=800,
            height=600,
        )
    )

    setup_plotly_style(fig)
    COLORS = ['red', 'blue', 'green']
    for number, (name, metrics_) in enumerate(groups.items()):
        for tup in list(metrics_.values())[0]:
            p, r, _ = tup
            trace = fig.add_scatter(x=r, y=p, name=name, fillcolor=COLORS[number])
            print(trace.to_json())
            # precision, recall, threshold = list(metrics_.values())[0]
            # plot_data = DataFrame({
            #     'recall': recall,
            #     'precision': precision,  # 'threshold': threshold,
            # })
            # fig.add_scatter(x=recall, y=precision, name=f'{name} (PR/AUC {list(metrics_.values())[1]:.3f})')
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

    fig.update_layout(hovermode="x unified")
    fig.show()

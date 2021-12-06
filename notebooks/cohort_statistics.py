for feature in categorical_features:
    if feature == 'SOCK':
        total = len(cluster[feature].dropna())
        classes = sorted(cluster[feature].unique())
        value = '/'.join(
            [
                f'{round((len(cluster[cluster[feature] == class_value]) / total) * 100, 1)}'
                for class_value in classes
            ]
        )
        rename_features[feature] = feature
    else:
        value_count = cluster[feature].value_counts()
        if len(value_count) < 2:
            warning(f'Skipped feature {feature}')
            continue
        value = format_count_and_percentage(value_count, decimals=1)

    cluster_feature_statistics[feature] = value

for column in continuous_features:
    mean_value = float(cluster[column].mean())

    if column in non_normal_features:
        spread_statistic = f' ({round(cluster[column].quantile(0.1), 2)}' \
                           f'-{round(cluster[column].quantile(0.9), 2)})'
    else:
        spread_statistic = f' ± {round(std(cluster[column], ddof=1), 3)}'

    cluster_feature_statistics[column] = str(round_digits(mean_value, 3)) + spread_statistic

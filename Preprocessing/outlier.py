import pandas as pd


def delete_outliers_z(data: pd.DataFrame(), target:list):
    temp_df = data.copy()
    for col in temp_df:
        if col not in target:
            mean = temp_df[col].mean()
            std = temp_df[col].std()
            upper_bound = mean + 3 * std
            lower_bound = mean - 3 * std
            temp_df = temp_df[(temp_df[col] < upper_bound) & (temp_df[col] > lower_bound)]
    return temp_df


def delete_outliers_qr(data: pd.DataFrame(), target:list):
    temp_df = data.copy()
    for col in temp_df:
        if col not in target:
            q1 = temp_df[col].quantile(0.25)
            q3 = temp_df[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            temp_df = temp_df[(temp_df[col] < upper_bound) & (temp_df[col] > lower_bound)]
    return temp_df

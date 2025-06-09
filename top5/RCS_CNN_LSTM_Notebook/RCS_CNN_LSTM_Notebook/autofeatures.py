import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

def extract_tsfresh_features(df, column_id="id", column_sort="time", column_value="value"):
    # df: should contain columns [id, time, value] where id=window/sample id, time=timestamp, value=series
    extracted = extract_features(df, column_id=column_id, column_sort=column_sort)
    impute(extracted)
    return extracted

def select_relevant_features(X, y):
    selected = select_features(X, y)
    return selected

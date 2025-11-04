"""src/data_cleaning.py - funciones reutilizables para limpieza"""
import pandas as pd
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()
def drop_high_na_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    return df.loc[:, df.isnull().mean() < threshold]
def impute_numeric_with_median(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=['float64','int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    return df
def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    if 'price_usd' in df.columns and 'surface_total_in_m2' in df.columns:
        return df.dropna(subset=['price_usd','surface_total_in_m2'])
    return df
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_duplicates(df)
    df = drop_high_na_columns(df)
    df = impute_numeric_with_median(df)
    df = normalize_column_names(df)
    df = filter_invalid_rows(df)
    return df

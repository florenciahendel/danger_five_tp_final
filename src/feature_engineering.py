"""src/feature_engineering.py - funciones para generar nuevas features"""
import pandas as pd
def add_price_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    if 'price_usd' in df.columns and 'surface_total_in_m2' in df.columns:
        df['price_m2'] = df['price_usd'] / df['surface_total_in_m2']
    return df
def add_age(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    if 'year_built' in df.columns:
        df['age'] = current_year - pd.to_numeric(df['year_built'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())
    return df
def select_features(df: pd.DataFrame):
    features = []
    for c in ['surface_total_in_m2','ambientes','price_m2','age','barrio']:
        if c in df.columns:
            features.append(c)
    return features
"""src/eda.py - funciones de análisis exploratorio"""
import matplotlib.pyplot as plt
import seaborn as sns
def basic_statistics(df):
    print(df.describe(include='all'))
def plot_price_distribution(df):
    plt.figure(figsize=(8,6))
    sns.histplot(df['price_usd'], bins=30, kde=True)
    plt.title('Distribución de precios (USD)')
    plt.xlabel('Precio (USD)')
    plt.show()
def plot_price_vs_surface(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='surface_total_in_m2', y='price_usd', data=df)
    plt.title('Precio vs Superficie')
    plt.show()
def boxplot_price_by_barrio(df):
    plt.figure(figsize=(12,6))
    sns.boxplot(x='barrio', y='price_usd', data=df)
    plt.xticks(rotation=90)
    plt.title('Precio por barrio')
    plt.show()
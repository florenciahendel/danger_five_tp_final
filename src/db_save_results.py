"""src/db_save_results.py - insertar datos y métricas en SQLite"""
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime


DB_PATH = Path('../models/propiedades.db')

def setup_database():
    """
    Crea las tablas en la base de datos a partir del script SQL.
    Asegura que la carpeta /models exista.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    sql_script_path = Path(__file__).parent.parent / 'db' / 'create_tables.sql'
    
    try:
        with open(sql_script_path, 'r') as f:
            sql_script = f.read()
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo SQL en {sql_script_path}")
        print("Asegúrate de estar ejecutando desde el notebook en la carpeta /notebooks/")
        sql_script_path = Path('../db/create_tables.sql')
        with open(sql_script_path, 'r') as f:
            sql_script = f.read()

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(sql_script)
    print(f"Base de datos configurada en: {DB_PATH}")


def insert_input_data(df: pd.DataFrame, table_name: str):
    """
    Inserta el dataframe de entrada limpio en la tabla especificada.
    Usa 'replace' para que cada ejecución actualice los datos.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Datos de entrada guardados en la tabla '{table_name}'.")
    except Exception as e:
        print(f"Error al guardar datos de entrada: {e}")


def insert_metrics(metrics_df: pd.DataFrame):
    """
    Inserta el DataFrame de métricas (RMSE, MAE, R2) en la tabla.
    Usa 'append' para mantener un historial de ejecuciones.
    """
    try:
        metrics_df['fecha'] = datetime.now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            metrics_df.to_sql('metricas_modelos', conn, if_exists='append', index=False)
        print("Métricas del modelo guardadas en la base de datos.")
    except Exception as e:
        print(f"Error al guardar métricas: {e}")


def insert_model_config(config_list: list):
    """
    Inserta la configuración (parámetros) de los modelos en la tabla.
    Usa 'append' para mantener historial.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            fecha_actual = datetime.now().isoformat()
            for config in config_list:
                cur.execute(
                    'INSERT INTO config_modelos (modelo, hiperparametros, features_usadas, fecha) VALUES (?, ?, ?, ?)',
                    (config['modelo'], config['hiperparametros'], config['features_usadas'], fecha_actual)
                )
            conn.commit()
        print("Configuración de modelos guardada en la base de datos.")
    except Exception as e:
        print(f"Error al guardar configuración: {e}")

if __name__ == '__main__':
    print("Este script no se ejecuta directamente. Importa sus funciones desde un notebook.")
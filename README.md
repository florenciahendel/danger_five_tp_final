# danger_five_tp_final

Proyecto final - Programación avanzada para Ciencia de Datos.
Objetivo: construir un modelo predictivo para estimar el precio de propiedades en Argentina usando Properati.

Estructura principal del repo:
- data/raw/: datos crudos 
- data/processed/: datos procesados 
- src/: scripts reutilizables (carga, limpieza, features, modelado, bd)
- notebooks/: Jupyter notebooks de EDA y modelado
- docs/: documentación del dataset y procesos
- db/: scripts SQL
- .github/: plantillas PR, workflows

Instrucciones rápidas:
1. Descargar dataset de Properati (ver docs/dataset_download.md) y colocarlo en data/raw/properati.csv
2. Crear y activar entorno virtual e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Ejecutar notebooks en orden: 
* **`01_data_cleaning.ipynb`**: Carga y limpia el dataset crudo.
* **`02_eda.ipynb`**: Realiza el análisis exploratorio.
* **`03_feature_engineering.ipynb`**: **¡Importante!** Aquí deberás seleccionar interactivamente si quieres modelar "Venta" (USD) o "Alquiler" (ARS).
* **`04_model_comparison.ipynb`**: Entrena los modelos para el escenario elegido y guarda todos los resultados en la base de datos.
* **`05_visualizations.ipynb`**: Carga los resultados desde la base de datos y los visualiza.
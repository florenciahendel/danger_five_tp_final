import pandas as pd
from pathlib import Path
import joblib
import json
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def build_preprocessor(df):
    """
    Define el preprocesador para que incluya las nuevas features
    de ingeniería (surface_ratio, es_nuevo, lat, lon, etc.).
    """
    
    # Agregamos las nuevas features creadas en el notebook 03
    numeric_features = [
        'rooms', 'bedrooms', 'bathrooms', 'surface_total', 'surface_covered',
        'lat', 'lon', # Las variables geoespaciales son numéricas
        'surface_ratio', 'es_nuevo', 'tiene_amenities', 'es_lujo', 'total_rooms_bathrooms'
    ]
    
    categorical_features = ['l3', 'property_type', 'l2'] 
    
    # Nos aseguramos de que solo usamos las columnas que existen en el DF
    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor, numeric_features, categorical_features

def get_feature_names(preprocessor, numeric_features, categorical_features):
    """Obtiene los nombres de las features después del OneHotEncoding."""
    feature_names = numeric_features.copy()
    try:
        if categorical_features and 'cat' in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
            ohe_features = list(ohe.get_feature_names_out(categorical_features))
            feature_names.extend(ohe_features)
    except Exception as e:
        print(f"Advertencia al obtener feature names: {e}")
        pass
    return feature_names

def train_models(df, target='price', out_dir='models', test_size=0.2, random_state=42, max_rows=None):
    """
    Función de entrenamiento (sin cambios en la lógica, solo usará
    las features definidas en build_preprocessor).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if max_rows:
        df = df.head(max_rows)
    
    preprocessor, numeric_features, categorical_features = build_preprocessor(df)
    
    features = numeric_features + categorical_features
    features = [f for f in features if f in df.columns]
    
    # Asegurarnos de que el target no esté en las features
    if target in features:
        features.remove(target)
        
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # --- Linear Regression ---
    print("Entrenando Linear Regression...")
    pipe_lr = Pipeline([('pre', preprocessor), ('model', LinearRegression())])
    pipe_lr.fit(X_train, y_train)
    joblib.dump(pipe_lr, f"{out_dir}/linear_regression.joblib")
    y_pred_lr = pipe_lr.predict(X_test)
    
    metrics_lr = {
        "modelo": "linear",
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "mae": mean_absolute_error(y_test, y_pred_lr),
        "r2": r2_score(y_test, y_pred_lr)
    }
    config_lr = {
        "modelo": "linear", 
        "hiperparametros": "{}", 
        "features_usadas": json.dumps(features)
    }

    # --- Random Forest ---
    print("Entrenando Random Forest (con GridSearchCV)...")
    pipe_rf = Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(random_state=random_state))])
    param_grid = {'model__n_estimators':[50, 100], 'model__max_depth':[5,10,None]}
    grid = GridSearchCV(pipe_rf, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=1) 
    grid.fit(X_train, y_train)
    
    rf_model = grid.best_estimator_
    joblib.dump(rf_model, f"{out_dir}/random_forest.joblib")
    y_pred_rf = rf_model.predict(X_test)
    
    metrics_rf = {
        "modelo": "rf",
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "mae": mean_absolute_error(y_test, y_pred_rf),
        "r2": r2_score(y_test, y_pred_rf)
    }
    config_rf = {
        "modelo": "rf", 
        "hiperparametros": json.dumps(grid.best_params_), 
        "features_usadas": json.dumps(features)
    }

    models = {'linear': pipe_lr, 'rf': rf_model}
    
    # --- Preparar resultados para devolver ---
    
    test_ids = df.loc[X_test.index]['id']

    results_list = []
    for name, y_pred in [('linear', y_pred_lr), ('rf', y_pred_rf)]:
        results_list.append(pd.DataFrame({
            'id': test_ids.values,
            'price_real': y_test.values,
            'price_pred': y_pred,
            'neighborhood': X_test['l3'].values,
            'model': name
        }))
    model_results_df = pd.concat(results_list, ignore_index=True)
    Path('../data/processed').mkdir(parents=True, exist_ok=True)
    model_results_df.to_csv('../data/processed/model_results.csv', index=False)
    
    try:
        avg_prices = df.groupby(['l2','l3'])[target].mean().reset_index()
        avg_prices.rename(columns={'l2':'province','l3':'neighborhood', target:'avg_price'}, inplace=True)
        avg_prices.to_csv('../data/processed/avg_prices_by_location.csv', index=False)
    except Exception as e:
        print(f"No se pudo generar avg_prices_by_location.csv: {e}")

    try:
        feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
        importances = rf_model.named_steps['model'].feature_importances_
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)
        Path('../models').mkdir(parents=True, exist_ok=True)
        feature_importance.to_csv('../models/feature_importance.csv', index=False)
    except Exception as e:
        print(f"No se pudo generar feature importance: {e}")

    metrics_df = pd.DataFrame([metrics_lr, metrics_rf])
    config_list = [config_lr, config_rf]
    
    print("Entrenamiento completado.")
    
    return models, metrics_df, config_list, X_test, y_test
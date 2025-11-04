-- SQL para crear tablas
CREATE TABLE IF NOT EXISTS datos_propiedades (
    id TEXT PRIMARY KEY,
    price_usd REAL,
    currency TEXT,
    surface_total_in_m2 REAL,
    ambientes INTEGER,
    barrio TEXT,
    date_published TEXT,
    year_built INTEGER,
    price_m2 REAL
);
CREATE TABLE IF NOT EXISTS resultados_modelos (
    id TEXT PRIMARY KEY,
    modelo TEXT,
    prediccion REAL
);
CREATE TABLE IF NOT EXISTS metricas_modelos (
    modelo TEXT,
    rmse REAL,
    mae REAL,
    r2 REAL,
    fecha TEXT
);
-- NUEVA TABLA (Puntos Extra)
CREATE TABLE IF NOT EXISTS config_modelos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    modelo TEXT,
    hiperparametros TEXT,
    features_usadas TEXT,
    fecha TEXT
);

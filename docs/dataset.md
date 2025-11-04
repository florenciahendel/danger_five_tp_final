# Dataset procesado (muestra)
El archivo data/processed/properati_clean.csv contiene una muestra procesada.
Columnas principales:
- id,lat, lon, l1 (Country), l2 (Province), l3(Neighborhood), rooms, bedrooms, bathrooms, surface_total, surface_covered, currency,title, description, property_type (Casa, departamento), operation_type (Alquiler o venta), price
Limpieza aplicada:
- Eliminados duplicados exactos
- Columnas con >50% NA eliminadas
- Numéricos imputados por mediana

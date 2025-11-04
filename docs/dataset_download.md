# Instrucciones para descargar Properati (Kaggle)
1. Crear cuenta en Kaggle y generar API token (kaggle.json): 
- En kaggle.com ir a la foto de perfil -> Settings -> Account -> API -> Create new token
- Se descarga el archivo kaggle.json
2. Configurar el Archivo de Credenciales:  
- Crear una carpeta llamada .kaggle en el directorio raíz del usuario.
- Mover el archivo kaggle.json a esta carpeta.
- **Importante (Linux/macOS)**: Asegurar permisos de acceso restringido: chmod 600 ~/.kaggle/kaggle.json.
- Ubicación final esperada:
   - Linux/macOS: ~/.kaggle/kaggle.json
   - Windows: C:\Users\<tu-usuario>\.kaggle\kaggle.json
3. Descargar, descomprimir y renombrar el dataset:
   ```Bash
      kaggle datasets download -d alejandroczernikier/properati-argentina-dataset -p data/raw --unzip
      move data\raw\entrenamiento.csv data\raw\properati.csv

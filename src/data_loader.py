import pandas as pd
import numpy as np

def cargar_y_limpiar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga el dataset Telco Customer Churn y realiza la limpieza inicial de tipos de datos.
    """
    df = pd.read_csv(ruta_archivo)
    
    # Corregir la columna TotalCharges que contiene espacios vacíos en texto
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
        # Imputar valores nulos iniciales de TotalCharges con la mediana o 0 para clientes con tenure=0
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
        
    return df
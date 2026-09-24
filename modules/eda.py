import pandas as pd

def perfil_calidad_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un perfil de calidad por variable en el DataFrame.
    """
    n_filas = len(df)
    perfil = pd.DataFrame({
        'Tipo de Dato': df.dtypes.astype(str),
        'Nulos': df.isnull().sum(),
        '% Nulos': (df.isnull().sum() / n_filas * 100).round(2),
        'Valores Únicos': df.nunique()
    })
    perfil.attrs['duplicados'] = int(df.duplicated().sum())
    return perfil

def resumir_variable(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Entrega un resumen específico según si la variable es numérica o categórica.
    """
    if columna not in df.columns:
        raise KeyError(f"La columna '{columna}' no existe.")
        
    serie = df[columna]
    if pd.api.types.is_numeric_dtype(serie):
        resumen = serie.describe().to_frame().T
    else:
        abs_count = serie.value_counts(dropna=False)
        rel_count = (serie.value_counts(normalize=True, dropna=False) * 100).round(2)
        resumen = pd.DataFrame({'Frecuencia': abs_count, 'Porcentaje (%)': rel_count})
    return resumen

def analizar_churn_por_grupo(df: pd.DataFrame, grupo_col: str) -> pd.DataFrame:
    """
    Calcula la tasa de Churn segmentada por una variable categórica.
    """
    if grupo_col not in df.columns or 'Churn' not in df.columns:
        raise KeyError("Columnas especificadas no válidas.")
        
    tabla = pd.crosstab(df[grupo_col], df['Churn'], normalize='index') * 100
    tabla = tabla.round(2)
    return tabla
import plotly.express as px
import pandas as pd

def grafico_distribucion(df: pd.DataFrame, columna: str, color_col: str = "Churn"):
    """
    Genera un histograma o gráfico de barras según el tipo de variable.
    """
    if pd.api.types.is_numeric_dtype(df[columna]):
        fig = px.histogram(
            df, x=columna, color=color_col, barmode="overlay",
            title=f"Distribución de {columna} por {color_col}",
            template="plotly_white"
        )
    else:
        fig = px.histogram(
            df, x=columna, color=color_col, barmode="group",
            title=f"Conteo de {columna} por {color_col}",
            template="plotly_white"
        )
    return fig

def grafico_caja(df: pd.DataFrame, col_num: str, col_cat: str = "Churn"):
    """
    Genera un diagrama de caja comparativo.
    """
    fig = px.box(
        df, x=col_cat, y=col_num, color=col_cat,
        title=f"Diagrama de Caja: {col_num} según {col_cat}",
        template="plotly_white"
    )
    return fig

def grafico_dispersion(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = "Churn"):
    """
    Genera un gráfico de dispersión entre dos variables continuas.
    """
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, opacity=0.7,
        title=f"Dispersión: {x_col} vs {y_col}",
        template="plotly_white"
    )
    return fig
import streamlit as st
import pandas as pd
import os

from src.data_loader import cargar_y_limpiar_datos
from src.analytics import (
    missing_report, 
    numeric_summary, 
    target_correlations, 
    train_regression_demo, 
    build_agent_context
)
from src.visualizations import grafico_distribucion, grafico_caja, grafico_dispersion
from src.agent import AgenteDatos

st.set_page_config(page_title="Telco Churn Analytics & AI Agent", layout="wide")

st.title("📊 Telco Churn Explorer & Agente de IA Local")
st.caption("Aplicación Modular en Streamlit con Analytics y Ollama (Llama 3.2)")

# Carga de datos
RUTA_DEFECTO = "data/Telco-Customer-Churn.csv"

@st.cache_data
def obtener_datos(ruta):
    return cargar_y_limpiar_datos(ruta)

if os.path.exists(RUTA_DEFECTO):
    df = obtener_datos(RUTA_DEFECTO)
else:
    archivo_subido = st.sidebar.file_uploader("Cargar dataset CSV", type=["csv"])
    if archivo_subido:
        df = cargar_y_limpiar_datos(archivo_subido)
    else:
        st.warning("Por favor, suba el archivo del dataset o colóquelo en `data/`.")
        st.stop()

# Navegación por pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Calidad y Reportes", 
    "📈 Correlaciones y Analytics", 
    "📊 Visualización Interactiva", 
    "🤖 Agente de IA Local"
])

# Pestaña 1: Calidad (Usando analytics)
with tab1:
    st.header("Reporte de Calidad y Resumen Estadístico")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Filas", df.shape[0])
    col2.metric("Total Columnas", df.shape[1])
    col3.metric("Filas Duplicadas", int(df.duplicated().sum()))
    
    st.subheader("Reporte de Valores Nulos")
    reporte_nulos = missing_report(df)
    st.dataframe(reporte_nulos, use_container_width=True)
    
    st.subheader("Resumen Estadístico Numérico")
    resumen_num = numeric_summary(df)
    if not resumen_num.empty:
        st.dataframe(resumen_num, use_container_width=True)
    else:
        st.info("No se encontraron variables numéricas para resumir.")

# Pestaña 2: Correlaciones y Analytics
with tab2:
    st.header("Análisis de Correlaciones y Modelado Predictivo")
    numeric_cols = list(df.select_dtypes(include="number").columns)
    
    if numeric_cols:
        target_default = "TotalCharges" if "TotalCharges" in numeric_cols else numeric_cols[0]
        target_col = st.selectbox("Seleccione la variable objetivo (Target):", options=numeric_cols, index=numeric_cols.index(target_default) if target_default in numeric_cols else 0)
        
        st.subheader(f"Correlaciones numéricas respecto a `{target_col}`")
        df_corrs = target_correlations(df, target_col)
        if not df_corrs.empty:
            st.dataframe(df_corrs, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para calcular correlaciones.")
            
        st.subheader("Demostración de Regresión Lineal")
        try:
            model_results = train_regression_demo(df, target_col)
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{model_results['metrics']['r2']:.3f}")
            m2.metric("MAE", f"{model_results['metrics']['mae']:.3f}")
            m3.metric("RMSE", f"{model_results['metrics']['rmse']:.3f}")
            
            with st.expander("Ver Coeficientes del Modelo"):
                st.dataframe(model_results["coefficients"], use_container_width=True)
        except Exception as e:
            st.info(f"No se pudo entrenar la regresión de demostración: {e}")
    else:
        st.warning("El dataset no cuenta con suficientes columnas numéricas.")

# Pestaña 3: Visualización
with tab3:
    st.header("Visualizaciones Interactivas (Plotly)")
    tipo_grafico = st.selectbox("Tipo de Gráfico", ["Distribución/Conteo", "Boxplot", "Dispersión"])
    
    if tipo_grafico == "Distribución/Conteo":
        var_x = st.selectbox("Variable X:", df.columns, index=df.columns.get_loc("tenure") if "tenure" in df.columns else 0)
        fig = grafico_distribucion(df, var_x)
        st.plotly_chart(fig, use_container_width=True)
    elif tipo_grafico == "Boxplot":
        num_opts = [c for c in ["MonthlyCharges", "TotalCharges", "tenure"] if c in df.columns]
        var_num = st.selectbox("Variable Numérica Y:", num_opts if num_opts else df.select_dtypes(include='number').columns)
        fig = grafico_caja(df, var_num)
        st.plotly_chart(fig, use_container_width=True)
    elif tipo_grafico == "Dispersión":
        fig = grafico_dispersion(df, "tenure", "MonthlyCharges") if "tenure" in df.columns and "MonthlyCharges" in df.columns else grafico_dispersion(df, df.columns[0], df.columns[1])
        st.plotly_chart(fig, use_container_width=True)

# Pestaña 4: Agente de IA
with tab4:
    st.header("Consulta Inteligente con Agente Local (Ollama)")
    agente = AgenteDatos(modelo="llama3.2:3b")
    
    if not agente.cliente.verificar_conexion():
        st.error("❌ Ollama no responde en `localhost:11434`. Asegúrese de ejecutar `ollama serve` y tener el modelo listo.")
    else:
        st.success("✅ Conectado a Ollama (llama3.2:3b)")
        pregunta = st.text_input("Realice una consulta sobre los datos:", "¿Cuáles son las conclusiones principales basadas en la analítica de este dataset?")
        
        if st.button("Consultar Agente"):
            with st.spinner("Procesando respuesta con IA local..."):
                # Generamos contexto estructurado usando analytics
                num_cols_list = list(df.select_dtypes(include="number").columns)
                t_col = num_cols_list[0] if num_cols_list else "MonthlyCharges"
                try:
                    m_res = train_regression_demo(df, t_col)
                except Exception:
                    m_res = None
                
                contexto_json = build_agent_context(df, t_col, m_res)
                contexto_extra = f"Contexto analítico estructurado del dataset:\n{contexto_json}"
                
                respuesta = agente.responder_consulta(pregunta, df, contexto_adicional=contexto_extra)
                st.markdown("### Respuesta del Agente")
                st.write(respuesta)
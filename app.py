import streamlit as st
import pandas as pd
from src.data_loader import cargar_y_limpiar_datos
from src.analytics import missing_report, numeric_summary, target_correlations, train_regression_demo, build_agent_context, analizar_churn_por_grupo
from src.visualizations import grafico_distribucion, grafico_caja, grafico_dispersion
from src.agent import AgenteDatos

# Configuración inicial de la página
st.set_page_config(
    page_title="Telco Churn - Aplicación Modular",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sistema Analítico de Retención de Clientes (Telco Churn)")
st.caption("Aplicación Modular con Python, Streamlit, Analytics y Agente LLM Local (Ollama)")

# Carga de datos optimizada con caché
@st.cache_data
def load_data():
    return cargar_y_limpiar_datos("data/Telco-Customer-Churn.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error al cargar el dataset: {e}")
    st.stop()

# Pestañas principales de navegación
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Calidad y Reportes", 
    "📈 Correlaciones y Analytics", 
    "📊 Visualización Interactiva", 
    "🤖 Agente de IA Local"
])

# ==========================================
# PESTAÑA 1: Calidad y Reportes
# ==========================================
with tab1:
    st.header("Reporte de Calidad y Resumen Estadístico")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Filas", df.shape[0])
    col2.metric("Total Columnas", df.shape[1])
    col3.metric("Filas Duplicadas", int(df.duplicated().sum()))
    
    st.subheader("Auditoría de Datos y Tipos")
    df_missing = missing_report(df)
    st.dataframe(df_missing, width='stretch')
    
    st.subheader("Resumen Estadístico Numérico")
    df_num_summary = numeric_summary(df)
    if not df_num_summary.empty:
        st.dataframe(df_num_summary, width='stretch')
    else:
        st.info("No hay variables numéricas disponibles para resumir.")

# ==========================================
# PESTAÑA 2: Correlaciones y Analytics
# ==========================================
with tab2:
    st.header("Análisis de Correlaciones, Segmentos y Modelado")
    
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category", "str"]).columns)
    
    sub_tab_a, sub_tab_b = st.tabs(["📉 Correlaciones y Regresión", "🏢 Análisis de Churn por Segmentos"])
    
    with sub_tab_a:
        if numeric_cols:
            target_default = "TotalCharges" if "TotalCharges" in numeric_cols else numeric_cols[0]
            target_col = st.selectbox("Seleccione la variable objetivo (Target numérico):", options=numeric_cols, index=numeric_cols.index(target_default) if target_default in numeric_cols else 0)
            
            st.subheader(f"Correlaciones numéricas respecto a `{target_col}`")
            df_corrs = target_correlations(df, target_col)
            if not df_corrs.empty:
                st.dataframe(df_corrs, width='stretch')
            else:
                st.warning("No hay suficientes datos para calcular correlaciones.")
                
            st.subheader("Demostración de Regresión Lineal")
            try:
                model_results = train_regression_demo(df, target_col)
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("R² Score", f"{model_results['metrics']['r2']:.3f}")
                m2.metric("MAE", f"{model_results['metrics']['mae']:.3f}")
                m3.metric("RMSE", f"{model_results['metrics']['rmse']:.3f}")
                m4.metric("Train Size", model_results['n_train'])
                m5.metric("Test Size", model_results['n_test'])
                
                mod_sub1, mod_sub2, mod_sub3 = st.tabs(["📊 Coeficientes", "📈 Reales vs Predichos", "⚙️ Features"])
                
                with mod_sub1:
                    st.dataframe(model_results["coefficients"], width='stretch')
                with mod_sub2:
                    st.dataframe(model_results["predictions"], width='stretch')
                    import plotly.express as px
                    fig_pred = px.scatter(
                        model_results["predictions"], x="real", y="predicho",
                        title="Valores Reales vs Predichos (Test)", template="plotly_white"
                    )
                    st.plotly_chart(fig_pred, width='stretch')
                with mod_sub3:
                    st.write(model_results["features"])
            except Exception as e:
                st.info(f"No se pudo entrenar la regresión: {e}")
        else:
            st.warning("El dataset no cuenta con suficientes columnas numéricas.")
            
    with sub_tab_b:
        st.subheader("Tasa de Cancelación (Churn) por Segmento de Negocio")
        if categorical_cols and "Churn" in df.columns:
            cat_default = "Contract" if "Contract" in categorical_cols else categorical_cols[0]
            cat_col = st.selectbox("Seleccione variable categórica clave:", options=categorical_cols, index=categorical_cols.index(cat_default))
            
            df_churn_seg = analizar_churn_por_grupo(df, cat_col)
            if not df_churn_seg.empty:
                st.dataframe(df_churn_seg, width='stretch')
                
                # Gráfico de barras apiladas para visualizar el Churn por segmento
                import plotly.express as px
                fig_seg = px.histogram(
                    df, x=cat_col, color="Churn", barmode="stack", barnorm="percent",
                    title=f"Distribución porcentual de Churn según {cat_col}",
                    template="plotly_white"
                )
                st.plotly_chart(fig_seg, width='stretch')
            else:
                st.warning("No se pudo calcular la segmentación.")
        else:
            st.warning("No hay variables categóricas o falta la columna Churn.")

# ==========================================
# PESTAÑA 3: Visualización Interactiva
# ==========================================
with tab3:
    st.header("Explorador Gráfico Interactivo")
    
    all_cols = list(df.columns)
    col_x = st.selectbox("Seleccione la variable a graficar (X):", options=all_cols, index=all_cols.index("MonthlyCharges") if "MonthlyCharges" in all_cols else 0)
    
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Distribución / Frecuencias")
        fig_dist = grafico_distribucion(df, col_x, color_col="Churn" if "Churn" in df.columns else all_cols[0])
        st.plotly_chart(fig_dist, width='stretch')
        
    with v2:
        st.subheader("Diagrama de Caja (Boxplot)")
        numeric_box_cols = list(df.select_dtypes(include="number").columns)
        if numeric_box_cols:
            col_box = st.selectbox("Variable numérica para caja:", options=numeric_box_cols, index=0)
            fig_box = grafico_caja(df, col_box, col_cat="Churn" if "Churn" in df.columns else numeric_box_cols[0])
            st.plotly_chart(fig_box, width='stretch')
        else:
            st.info("No hay variables numéricas para boxplot.")

# ==========================================
# PESTAÑA 4: Agente de IA Local
# ==========================================
with tab4:
    st.header("Asistente Inteligente de Retención (Ollama / Llama 3.2)")
    st.markdown("Consulta en lenguaje natural sobre los datos y recibe recomendaciones analíticas contextualizadas.")
    
    agente = AgenteDatos(modelo="llama3.2:3b")
    
    # Preguntas de ejemplo predefinidas para facilitar la interacción
    prompt_predefinido = st.selectbox(
        "Seleccione una consulta sugerida o escriba abajo:",
        options=[
            "-- Escriba su propia consulta o elija una opción --",
            "¿Cuáles son los principales factores de riesgo que impulsan el Churn según los contratos?",
            "Haz un resumen ejecutivo de las métricas clave de cobros y permanencia.",
            "¿Qué recomendaciones comerciales darías para reducir la cancelación en clientes con contratos mensuales?"
        ]
    )
    
    pregunta_usuario = st.text_area("Su pregunta al Agente de Datos:", value="" if prompt_predefinido.startswith("--") else prompt_predefinido)
    
    if st.button("Consultar al Agente", type="primary"):
        if pregunta_usuario.strip():
            with st.spinner("Analizando contexto y generando respuesta con Ollama..."):
                respuesta = agente.responder_consulta(pregunta_usuario, df)
                st.markdown("### 🤖 Respuesta del Consultor IA:")
                st.write(respuesta)
        else:
            st.warning("Por favor, ingrese o seleccione una pregunta válida.")

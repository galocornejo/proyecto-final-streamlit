import streamlit as st
import pandas as pd
import os

from src.data_loader import cargar_y_limpiar_datos
from src.eda import perfil_calidad_datos, resumir_variable, analizar_churn_por_grupo
from src.visualizations import grafico_distribucion, grafico_caja, grafico_dispersion
from src.agent import AgenteDatos

st.set_page_config(page_title="Telco Churn Analytics & AI Agent", layout="wide")

st.title("📊 Telco Churn Explorer & Agente de IA Local")
st.caption("Aplicación Modular en Streamlit con Integración Ollama (Llama 3.2)")

# Carga de datos
RUTA_DEFECTO = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

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
    "📋 Calidad y Perfilado", 
    "📈 Exploración & Resumen", 
    "📊 Visualización Interactiva", 
    "🤖 Agente de IA Local"
])

# Pestaña 1: Calidad
with tab1:
    st.header("Perfil de Calidad del Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Total Filas", df.shape[0])
    col2.metric("Total Columnas", df.shape[1])
    
    perfil = perfil_calidad_datos(df)
    st.dataframe(perfil, use_container_width=True)
    st.info(f"Filas duplicadas identificadas: {perfil.attrs.get('duplicados', 0)}")

# Pestaña 2: Exploración
with tab2:
    st.header("Resumen de Variables y Segmentación de Churn")
    col_sel = st.selectbox("Seleccione variable a resumir:", df.columns)
    resumen = resumir_variable(df, col_sel)
    st.dataframe(resumen)
    
    st.subheader("Análisis de Churn por Creador de Segmentos Categóricos")
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c not in ['customerID', 'Churn']]
    grupo_sel = st.selectbox("Seleccione segmento para cruzar con Churn:", cat_cols)
    if grupo_sel:
        tabla_churn = analizar_churn_por_grupo(df, grupo_sel)
        st.write(f"Porcentaje de Churn según **{grupo_sel}**:")
        st.dataframe(tabla_churn)

# Pestaña 3: Visualización
with tab3:
    st.header("Visualizaciones Interactivas (Plotly)")
    tipo_grafico = st.selectbox("Tipo de Gráfico", ["Distribución/Conteo", "Boxplot", "Dispersión"])
    
    if tipo_grafico == "Distribución/Conteo":
        var_x = st.selectbox("Variable X:", df.columns, index=df.columns.get_loc("tenure"))
        fig = grafico_distribucion(df, var_x)
        st.plotly_chart(fig, use_container_width=True)
    elif tipo_grafico == "Boxplot":
        var_num = st.selectbox("Variable Numérica Y:", ["MonthlyCharges", "TotalCharges", "tenure"])
        fig = grafico_caja(df, var_num)
        st.plotly_chart(fig, use_container_width=True)
    elif tipo_grafico == "Dispersión":
        fig = grafico_dispersion(df, "tenure", "MonthlyCharges")
        st.plotly_chart(fig, use_container_width=True)

# Pestaña 4: Agente de IA
with tab4:
    st.header("Consulta Inteligente con Agente Local (Ollama)")
    agente = AgenteDatos(modelo="llama3.2:3b")
    
    if not agente.cliente.verificar_conexion():
        st.error("❌ Ollama no responde en `localhost:11434`. Asegúrese de ejecutar `ollama serve` y tener el modelo listo.")
    else:
        st.success("✅ Conectado a Ollama (llama3.2:3b)")
        pregunta = st.text_input("Realice una consulta sobre los datos:", "¿Cuáles son las variables con mayor influencia en la cancelación de clientes?")
        
        if st.button("Consultar Agente"):
            with st.spinner("Procesando respuesta con IA local..."):
                # Se envía información calculada como contexto
                resumen_contratos = analizar_churn_por_grupo(df, "Contract").to_string()
                contexto_extra = f"Tabla de Churn por Tipo de Contrato:\n{resumen_contratos}"
                
                respuesta = agente.responder_consulta(pregunta, df, contexto_adicional=contexto_extra)
                st.markdown("### Respuesta del Agente")
                st.write(respuesta)
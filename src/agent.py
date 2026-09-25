import pandas as pd
from .llm_client import ClienteOllama

class AgenteDatos:
    """
    Agente inteligente que toma contexto analítico precalculado de Python
    y genera interpretaciones justificadas con Ollama.
    """
    def __init__(self, modelo: str = "llama3.2:3b"):
        self.cliente = ClienteOllama(modelo=modelo)

    def construir_contexto_sintetico(self, df: pd.DataFrame) -> str:
        total_filas, total_cols = df.shape
        tasa_churn = (df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100) if 'Churn' in df.columns else 0
        promedio_mensual = df['MonthlyCharges'].mean() if 'MonthlyCharges' in df.columns else 0
        mediana_tenure = df['tenure'].median() if 'tenure' in df.columns else 0
        
        contexto = f"""
        [CONTEXTO DEL DATASET - TELCO CUSTOMER CHURN]
        - Registros totales: {total_filas}
        - Total de columnas: {total_cols}
        - Tasa general de cancelación (Churn): {tasa_churn:.2f}%
        - Promedio de cobro mensual (MonthlyCharges): ${promedio_mensual:.2f}
        - Mediana de permanencia en meses (tenure): {mediana_tenure} meses
        - Principales variables: Contract, PaymentMethod, InternetService, tenure, MonthlyCharges, TotalCharges, Churn.
        """
        return contexto

    def responder_consulta(self, pregunta: str, df: pd.DataFrame, contexto_adicional: str = "") -> str:
        contexto_base = self.construir_contexto_sintetico(df)
        
        system_instruction = (
            "Eres un experto consultor de ciencia de datos y retención de clientes. "
            "Responde a las preguntas basándote ESTRICTAMENTE en las métricas y cálculos previstos en el contexto. "
            "Sé conciso, profesional y directo en tus recomendaciones técnicas."
        )
        
        prompt_completo = f"""
        {contexto_base}
        
        [INFORMACIÓN ADICIONAL CALCULADA]
        {contexto_adicional}
        
        [PREGUNTA DEL USUARIO]
        {pregunta}
        """
        
        return self.cliente.generar_respuesta(prompt_completo, system_prompt=system_instruction)
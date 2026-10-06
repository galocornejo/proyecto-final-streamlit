# 📊 Sistema Analítico de Retención de Clientes (Telco Churn)

Proyecto final desarrollado para el módulo **Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos**.

## Caso de demostración

Se utiliza el dataset **Telco Customer Churn**, orientado al análisis predictivo y la retención de clientes en empresas de telecomunicaciones. El archivo se encuentra ubicado dentro de `data/Telco-Customer-Churn.csv`, permitiendo ejecutar la aplicación de forma completamente local.

> Este proyecto es de carácter estrictamente académico. Está diseñado para ilustrar flujos analíticos, ciencia de datos modular, modelado predictivo y la integración de asistentes conversacionales con Inteligencia Artificial local.

## Qué demuestra

- Carga y limpieza automatizada de un dataset tabular real.
- Auditoría de calidad de datos (valores nulos, tipos de variables y duplicados).
- Análisis de correlaciones numéricas y demostración de modelos de regresión lineal.
- Análisis segmentado de la tasa de cancelación (*Churn*) por variables categóricas clave (como contratos y servicios).
- Visualizaciones interactivas y dinámicas impulsadas por Plotly.
- Integración de un agente inteligente local mediante un cliente HTTP dedicado y **Ollama (Llama 3.2:3b)**.
- Contexto analítico resumido e inyectado al LLM para optimizar las respuestas del consultor de IA.
- Organización del código bajo una estricta arquitectura modular reutilizable.

## Estructura

proyecto-final-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── run_app.bat
├── data/
│   └── Telco-Customer-Churn.csv
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── analytics.py
    ├── visualizations.py
    ├── llm_client.py
    └── agent.py

## 1. Crear entorno virtual

### Windows

python -m venv .venv
.venv\Scripts\activate

### macOS / Linux

python3 -m venv .venv
source .venv/bin/activate


## 2. Instalar dependencias

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


## 3. Ollama

Instalar Ollama desde su sitio oficial y, con el servicio activo, descargar un modelo ligero:

ollama pull llama3.2:3b

En equipos con recursos limitados puede usarse:

ollama pull llama3.2:1b


## 4. Ejecutar

.\run_app.bat

La aplicación cuenta con mecanismos de control de errores y se ejecutará de forma fluida. Si el servidor local de Ollama no se encuentra activo, el agente notificará el estado de la conexión de manera controlada.


## GitHub

El proyecto está estructurado y documentado conforme a los estándares de reproducibilidad y buenas prácticas para la presentación de trabajos finales.

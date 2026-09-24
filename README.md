# Demo de Proyecto Final · Streamlit + Ollama + Scikit-learn

Ejemplo básico para clase del módulo **Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos**.

## Caso de demostración

Se utiliza el dataset de regresión **Diabetes** incluido en Scikit-learn. El CSV se genera desde `sklearn.datasets.load_diabetes(scaled=False)` y se entrega dentro de `data/diabetes_sklearn.csv`, por lo que el demo puede ejecutarse sin descargar el dataset desde Internet.

Documentación oficial del dataset:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

> Este demo es exclusivamente académico. El dataset se utiliza para ilustrar análisis de datos y regresión; no debe interpretarse como una herramienta clínica o diagnóstica.

## Qué demuestra

- Carga de un CSV incluido o de un CSV propio.
- Perfil general y calidad de datos.
- Exploración de variables y correlaciones.
- Visualizaciones interactivas con Plotly.
- Modelo básico de regresión lineal.
- Funciones reutilizables organizadas por módulos.
- Clase `LocalDataAgent` para conectar con Ollama.
- Contexto analítico resumido para evitar enviar todo el dataset al LLM.
- Explicación local de respaldo cuando Ollama no está disponible.

## Estructura

```text
Demo_Proyecto_Final_Streamlit_Ollama_ScikitLearn/
├── app.py
├── create_demo_data.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── START_HERE.txt
├── run_app.bat
├── run_app.command
├── data/
│   └── diabetes_sklearn.csv
└── modules/
    ├── __init__.py
    ├── data.py
    ├── analytics.py
    ├── visualizations.py
    └── agent.py
```

## 1. Crear entorno virtual

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Ollama

Instalar Ollama desde su sitio oficial y, con el servicio activo, descargar un modelo ligero:

```bash
ollama pull llama3.2:3b
```

En equipos con recursos limitados puede usarse:

```bash
ollama pull llama3.2:1b
```

## 4. Ejecutar

```bash
python -m streamlit run app.py
```

La aplicación funciona en modo analítico aun si Ollama no está disponible. El agente mostrará una explicación local de respaldo.

## GitHub

El proyecto está preparado para ser subido a GitHub. Para el proyecto final, el `README.md` debe documentar claramente instalación, dependencias, modelo de Ollama y ejecución local.

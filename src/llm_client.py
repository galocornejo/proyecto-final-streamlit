import requests
import json

class ClienteOllama:
    """
    Cliente para la API local de Ollama con manejo de fallos y timeouts.
    """
    def __init__(self, modelo: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.modelo = modelo
        self.base_url = base_url

    def verificar_conexion(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return res.status_code == 200
        except Exception:
            return False

    def generar_respuesta(self, prompt: str, system_prompt: str = "") -> str:
        if not self.verificar_conexion():
            return "Error: No se pudo conectar con el servidor local de Ollama. Verifique que esté iniciado (`ollama serve`)."

        payload = {
            "model": self.modelo,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "Sin respuesta.")
            else:
                return f"Error en el servidor Ollama: Código {response.status_code}"
        except Exception as e:
            return f"Excepción al llamar a Ollama: {str(e)}"
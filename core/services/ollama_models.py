
import time
import requests

class OllamaModels:

    def __init__(self, url):

        self.base_url = url.rstrip("/")


    def check_health(self, max_try=3, try_wait=10) -> bool:
        """Check if the Ollama server is reachable."""

        for i in range(0, max_try):

            try:
                res = requests.get(f"{self.base_url}", timeout=5)
                if res.status_code == 200:
                    return True
            except Exception as e:
                print(f"try ({i+1}/{max_try}): Ollama server health check failed: {e}")

            time.sleep(try_wait)

        return False


    def list_models(self) -> list:
        """Returns a list of all available model names."""

        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=5)
            res.raise_for_status()
            result = res.json()
            return [
                m["name"] for m in result.get("models", [])
            ]
        except Exception as e:
            print(f"[Model List Error] {e}")
            return []


    def get_model_details(self, model_name: str) -> dict:
        """Returns details about a specific model, if available."""

        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=5)
            res.raise_for_status()
            result = res.json()
            for model in result.get("models", []):
                if model["name"] == model_name:
                    return model
            raise ValueError(f"Model '{model_name}' not found.")
        except Exception as e:
            print(f"[Model Detail Error] {e}")
            return {}


    def is_available(self, model_name: str) -> bool:
        """Checks if a model is available on the Ollama server."""

        return model_name in self.list_models()

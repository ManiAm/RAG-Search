
# if you are running this server inside WSL2 then
# netsh interface portproxy add v4tov4 listenport=5005 listenaddress=0.0.0.0 connectport=5005 connectaddress=127.0.0.1

import requests
import time
from typing import List, Dict
from langchain.embeddings.base import Embeddings


class RemoteEmbedding(Embeddings):

    def __init__(self, endpoint: str, model = None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    ####################

    def check_health(self, max_try=3, try_wait=30):

        for i in range(0, max_try):

            try:
                response = requests.get(f"{self.endpoint}/health", timeout=10)
                if response.status_code == 200:
                    return True
            except Exception as e:
                print(f"try ({i+1}/{max_try}): Embedding server health check failed: {e}")

            time.sleep(try_wait)

        return False

    ####################

    def load_model(self, model_list):
        """Load one or more embedding models."""

        try:
            res = requests.post(f"{self.endpoint}/load-model", params={"models": model_list}, timeout=5*60)
            res.raise_for_status()
            return True, res.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = res.json().get("detail", str(http_err))
            except Exception:
                error_detail = str(http_err)
            return False, f"Failed to load model(s): {error_detail}"
        except Exception as e:
            return False, f"Failed to load model(s): {str(e)}"


    def unload_model(self, model_name):
        """Unload an embedding model."""

        try:
            res = requests.delete(f"{self.endpoint}/unload-model/{model_name}", timeout=1*60)
            res.raise_for_status()
            return True, res.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = res.json().get("detail", str(http_err))
            except Exception:
                error_detail = str(http_err)
            return False, f"Failed to unload model(s): {error_detail}"
        except Exception as e:
            return False, f"Failed to unload model(s): {str(e)}"


    def unload_all_models(self):

        try:
            res = requests.delete(f"{self.endpoint}/unload-all-models", timeout=3*60)
            res.raise_for_status()
            return True, res.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = res.json().get("detail", str(http_err))
            except Exception:
                error_detail = str(http_err)
            return False, f"Failed to unload all models: {error_detail}"
        except Exception as e:
            return False, f"Failed to unload all models: {str(e)}"

    ####################

    def list_models(self) -> List[str]:
        """Fetch list of available models from the remote endpoint."""

        try:
            res = requests.get(f"{self.endpoint}/models", timeout=10)
            res.raise_for_status()
            return True, res.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = res.json().get("detail", str(http_err))
            except Exception:
                error_detail = str(http_err)
            return False, f"Failed to fetch model list: {error_detail}"
        except Exception as e:
            return False, f"Failed to fetch model list: {str(e)}"


    def get_vector_sizes(self) -> Dict[str, int]:
        """Fetch vector sizes for all models from the remote endpoint."""

        try:
            res = requests.get(f"{self.endpoint}/vector-sizes", timeout=10)
            res.raise_for_status()
            return True, res.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = res.json().get("detail", str(http_err))
            except Exception:
                error_detail = str(http_err)
            return False, f"Failed to fetch vector size: {error_detail}"
        except Exception as e:
            return False, f"Failed to fetch vector size: {str(e)}"


    def get_max_tokens(self) -> Dict[str, int]:

        try:
            res = requests.get(f"{self.endpoint}/max-tokens", timeout=10)
            res.raise_for_status()
            return True, res.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = res.json().get("detail", str(http_err))
            except Exception:
                error_detail = str(http_err)
            return False, f"Failed to fetch max-tokens: {error_detail}"
        except Exception as e:
            return False, f"Failed to fetch max-tokens: {str(e)}"

    ####################

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of input texts."""

        status, output = self._request_embeddings(texts)
        if not status:
            raise ValueError(output)
        embeddings = output.get("embeddings", [])
        if not embeddings or not isinstance(embeddings, list):
            raise ValueError(f"No valid embedding list returned for model '{self.model}'")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single input text."""

        embeddings = self.embed_documents([text])
        return embeddings[0]

    def _request_embeddings(self, texts: List[str]) -> dict:

        try:
            res = requests.post(self.endpoint, json={"texts": texts, "model": self.model}, timeout=5*60)
            res.raise_for_status()
            return True, res.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = res.json().get("detail", str(http_err))
            except Exception:
                error_detail = str(http_err)
            return False, f"Embedding request failed for model '{self.model}': {error_detail}"
        except Exception as e:
            return False, f"Embedding request failed for model '{self.model}': {e}"

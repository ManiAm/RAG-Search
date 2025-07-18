
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()


class LiteLLM:

    def __init__(self, url):

        self.base_url = url.rstrip("/")

        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)
        if not OPENAI_API_KEY:
            print("OPENAI_API_KEY environment variable is not set")
            sys.exit(1)

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }


    def is_reachable(self, timeout=3):

        try:
            response = requests.get(self.base_url, timeout=timeout)
            if response.status_code == 200:
                return True
            print(f"LiteLLM is not reachable: Unexpected status code: {response.status_code}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"LiteLLM is not reachable: {str(e)}")
            return False


    def list_models(self) -> list:
        """ Returns a list of all available model names. """

        try:
            res = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=10)
            res.raise_for_status()
            resp = res.json()
            data = resp.get("data", {})
            model_ids = [item["id"] for item in data]
            return model_ids
        except Exception as e:
            print(f"[Model List Error] {e}")
            return []


    def is_available(self, model_name: str) -> bool:
        """ Checks if a model is available on litellm. """

        return model_name in self.list_models()


    def get_model_info(self, model_name: str):
        """ Returns model info. """

        try:

            res = requests.get(f"{self.base_url}/model/info", headers=self.headers, timeout=10)
            res.raise_for_status()
            result = res.json()
            data = result.get("data", [])

            model_details = next((item for item in data if item["model_name"] == model_name), None)
            if not model_details:
                return {}

            litellm_params = model_details.get("litellm_params", {})
            model_info = model_details.get("model_info", {})

            if model_name.startswith("ollama/"):
                return self.get_ollama_model_info(model_name, litellm_params)

            return model_info

        except Exception as e:
            print(f"[Model Detail Error] {e}")
            return {}


    def get_ollama_model_info(self, model_name, litellm_params):
        """ Returns model info.

        {
            "general.architecture": "llama",
            "general.file_type": 2,
            "general.parameter_count": 8030261248,
            "general.quantization_version": 2,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.layer_norm_rms_epsilon": 1e-05,
            "llama.block_count": 32,
            "llama.context_length": 8192,
            "llama.embedding_length": 4096,
            "llama.feed_forward_length": 14336,
            "llama.rope.dimension_count": 128,
            "llama.rope.freq_base": 500000,
            "llama.vocab_size": 128256,
            "tokenizer.ggml.bos_token_id": 128000,
            "tokenizer.ggml.eos_token_id": 128009,
            "tokenizer.ggml.merges": null,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "llama-bpe",
            "tokenizer.ggml.token_type": null,
            "tokenizer.ggml.tokens": null
        }
        """

        try:

            api_base = litellm_params.get("api_base", None)
            if not api_base:
                return {}

            model_name = model_name.removeprefix("ollama/")
            data = {"model": model_name}
            res = requests.post(f"{api_base}/api/show", json=data, timeout=10)
            res.raise_for_status()

            result = res.json()
            return result["model_info"]
        except Exception as e:
            print(f"[Model Detail Error] {e}")
            return {}

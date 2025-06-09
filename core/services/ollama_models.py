
import time
import requests

class OllamaModels:

    def __init__(self, url):

        self.base_url = url.rstrip("/")


    def check_health(self, max_try=3, try_wait=10) -> bool:
        """ Check if the Ollama server is reachable. """

        for i in range(0, max_try):

            try:
                res = requests.get(f"{self.base_url}", timeout=10)
                if res.status_code == 200:
                    return True
            except Exception as e:
                print(f"try ({i+1}/{max_try}): Ollama server health check failed: {e}")

            time.sleep(try_wait)

        return False


    def list_models(self) -> list:
        """ Returns a list of all available model names. """

        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=10)
            res.raise_for_status()
            result = res.json()
            return [
                m["name"] for m in result.get("models", [])
            ]
        except Exception as e:
            print(f"[Model List Error] {e}")
            return []


    def get_model_details(self, model_name: str) -> dict:
        """ Returns model details.

        {
            "name":"llama3:8b",
            "model":"llama3:8b",
            "modified_at":"2025-05-30T08:45:26.500386471Z",
            "size":4661224676,
            "digest":"365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1",
            "details":{
                "parent_model":"",
                "format":"gguf",
                "family":"llama",
                "families":["llama"],
                "parameter_size":"8.0B",
                "quantization_level":"Q4_0"
            }
        }
        """

        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=10)
            res.raise_for_status()
            result = res.json()
            for model in result.get("models", []):
                if model["name"] == model_name:
                    return model
            raise ValueError(f"Model '{model_name}' not found.")
        except Exception as e:
            print(f"[Model Detail Error] {e}")
            return {}


    def get_model_info(self, model_name: str):
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
            data = {"model": model_name}
            res = requests.post(f"{self.base_url}/api/show", json=data, timeout=10)
            res.raise_for_status()
            result = res.json()
            return result["model_info"]
        except Exception as e:
            print(f"[Model Detail Error] {e}")
            return {}


    def is_available(self, model_name: str) -> bool:
        """ Checks if a model is available on the Ollama server. """

        return model_name in self.list_models()

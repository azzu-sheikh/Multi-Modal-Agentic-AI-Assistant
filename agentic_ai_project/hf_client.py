"""
hf_client.py — HuggingFace Inference API client with robust error handling.
On ANY failure (no API key, cold start, timeout, error) — raises exception
so the health monitor in models.py can immediately fall through to Groq.
"""
import os
import time
import logging
import requests
from typing import Iterator, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models"

HF_MODELS = {
    "smollm2_135m": {
        "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "max_new_tokens": 256,
    },
    "phi3_mini": {
        "id": "microsoft/Phi-3-mini-128k-instruct",
        "max_new_tokens": 512,
    },
    "mistral_7b_hf": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_new_tokens": 1024,
    },
    "zephyr_7b": {
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "max_new_tokens": 1024,
    },
}

def _build_prompt(messages: List[BaseMessage]) -> str:
    parts = []
    for m in messages:
        if isinstance(m, SystemMessage):
            parts.append(f"<|system|>\n{m.content}")
        elif isinstance(m, HumanMessage):
            parts.append(f"<|user|>\n{m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"<|assistant|>\n{m.content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)

class HFInferenceModel(BaseChatModel):
    model_key:   str   = "phi3_mini"
    temperature: float = 0.1
    timeout:     int   = 20      # short timeout — fail fast, fall to Groq

    @property
    def _llm_type(self) -> str:
        return f"hf_{self.model_key}"

    def _call_api(self, prompt: str, max_new_tokens: int) -> str:
        api_key = os.getenv("HF_API_KEY", "").strip()

        # No API key → raise immediately so fallback triggers
        if not api_key or api_key == "your_hf_api_token_here":
            raise ValueError("HF_API_KEY not set — falling through to next model tier.")

        model_id = HF_MODELS[self.model_key]["id"]
        url      = f"{HF_API_URL}/{model_id}"
        headers  = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens":   max_new_tokens,
                "temperature":      max(self.temperature, 0.01),
                "return_full_text": False,
                "do_sample":        self.temperature > 0,
            },
            "options": {
                "wait_for_model": True,   # wait during cold start instead of 503
                "use_cache":      True,
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        # 503 model loading with wait_for_model=True shouldn't happen,
        # but handle it just in case
        if resp.status_code == 503:
            raise RuntimeError(f"HF model {model_id} unavailable (503) — falling through.")

        # 401/403 bad token
        if resp.status_code in (401, 403):
            raise ValueError(f"HF API auth failed ({resp.status_code}) — check HF_API_KEY.")

        # 429 rate limit
        if resp.status_code == 429:
            raise RuntimeError("HF API rate limited (429) — falling through to next tier.")

        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "").strip()
        elif isinstance(data, dict):
            text = data.get("generated_text", "").strip()
        else:
            text = str(data).strip()

        if not text:
            raise RuntimeError("HF API returned empty response — falling through.")

        return text

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt         = _build_prompt(messages)
        max_new_tokens = HF_MODELS[self.model_key]["max_new_tokens"]
        text           = self._call_api(prompt, max_new_tokens)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def stream(self, messages, **kwargs) -> Iterator[AIMessage]:
        result = self._generate(messages)
        text   = result.generations[0].message.content
        words  = text.split(" ")
        for i, word in enumerate(words):
            yield AIMessage(content=word + (" " if i < len(words) - 1 else ""))

    @property
    def _identifying_params(self) -> dict:
        return {"model_key": self.model_key}


# Pre-built instances
hf_smollm2 = HFInferenceModel(model_key="smollm2_135m", temperature=0.1)
hf_phi3    = HFInferenceModel(model_key="phi3_mini",    temperature=0.1)
hf_mistral = HFInferenceModel(model_key="mistral_7b_hf",temperature=0.1)
hf_zephyr  = HFInferenceModel(model_key="zephyr_7b",    temperature=0.1)
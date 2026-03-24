"""
models.py — Zero Gemini. Full stack: HuggingFace → Groq → Mistral → OpenRouter.

Tier 1: HuggingFace Inference API (free, no local RAM)
Tier 2: Groq (fast, but watch TPM limits)
Tier 3: Mistral API (reliable fallback)
Tier 4: OpenRouter (last resort)
"""
import os
import time
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from hf_client import hf_smollm2, hf_phi3, hf_mistral, hf_zephyr

load_dotenv()
logger = logging.getLogger(__name__)

# ── Task types ────────────────────────────────────────────────────
class TaskType:
    CHAT      = "chat"
    TOOL_USE  = "tool_use"
    SENSITIVE = "sensitive"
    CODING    = "coding"
    REASONING = "reasoning"
    RAG       = "rag"
    CREATIVE  = "creative"
    MATH      = "math"

# ── Health tracker ────────────────────────────────────────────────
@dataclass
class ModelHealth:
    name:           str
    failures:       int   = 0
    disabled_until: float = 0.0
    total_calls:    int   = 0
    total_latency:  float = 0.0

    @property
    def is_healthy(self) -> bool:
        return time.time() > self.disabled_until

    @property
    def avg_latency(self) -> float:
        return (self.total_latency / self.total_calls) if self.total_calls > 0 else 0.0

    def record_failure(self):
        self.failures += 1
        backoff = min(30 * (2 ** (self.failures - 1)), 300)
        self.disabled_until = time.time() + backoff
        logger.warning(f"[Health] {self.name} disabled {backoff}s (failures={self.failures})")

    def record_success(self, latency: float):
        self.failures = max(0, self.failures - 1)
        self.total_calls += 1
        self.total_latency += latency

# ── Registry ──────────────────────────────────────────────────────
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.health = {}
        self._init_models()

    def _reg(self, name, model, tasks, priority, tier="hf"):
        self.models[name] = {"model": model, "tasks": tasks, "priority": priority, "tier": tier}
        self.health[name] = ModelHealth(name=name)

    def _init_models(self):
        ALL = [TaskType.CHAT, TaskType.TOOL_USE, TaskType.SENSITIVE,
               TaskType.CODING, TaskType.REASONING, TaskType.RAG,
               TaskType.CREATIVE, TaskType.MATH]

        # ── Tier 1: HuggingFace API (primary — no quota issues) ───
        self._reg("hf_smollm2", hf_smollm2,
                  tasks=[TaskType.SENSITIVE, TaskType.CHAT],
                  priority=1, tier="hf")

        self._reg("hf_phi3", hf_phi3,
                  tasks=[TaskType.CHAT, TaskType.TOOL_USE,
                         TaskType.CREATIVE, TaskType.SENSITIVE],
                  priority=1, tier="hf")

        self._reg("hf_mistral", hf_mistral,
                  tasks=[TaskType.CODING, TaskType.RAG,
                         TaskType.MATH, TaskType.REASONING],
                  priority=1, tier="hf")

        self._reg("hf_zephyr", hf_zephyr,
                  tasks=[TaskType.REASONING, TaskType.CREATIVE],
                  priority=2, tier="hf")

        # ── Tier 2: Groq (fast fallback — use sparingly) ──────────
        self._reg("groq_70b",
                  ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0),
                  tasks=ALL, priority=3, tier="groq")

        self._reg("groq_8b",
                  ChatGroq(model_name="llama-3.1-8b-instant", temperature=0),
                  tasks=[TaskType.CHAT, TaskType.SENSITIVE],
                  priority=4, tier="groq")

        # ── Tier 3: Mistral API (reliable general fallback) ───────
        self._reg("mistral_7b",
                  ChatMistralAI(model="open-mistral-7b", temperature=0),
                  tasks=ALL, priority=5, tier="mistral")

        self._reg("mixtral",
                  ChatMistralAI(model="open-mixtral-8x7b", temperature=0),
                  tasks=[TaskType.REASONING, TaskType.CODING, TaskType.RAG],
                  priority=5, tier="mistral")

        # ── Tier 4: OpenRouter (last resort) ──────────────────────
        self._reg("openrouter",
                  ChatOpenAI(
                      base_url="https://openrouter.ai/api/v1",
                      api_key=os.getenv("OPENROUTER_API_KEY"),
                      model="meta-llama/llama-3-8b-instruct",
                      temperature=0),
                  tasks=ALL, priority=6, tier="openrouter")

    def get_model_for_task(self, task: str, fallback_chain: list = None):
        candidates = sorted(
            [(n, i) for n, i in self.models.items()
             if task in i["tasks"] and self.health[n].is_healthy],
            key=lambda x: x[1]["priority"]
        )
        if candidates:
            return candidates[0][0], candidates[0][1]["model"]
        if fallback_chain:
            for name in fallback_chain:
                if name in self.models:
                    logger.warning(f"Using fallback: {name}")
                    return name, self.models[name]["model"]
        # Absolute last resort — groq_70b regardless of health
        return "groq_70b", self.models["groq_70b"]["model"]

    def get_health_report(self):
        return {
            n: {
                "healthy":        h.is_healthy,
                "failures":       h.failures,
                "avg_latency_ms": round(h.avg_latency * 1000, 1),
                "total_calls":    h.total_calls,
                "tier":           self.models[n]["tier"],
            }
            for n, h in self.health.items()
        }

# ── Intent classifier ─────────────────────────────────────────────
class IntentClassifier:
    PATTERNS = {
        TaskType.SENSITIVE: [
            "aadhar", "aadhaar", "bank account", "ifsc", "account number",
            "document id", "sensitive data", "bank number", "my bank",
            "my account", "pan card", "secure data",
        ],
        TaskType.CODING: [
            "code", "program", "function", "class", "debug", "script",
            "algorithm", "python", "javascript", "java", "html", "css",
            "implement", "refactor", "unit test", "fix this code",
        ],
        TaskType.REASONING: [
            "analyze", "explain", "compare", "evaluate", "summarize",
            "difference between", "pros and cons", "should i", "recommend",
            "what is the best", "why does", "how does",
        ],
        TaskType.RAG: [
            "search", "find in", "look up", "from my document",
            "based on my", "what does it say", "from the file",
        ],
        TaskType.TOOL_USE: [
            "my name", "my education", "my cgpa", "my skills", "my projects",
            "my experience", "who am i", "about me", "remember this",
            "save this", "my background", "my college", "my degree",
        ],
        TaskType.MATH: [
            "calculate", "compute", "solve", "equation", "formula",
            "percentage", "average", "sum of", "multiply", "divide",
        ],
        TaskType.CREATIVE: [
            "write a story", "poem", "essay", "brainstorm",
            "generate ideas", "cover letter", "blog post", "write an email",
        ],
    }

    @classmethod
    def classify(cls, query: str) -> str:
        q = query.lower().strip()
        scores = {task: 0 for task in cls.PATTERNS}
        for task, keywords in cls.PATTERNS.items():
            for kw in keywords:
                if kw in q:
                    scores[task] += len(kw.split())
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else TaskType.CHAT

# ── Fallback chains (no Gemini anywhere) ─────────────────────────
FALLBACK_CHAINS = {
    TaskType.SENSITIVE: ["hf_smollm2", "hf_phi3",   "groq_8b",   "mistral_7b"],
    TaskType.CHAT:      ["hf_phi3",    "groq_8b",   "mistral_7b","openrouter"],
    TaskType.TOOL_USE:  ["hf_phi3",    "groq_70b",  "mistral_7b"],
    TaskType.CODING:    ["hf_mistral", "groq_70b",  "mixtral",   "openrouter"],
    TaskType.REASONING: ["hf_zephyr",  "hf_mistral","groq_70b",  "mixtral"],
    TaskType.RAG:       ["hf_mistral", "groq_70b",  "mistral_7b"],
    TaskType.MATH:      ["hf_mistral", "groq_70b",  "openrouter"],
    TaskType.CREATIVE:  ["hf_phi3",    "hf_zephyr", "mistral_7b"],
}

# ── Singletons ────────────────────────────────────────────────────
registry   = ModelRegistry()
classifier = IntentClassifier()

# ── Public API ────────────────────────────────────────────────────
def choose_model(query: str):
    task     = classifier.classify(query)
    chain    = FALLBACK_CHAINS.get(task, ["hf_phi3", "groq_70b", "mistral_7b"])
    _, model = registry.get_model_for_task(task, chain)
    return model

def record_model_result(model_name: str, success: bool, latency: float = 0.0):
    if model_name in registry.health:
        if success:
            registry.health[model_name].record_success(latency)
        else:
            registry.health[model_name].record_failure()

def get_health_report() -> dict:
    return registry.get_health_report()

def get_task_for_query(query: str) -> str:
    return classifier.classify(query)
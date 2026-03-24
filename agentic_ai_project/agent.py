"""
agent.py — Token-optimized agent with dynamic tool injection and LLM polish.
"""
import time
import re
import logging
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from models import choose_model, record_model_result, get_task_for_query, registry
import tools as T

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Feature flags ─────────────────────────────────────────────────
features = {"web_search": False, "email": False, "calendar": False}

def set_feature(name: str, enabled: bool):
    features[name] = enabled
    _agent_cache.clear()
    logger.info(f"Feature '{name}' = {enabled}")

def get_active_tools():
    active = list(T.MEMORY_TOOLS)
    if features["web_search"]: active += T.SEARCH_TOOLS
    if features["email"]:      active += T.EMAIL_TOOLS
    if features["calendar"]:   active += T.CALENDAR_TOOLS
    return active

# ── Resume field extractor ────────────────────────────────────────
def _read_resume_field(field):
    try:
        with open("resume.txt", "r") as f:
            content = f.read()
        # Single-line fields — stop at newline (no DOTALL)
        single = {
            "name":       r"Name:[ \t]*([^\n]+)",
            "education":  r"Education:[ \t]*([^\n]+)",
            "cgpa":       r"CGPA:[ \t]*([^\n]+)",
        }
        # Multi-line fields — use DOTALL to capture across lines
        multi = {
            "skills":     r"Skills:\s*(.+?)(?:\n\nProjects|\n\nExperience|\Z)",
            "projects":   r"Projects:\s*(.+?)(?:\n\nExperience|\Z)",
            "experience": r"Experience.*?:\s*(.+?)(?:\Z)",
        }
        if field == "full":
            return content.strip()
        if field in single:
            m = re.search(single[field], content, re.IGNORECASE)
            return m.group(1).strip() if m else ""
        if field in multi:
            m = re.search(multi[field], content, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""
        return content.strip()
    except FileNotFoundError:
        return ""

def _polish_with_llm(query, raw_data):
    try:
        from models import registry, FALLBACK_CHAINS, TaskType
        _, llm = registry.get_model_for_task(
            TaskType.CHAT,
            FALLBACK_CHAINS.get(TaskType.CHAT, ["groq_70b", "mistral_7b"])
        )
        prompt = (
            "You are a personal assistant. Answer the user question in a clear, "
            "friendly and concise way using only the data below. "
            "Do not dump raw text. Give a natural polished response.\n\n"
            "Question: " + query + "\n\nData:\n" + raw_data
        )
        result = llm.invoke([HumanMessage(content=prompt)])
        return result.content.strip()
    except Exception as e:
        logger.warning(f"LLM polish failed: {e}")
        return raw_data

def _extract_from_resume(field, query=""):
    raw = _read_resume_field(field)
    if not raw:
        try:
            from db_memory import permanent_db
            if permanent_db:
                docs = permanent_db.similarity_search(field, k=2)
                raw = "\n".join(d.page_content for d in docs)
        except Exception:
            pass
    if not raw:
        return "Could not find " + field + " in resume."
    # Simple fields — instant return, zero LLM cost
    if field == "name":
        return "Your name is **" + raw + "**."
    if field == "cgpa":
        return "Your CGPA is **" + raw + "**."
    if field == "education":
        return "Your education: **" + raw + "**."
    # Rich fields — LLM polish
    return _polish_with_llm(query or "What are my " + field + "?", raw)

def _extract_sensitive(field):
    try:
        with open("sensitive_data.txt", "r") as f:
            lines = f.read().strip().splitlines()
        if field == "all":
            return "\n".join(lines)
        for line in lines:
            if field.lower() in line.lower():
                # Extract value after the colon
                if ":" in line:
                    label, _, value = line.partition(":")
                    value = value.strip()
                    if value and not value.startswith("["):
                        return label.strip() + ": **" + value + "**"
                    else:
                        return ("⚠️ " + label.strip() + " is not set yet. "
                                "Please update sensitive_data.txt with your real details.")
                return line.strip()
        return "Field not found in sensitive_data.txt."
    except FileNotFoundError:
        return "sensitive_data.txt not found."

def _extract_longterm(query):
    try:
        from db_memory import long_term_db
        docs = long_term_db.similarity_search(query, k=3)
        return "\n".join(d.page_content for d in docs) if docs else "No saved data found."
    except Exception as e:
        return "Error: " + str(e)

# ── Direct lookup patterns (0 LLM tokens for simple queries) ──────
DIRECT_PATTERNS = [
    (r"\bwhat('s| is) my name\b|\bwho am i\b",                     "resume",    "name"),
    (r"\bmy (education|college|university|degree|qualification)\b",  "resume",    "education"),
    (r"\bmy (cgpa|gpa|grade|marks)\b",                              "resume",    "cgpa"),
    (r"\bmy (skills?|tech stack|technologies|languages)\b",         "resume",    "skills"),
    (r"\bmy (projects?|portfolio|apps?|built|created|made)\b",      "resume",    "projects"),
    (r"\bmy (experience|background|career|job|worked at)\b",        "resume",    "experience"),
    (r"\bmy resume\b|\btell me about (me|myself)\b",                "resume",    "full"),
    (r"\bmy aadhar\b|\baadhaar\b",                                  "sensitive", "Aadhar"),
    (r"\bmy bank (account|number)\b|\baccount number\b",            "sensitive", "Bank Account"),
    (r"\bmy (ifsc|ifsc code|bank ifsc)\b",                          "sensitive", "Bank IFSC"),
    (r"\bmy document id\b|\bdoc id\b",                              "sensitive", "Document ID"),
    (r"\ball (my )?(sensitive|personal|private|secure) data\b",     "sensitive", "all"),
    (r"\bmy saved (notes?|facts?|data|info)\b|\bwhat did i save\b", "longterm",  ""),
]

def _match_direct(query):
    q = query.lower().strip()
    for pattern, handler, context in DIRECT_PATTERNS:
        if re.search(pattern, q):
            return handler, context
    return None, None

def _direct_lookup(handler, context, query):
    if handler == "resume":    return _extract_from_resume(context, query)
    if handler == "sensitive": return _extract_sensitive(context)
    if handler == "longterm":  return _extract_longterm(query)
    return "Unknown handler."

# ── Agent prompt ──────────────────────────────────────────────────
def _build_prompt():
    extras = []
    if features["web_search"]: extras.append("- web_search: use for current events, news, real-time info")
    if features["email"]:      extras.append("- send_email / read_emails: use for Gmail")
    if features["calendar"]:   extras.append("- create_calendar_event / list_calendar_events: use for scheduling")
    base = (
        "Assistant for Abdul Azeem Sheikh. Tool rules:\n"
        "- Resume/profile questions -> search_permanent_memory\n"
        "- Saved facts -> search_long_term_memory\n"
        "- Save/remember -> save_important_data\n"
        "- Greetings -> reply directly, no tools"
    )
    if extras:
        base += "\nActive extra tools:\n" + "\n".join(extras)
    return base

# ── Agent cache ───────────────────────────────────────────────────
_agent_cache = {}

def _get_agent(llm):
    key = id(llm)
    if key not in _agent_cache:
        _agent_cache[key] = create_agent(
            model=llm,
            tools=get_active_tools(),
            system_prompt=_build_prompt(),
        )
    return _agent_cache[key]

# ── Short-term memory ─────────────────────────────────────────────
chat_history = []
MAX_TURNS = 4

def _trim():
    global chat_history
    if len(chat_history) > MAX_TURNS * 2:
        chat_history = chat_history[-(MAX_TURNS * 2):]

# ── Main streaming function ───────────────────────────────────────
def stream_agent(query: str):
    from tools import is_safe_input
    if not is_safe_input(query):
        yield "Request blocked by safety filter."
        return

    # Fast path — direct lookup
    handler, context = _match_direct(query)
    if handler:
        logger.info("Direct lookup: " + handler + "/" + context)
        result = _direct_lookup(handler, context, query)
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result))
        yield result
        return

    # Session document QA
    from db_memory import has_session_docs
    import db_memory as dbm
    if has_session_docs():
        doc_results = dbm.session_db.similarity_search(query, k=3)
        if doc_results and len("\n".join(d.page_content for d in doc_results).strip()) > 50:
            context_text = "\n\n".join(d.page_content for d in doc_results)
            llm = choose_model(query)
            prompt = (
                "Answer based on this document content:\n" + context_text +
                "\n\nQuestion: " + query + "\n\nAnswer clearly and concisely."
            )
            full = ""
            try:
                for chunk in llm.stream([HumanMessage(content=prompt)]):
                    token = chunk.content
                    full += token
                    yield token
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=full))
                return
            except Exception:
                pass

    # Full agent — walk fallback chain
    task = get_task_for_query(query)
    _trim()
    messages = chat_history + [HumanMessage(content=query)]

    from models import FALLBACK_CHAINS
    fallback_order = list(FALLBACK_CHAINS.get(task, ["hf_phi3", "groq_70b", "mistral_7b", "openrouter"]))
    chosen_name, _ = registry.get_model_for_task(task, fallback_order)
    if not fallback_order or fallback_order[0] != chosen_name:
        fallback_order = [chosen_name] + fallback_order

    full = ""
    output = ""

    for model_name in fallback_order:
        if model_name not in registry.models:
            continue
        if not registry.health[model_name].is_healthy:
            continue
        llm   = registry.models[model_name]["model"]
        start = time.time()
        try:
            agent   = _get_agent(llm)
            result  = agent.invoke({"messages": messages})
            latency = time.time() - start
            record_model_result(model_name, success=True, latency=latency)
            output_msgs = result.get("messages", [])
            output = output_msgs[-1].content if output_msgs else str(result)
            logger.info("Response via " + model_name + " in " + str(round(latency, 2)) + "s")
            break
        except Exception as e:
            record_model_result(model_name, success=False, latency=time.time() - start)
            logger.warning(model_name + " failed: " + str(e)[:80] + " — trying next")
            continue

    if not output:
        output = "Sorry, all models are currently unavailable. Please try again in a moment."

    words = output.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        full += chunk
        yield chunk

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=full.strip()))

def run_agent(query: str) -> str:
    return "".join(stream_agent(query))

def clear_short_term_memory():
    chat_history.clear()
    _agent_cache.clear()
    logger.info("Memory cleared.")

def get_agent_info():
    from models import get_health_report
    return {
        "available_agents": ["direct_lookup", "full_agent"],
        "active_features":  {k: v for k, v in features.items()},
        "model_health":     get_health_report(),
        "history_length":   len(chat_history) // 2,
    }
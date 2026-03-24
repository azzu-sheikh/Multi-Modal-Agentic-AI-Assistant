"""
app.py — Full Streamlit UI with:
- Streaming responses
- Voice input (mic button) + TTS output (speaker button)
- Web search toggle
- Email toggle
- Calendar toggle
- Session document QA
- Password-gated sensitive data
"""
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import os
import base64
import logging

load_dotenv()
logger = logging.getLogger(__name__)

from agent import stream_agent, run_agent, clear_short_term_memory, get_agent_info, set_feature
from tools import save_important_data, TOOL_REGISTRY
from models import get_health_report, get_task_for_query
from db_memory import reset_session_db, has_session_docs

# ── Password ──────────────────────────────────────────────────────
SENSITIVE_PASSWORD = os.getenv("SENSITIVE_DATA_PASSWORD", "azeem123")
SENSITIVE_KEYWORDS = [
    "aadhar","aadhaar","bank account","ifsc","account number",
    "document id","doc id","sensitive","bank details","bank number",
    "my bank","my account","pan card","passport","routing number","swift"
]

def is_sensitive_query(q: str) -> bool:
    return any(kw in q.lower() for kw in SENSITIVE_KEYWORDS)

# ── Document ingestion ────────────────────────────────────────────
def ingest_to_session(content: str, doc_name: str) -> int:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import db_memory as dbm
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_text(content)
    if chunks:
        dbm.session_db.add_texts(chunks)
        dbm.session_doc_name = doc_name
    return len(chunks)

# ── TTS helper ────────────────────────────────────────────────────
def text_to_speech_b64(text: str) -> str | None:
    """Convert text to base64 MP3 using gTTS."""
    try:
        from gtts import gTTS
        import io
        tts = gTTS(text=text[:500], lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title="Agentic AI Assistant", page_icon="🤖", layout="wide")

# ── Session state ─────────────────────────────────────────────────
defaults = {
    "messages": [],
    "pending_sensitive_query": None,
    "show_dashboard": False,
    "session_doc_loaded": None,
    "feature_web":      False,
    "feature_email":    False,
    "feature_calendar": False,
    "feature_voice":    False,
    "feature_tts":      False,
    "voice_transcript": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sync feature flags to agent
set_feature("web_search", st.session_state.feature_web)
set_feature("email",      st.session_state.feature_email)
set_feature("calendar",   st.session_state.feature_calendar)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agentic AI")
    st.caption("Personal Assistant · Abdul Azeem Sheikh")
    st.divider()

    # Controls row
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 New Chat", use_container_width=True):
            clear_short_term_memory()
            reset_session_db()
            for k in ["messages","pending_sensitive_query","session_doc_loaded","voice_transcript"]:
                st.session_state[k] = [] if k == "messages" else None
            st.success("Cleared!")
    with col2:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.show_dashboard = not st.session_state.show_dashboard

    st.divider()

    # ── Feature toggles ───────────────────────────────────────────
    st.markdown("**⚡ Features**")

    # Web search
    prev_web = st.session_state.feature_web
    st.session_state.feature_web = st.toggle(
        "🌐 Web Search",
        value=st.session_state.feature_web,
        help="Search the internet for real-time info. Requires TAVILY_API_KEY in .env"
    )
    if st.session_state.feature_web != prev_web:
        set_feature("web_search", st.session_state.feature_web)

    # Email
    prev_email = st.session_state.feature_email
    st.session_state.feature_email = st.toggle(
        "📧 Email (Gmail)",
        value=st.session_state.feature_email,
        help="Send and read Gmail. Requires GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env"
    )
    if st.session_state.feature_email != prev_email:
        set_feature("email", st.session_state.feature_email)

    # Calendar
    prev_cal = st.session_state.feature_calendar
    st.session_state.feature_calendar = st.toggle(
        "📅 Google Calendar",
        value=st.session_state.feature_calendar,
        help="Create and list events. Requires GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_CALENDAR_ID in .env"
    )
    if st.session_state.feature_calendar != prev_cal:
        set_feature("calendar", st.session_state.feature_calendar)

    # Voice input
    st.session_state.feature_voice = st.toggle(
        "🎙️ Voice Input",
        value=st.session_state.feature_voice,
        help="Speak your query using the mic button in the chat"
    )

    # TTS output
    st.session_state.feature_tts = st.toggle(
        "🔊 Read Aloud (TTS)",
        value=st.session_state.feature_tts,
        help="AI responses are read aloud. Requires: pip install gTTS"
    )

    # Active features display
    active = []
    if st.session_state.feature_web:      active.append("🌐 Web")
    if st.session_state.feature_email:    active.append("📧 Email")
    if st.session_state.feature_calendar: active.append("📅 Calendar")
    if st.session_state.feature_voice:    active.append("🎙️ Voice")
    if st.session_state.feature_tts:      active.append("🔊 TTS")
    if active:
        st.caption("Active: " + "  ".join(active))

    st.divider()

    # Session doc status
    if st.session_state.session_doc_loaded:
        st.success(f"📄 **Loaded:** {st.session_state.session_doc_loaded}")
        st.caption("Ask anything about this document.")
    else:
        st.info("No document loaded.")

    st.divider()

    # File upload
    st.markdown("**📁 Upload Document**")
    uploaded_file = st.file_uploader("PDF or TXT", type=["txt","pdf"], label_visibility="collapsed")
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            content = "".join(p.extract_text()+"\n" for p in pdf_reader.pages)
        else:
            content = uploaded_file.read().decode("utf-8")

        priority = st.radio("Memory:", ["Short-Term (session only)", "Long-Term (permanent)"])
        if st.button("⚡ Process", use_container_width=True):
            with st.spinner("Processing..."):
                if priority == "Long-Term (permanent)":
                    save_important_data.invoke({"text": content})
                    st.success("✅ Saved to long-term memory.")
                else:
                    n = ingest_to_session(content, uploaded_file.name)
                    st.session_state.session_doc_loaded = uploaded_file.name
                    st.success(f"✅ Loaded ({n} chunks). Ask anything about it!")
                    st.rerun()

# ── Dashboard ─────────────────────────────────────────────────────
if st.session_state.show_dashboard:
    st.markdown("### 📊 Model Health")
    health = get_health_report()
    cols = st.columns(len(health))
    for i,(mname,stats) in enumerate(health.items()):
        with cols[i]:
            icon = "🟢" if stats["healthy"] else "🔴"
            st.metric(f"{icon} {mname}", f"{stats['avg_latency_ms']}ms", f"{stats['total_calls']} calls")
    st.divider()

# ── Main chat ─────────────────────────────────────────────────────
st.title("🤖 Agentic AI Assistant")
if st.session_state.session_doc_loaded:
    st.caption(f"📄 Session doc: **{st.session_state.session_doc_loaded}**")

# ── Voice input component ─────────────────────────────────────────
if st.session_state.feature_voice:
    st.components.v1.html("""
    <div style="margin:8px 0">
      <button id="micBtn" onclick="toggleMic()"
        style="padding:8px 18px;border-radius:20px;border:1.5px solid #ccc;
               background:#fff;cursor:pointer;font-size:14px;transition:all .2s">
        🎙️ Hold to speak
      </button>
      <span id="micStatus" style="margin-left:10px;font-size:13px;color:#666"></span>
    </div>
    <script>
    let recognition, recording = false;
    function toggleMic() {
        if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            document.getElementById('micStatus').textContent = 'Speech not supported in this browser.';
            return;
        }
        if (recording) { recognition.stop(); return; }
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.onstart = () => {
            recording = true;
            document.getElementById('micBtn').style.background = '#fee2e2';
            document.getElementById('micBtn').textContent = '🔴 Recording...';
            document.getElementById('micStatus').textContent = 'Listening...';
        };
        recognition.onresult = (e) => {
            const t = e.results[0][0].transcript;
            document.getElementById('micStatus').textContent = 'Heard: ' + t;
            window.parent.postMessage({type:'streamlit:setComponentValue', value: t}, '*');
        };
        recognition.onend = () => {
            recording = false;
            document.getElementById('micBtn').style.background = '#fff';
            document.getElementById('micBtn').textContent = '🎙️ Hold to speak';
        };
        recognition.start();
    }
    </script>
    """, height=70)

# ── Chat history ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # TTS play button on assistant messages
        if msg["role"] == "assistant" and st.session_state.feature_tts and msg.get("audio_b64"):
            audio_html = f'<audio autoplay controls style="width:100%;margin-top:4px"><source src="data:audio/mp3;base64,{msg["audio_b64"]}" type="audio/mp3"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)

# ── Password gate ─────────────────────────────────────────────────
if st.session_state.pending_sensitive_query:
    with st.chat_message("assistant", avatar="🔒"):
        st.warning("🔒 **Sensitive data requested.** Enter your password to proceed.")
        pwd = st.text_input("Password", type="password", key="pwd_input", placeholder="Enter password...")
        c1, c2 = st.columns([1,5])
        with c1:
            confirm = st.button("✅ Confirm")
        with c2:
            cancel = st.button("✖ Cancel")

        if confirm:
            if pwd == SENSITIVE_PASSWORD:
                query = st.session_state.pending_sensitive_query
                st.session_state.pending_sensitive_query = None
                with st.spinner("🔓 Fetching..."):
                    try:
                        with open("sensitive_data.txt","r") as f: raw = f.read()
                        response = run_agent(
                            f"User asked: '{query}'. Sensitive data:\n{raw}\n"
                            "Answer only the relevant field clearly."
                        )
                    except Exception as e:
                        response = f"Error: {e}"
                audio_b64 = text_to_speech_b64(response) if st.session_state.feature_tts else None
                st.session_state.messages.append({"role":"assistant","content":response,"audio_b64":audio_b64})
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
        if cancel:
            st.session_state.pending_sensitive_query = None
            st.session_state.messages.append({"role":"assistant","content":"🔒 Access cancelled."})
            st.rerun()

# ── Chat input ────────────────────────────────────────────────────
elif query := st.chat_input("Ask me anything..."):
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    st.session_state.messages.append({"role":"user","content":query})

    if is_sensitive_query(query):
        st.session_state.pending_sensitive_query = query
        st.rerun()
    else:
        with st.chat_message("assistant", avatar="🤖"):
            task = get_task_for_query(query)
            label = ""
            if st.session_state.feature_web and any(w in query.lower() for w in ["latest","news","current","today","search","find"]):
                label = "🌐 Searching web..."
            elif st.session_state.feature_email and any(w in query.lower() for w in ["email","mail","inbox","send"]):
                label = "📧 Accessing email..."
            elif st.session_state.feature_calendar and any(w in query.lower() for w in ["calendar","event","schedule","meeting"]):
                label = "📅 Checking calendar..."
            elif has_session_docs():
                label = "📄 Searching document..."
            else:
                label = f"🧠 Thinking... `{task}`"

            with st.spinner(label):
                # ── STREAMING response ────────────────────────────
                response_placeholder = st.empty()
                full_response = ""
                for chunk in stream_agent(query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

        # TTS
        audio_b64 = text_to_speech_b64(full_response) if st.session_state.feature_tts else None
        if audio_b64 and st.session_state.feature_tts:
            audio_html = f'<audio autoplay style="display:none"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role":"assistant",
            "content": full_response,
            "audio_b64": audio_b64
        })
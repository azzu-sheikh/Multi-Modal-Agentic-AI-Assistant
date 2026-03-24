"""
tools.py — Tool registry with web search, email, calendar, and memory tools.
"""
from langchain.tools import tool
from db_memory import permanent_db, long_term_db
import logging
import re
import os

logger = logging.getLogger(__name__)

# ── Safety filter ─────────────────────────────────────────────────
BLOCKED_PATTERNS = [
    r"(ignore|forget|override).*(instruction|prompt|system)",
    r"(jailbreak|bypass|hack).*(ai|model|filter)",
    r"(delete|drop|truncate).*(table|database|db)",
    r"(rm -rf|format|shutdown|kill -9)",
]

def is_safe_input(text: str) -> bool:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text.lower()):
            logger.warning(f"Blocked unsafe input: {text[:80]}")
            return False
    return True

# ── Tool 1: Web search ────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the internet for current information, news, job postings, or any real-time data.
    Use when the user asks about recent events, current info, or anything not in personal memory.
    Requires TAVILY_API_KEY in .env
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        search = TavilySearchResults(max_results=3)
        results = search.invoke(query)
        if not results:
            return "No results found."
        out = []
        for r in results:
            out.append(f"**{r.get('title','Result')}**\n{r.get('content','')}\nSource: {r.get('url','')}")
        return "\n\n".join(out)
    except Exception as e:
        return f"Web search error: {e}. Make sure TAVILY_API_KEY is set in your .env file."

# ── Tool 2: Send email ────────────────────────────────────────────
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email via Gmail SMTP.
    Args: to (recipient email), subject (email subject), body (email body text).
    Requires GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        gmail_address  = os.getenv("GMAIL_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")

        if not gmail_address or not gmail_password:
            return "Email not configured. Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD to your .env file."

        msg = MIMEMultipart()
        msg["From"]    = gmail_address
        msg["To"]      = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, to, msg.as_string())

        return f"Email sent to {to} with subject '{subject}'."
    except Exception as e:
        return f"Email error: {e}"

# ── Tool 3: Read emails ───────────────────────────────────────────
@tool
def read_emails(n: int = 5) -> str:
    """
    Read the latest N emails from Gmail inbox.
    Requires GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env
    """
    try:
        import imaplib
        import email as email_lib
        from email.header import decode_header

        gmail_address  = os.getenv("GMAIL_ADDRESS")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")

        if not gmail_address or not gmail_password:
            return "Email not configured. Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD to your .env file."

        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(gmail_address, gmail_password)
        mail.select("inbox")

        _, msgs = mail.search(None, "ALL")
        ids = msgs[0].split()[-n:]
        results = []

        for uid in reversed(ids):
            _, data = mail.fetch(uid, "(RFC822)")
            msg = email_lib.message_from_bytes(data[0][1])
            subject, enc = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(enc or "utf-8", errors="replace")
            sender = msg.get("From", "Unknown")
            date   = msg.get("Date", "")
            results.append(f"From: {sender}\nDate: {date}\nSubject: {subject}")

        mail.logout()
        return "\n\n".join(results) if results else "No emails found."
    except Exception as e:
        return f"Email read error: {e}"

# ── Tool 4: Create calendar event ────────────────────────────────
@tool
def create_calendar_event(summary: str, start_datetime: str, end_datetime: str, description: str = "") -> str:
    """
    Create a Google Calendar event.
    Args:
      summary: event title
      start_datetime: ISO format e.g. '2025-06-01T10:00:00'
      end_datetime:   ISO format e.g. '2025-06-01T11:00:00'
      description: optional event notes
    Requires GOOGLE_CALENDAR_ID and Google service account credentials.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds_file   = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
        calendar_id  = os.getenv("GOOGLE_CALENDAR_ID", "primary")
        scopes       = ["https://www.googleapis.com/auth/calendar"]

        creds   = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
        service = build("calendar", "v3", credentials=creds)

        event = {
            "summary":     summary,
            "description": description,
            "start": {"dateTime": start_datetime, "timeZone": "Asia/Kolkata"},
            "end":   {"dateTime": end_datetime,   "timeZone": "Asia/Kolkata"},
        }
        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        return f"Event created: '{summary}' on {start_datetime}. Link: {created.get('htmlLink','')}"
    except Exception as e:
        return f"Calendar error: {e}. Ensure GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_CALENDAR_ID are set."

# ── Tool 5: List calendar events ─────────────────────────────────
@tool
def list_calendar_events(days_ahead: int = 7) -> str:
    """
    List upcoming Google Calendar events for the next N days.
    Requires GOOGLE_CALENDAR_ID and Google service account credentials.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from datetime import datetime, timezone, timedelta

        creds_file  = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
        calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
        scopes      = ["https://www.googleapis.com/auth/calendar.readonly"]

        creds   = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
        service = build("calendar", "v3", credentials=creds)

        now     = datetime.now(timezone.utc)
        end     = now + timedelta(days=days_ahead)
        events  = service.events().list(
            calendarId=calendar_id,
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute().get("items", [])

        if not events:
            return f"No events in the next {days_ahead} days."

        lines = []
        for e in events:
            start = e["start"].get("dateTime", e["start"].get("date"))
            lines.append(f"• {e.get('summary','(no title)')} — {start}")
        return "\n".join(lines)
    except Exception as e:
        return f"Calendar list error: {e}"

# ── Tool 6–9: Memory tools (unchanged) ───────────────────────────
@tool
def read_sensitive_data(query: str = "") -> str:
    """Read sensitive credentials: Aadhar, bank account, IFSC, document ID.
    ONLY call when user explicitly asks for these items by name."""
    if not is_safe_input(query):
        return "Request blocked by safety filter."
    try:
        with open("sensitive_data.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "sensitive_data.txt not found."

@tool
def search_permanent_memory(query: str) -> str:
    """Search resume/professional background: name, education, skills, projects, experience."""
    if not is_safe_input(query):
        return "Request blocked by safety filter."
    if permanent_db is None:
        return "Permanent memory not loaded. Run: python ingest_permanent.py"
    try:
        docs = permanent_db.similarity_search(query, k=3)
        return "\n---\n".join(d.page_content for d in docs) if docs else "No matching data found."
    except Exception as e:
        return f"Error: {e}"

@tool
def search_long_term_memory(query: str) -> str:
    """Search saved long-term facts from previous sessions."""
    try:
        docs = long_term_db.similarity_search(query, k=3)
        return "\n---\n".join(d.page_content for d in docs) if docs else "No data found."
    except Exception as e:
        return f"Error: {e}"

@tool
def save_important_data(text: str) -> str:
    """Save new important facts to long-term memory. Only call when user says save/remember."""
    if not is_safe_input(text):
        return "Request blocked."
    try:
        long_term_db.add_texts([text[:5000]])
        long_term_db.save_local("faiss_long_term")
        return f"Saved: '{text[:80]}...'" if len(text) > 80 else f"Saved: '{text}'"
    except Exception as e:
        return f"Error saving: {e}"

# ── Tool sets ─────────────────────────────────────────────────────
MEMORY_TOOLS  = [read_sensitive_data, search_permanent_memory, search_long_term_memory, save_important_data]
SEARCH_TOOLS  = [web_search]
EMAIL_TOOLS   = [send_email, read_emails]
CALENDAR_TOOLS= [create_calendar_event, list_calendar_events]
ALL_TOOLS     = MEMORY_TOOLS + SEARCH_TOOLS + EMAIL_TOOLS + CALENDAR_TOOLS

# ── Tool registry for UI ──────────────────────────────────────────
TOOL_REGISTRY = {
    "read_sensitive_data":     {"fn": read_sensitive_data,     "category": "Security", "desc": "Read sensitive credentials"},
    "search_permanent_memory": {"fn": search_permanent_memory, "category": "Memory",   "desc": "Search resume & profile"},
    "search_long_term_memory": {"fn": search_long_term_memory, "category": "Memory",   "desc": "Search saved facts"},
    "save_important_data":     {"fn": save_important_data,     "category": "Memory",   "desc": "Save new facts"},
    "web_search":              {"fn": web_search,              "category": "Web",      "desc": "Search the internet"},
    "send_email":              {"fn": send_email,              "category": "Email",    "desc": "Send Gmail email"},
    "read_emails":             {"fn": read_emails,             "category": "Email",    "desc": "Read Gmail inbox"},
    "create_calendar_event":   {"fn": create_calendar_event,   "category": "Calendar", "desc": "Create Google Calendar event"},
    "list_calendar_events":    {"fn": list_calendar_events,    "category": "Calendar", "desc": "List upcoming events"},
}
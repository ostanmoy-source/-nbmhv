# app.py — ChatDouble v2
import os
import re
import json
from datetime import datetime

import streamlit as st
from sentence_transformers import SentenceTransformer
import google.genai as genai

from supabase_db import (
    get_user_bots, add_bot, delete_bot, update_bot, update_bot_persona,
    register_user, login_user, get_bot_file,
    save_chat_history_cloud, load_chat_history_cloud,
    save_embeddings, search_embeddings
)

# ─────────────────────────────────────────────────────────
# Page config + Gemini
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="ChatDouble", page_icon="🤖", layout="wide")

API_KEY = os.getenv("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY") if st.secrets else None)
genai_client = genai.Client(api_key=API_KEY) if API_KEY else None

os.makedirs("chats", exist_ok=True)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

#MainMenu, header, footer { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }

html, body, [data-testid="stAppViewContainer"] {
  background: #07070f;
  color: #dde4f0;
  font-family: 'DM Sans', sans-serif;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: #08080f !important;
  border-right: 1px solid #16162a !important;
  min-width: 260px !important;
  max-width: 260px !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── main block ── */
.block-container {
  padding: 28px 40px 60px 40px !important;
  max-width: 880px !important;
}

/* ── logo ── */
.cd-logo {
  display: flex; align-items: center; gap: 12px;
  padding: 22px 18px 16px 18px;
  border-bottom: 1px solid #16162a;
  margin-bottom: 12px;
}
.cd-logo-badge {
  width: 42px; height: 42px; border-radius: 12px;
  background: linear-gradient(135deg, #7c6fef 0%, #3ecfea 100%);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif;
  font-weight: 800; font-size: 15px; color: #fff; flex-shrink: 0;
}
.cd-logo-text { line-height: 1.15; }
.cd-logo-name { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 16px; color: #fff !important; }
.cd-logo-tag  { font-size: 11px; color: #555c72 !important; margin-top: 1px; }

/* ── nav ── */
.nav-section { padding: 0 10px; margin-bottom: 6px; }
.nav-label {
  font-size: 10px; font-weight: 600; letter-spacing: 1.2px;
  color: #353550 !important; text-transform: uppercase;
  padding: 8px 8px 4px 8px;
}
.nav-btn {
  display: flex; align-items: center; gap: 11px;
  padding: 10px 12px; border-radius: 10px;
  font-size: 14px; font-weight: 500; cursor: pointer;
  margin-bottom: 2px; color: #7a829a !important;
  transition: all 0.15s ease;
  text-decoration: none !important;
}
.nav-btn:hover { background: #111120; color: #c8d0e8 !important; }
.nav-btn.active {
  background: linear-gradient(90deg, #13132a 0%, #0f0f22 100%);
  color: #fff !important;
  border-left: 3px solid #7c6fef;
  padding-left: 9px;
}
.nav-icon { font-size: 17px; width: 20px; text-align: center; flex-shrink: 0; }

/* ── user pill ── */
.user-pill {
  margin: 10px 10px 0 10px;
  padding: 10px 14px; border-radius: 10px;
  background: #0e0e1e; border: 1px solid #1c1c30;
  display: flex; align-items: center; gap: 10px;
}
.user-pill-avatar {
  width: 32px; height: 32px; border-radius: 50%;
  background: linear-gradient(135deg, #7c6fef, #3ecfea);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 13px; color: #fff; flex-shrink: 0;
}
.user-pill-name { font-size: 13px; font-weight: 600; color: #d0d8f0 !important; }
.user-pill-role { font-size: 11px; color: #454560 !important; }

/* ── page header ── */
.page-header {
  margin-bottom: 28px;
}
.page-header h1 {
  font-family: 'Syne', sans-serif;
  font-size: 28px; font-weight: 800;
  color: #fff; margin: 0 0 4px 0; line-height: 1.1;
}
.page-header p { font-size: 14px; color: #555c72; margin: 0; }

/* ── cards ── */
.cd-card {
  background: #0c0c18;
  border: 1px solid #18182e;
  border-radius: 14px;
  padding: 20px 22px;
  margin-bottom: 16px;
}
.cd-card-title {
  font-family: 'Syne', sans-serif;
  font-size: 15px; font-weight: 700; color: #c8d0e8;
  margin-bottom: 14px;
}

/* ── bot card ── */
.bot-card {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 14px; border-radius: 12px;
  border: 1px solid #16162a; background: #0a0a16;
  margin-bottom: 8px; cursor: pointer;
  transition: all 0.15s ease;
}
.bot-card:hover  { border-color: #2a2a45; background: #0e0e20; }
.bot-card.active { border-color: #7c6fef; background: #0f0f22; }
.bot-avatar {
  width: 42px; height: 42px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif;
  font-weight: 800; font-size: 16px; color: #fff; flex-shrink: 0;
}
.bot-name { font-weight: 600; font-size: 14px; color: #d0d8f0; }
.bot-persona { font-size: 12px; color: #454560; margin-top: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px; }

/* ── chat window (iframe) ── */
.chat-frame-wrap {
  border-radius: 14px; overflow: hidden;
  border: 1px solid #16162a;
  background: #08080f;
}

/* ── chat header ── */
.chat-top {
  display: flex; align-items: center; gap: 14px;
  padding: 14px 18px;
  background: #0c0c18; border-bottom: 1px solid #16162a;
  border-radius: 14px 14px 0 0;
}
.chat-top-name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 16px; color: #fff; }
.chat-top-persona { font-size: 12px; color: #555c72; }
.chat-online { width: 8px; height: 8px; border-radius: 50%; background: #25D366; flex-shrink: 0; }

/* ── input row ── */
.stTextInput > div > div > input {
  background: #0c0c18 !important;
  border: 1px solid #1e1e32 !important;
  border-radius: 12px !important;
  color: #dde4f0 !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
  border-color: #7c6fef !important;
  box-shadow: 0 0 0 3px rgba(124,111,239,0.12) !important;
}
.stTextInput > label { display: none !important; }

/* ── buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #7c6fef, #5a4fcf) !important;
  color: #fff !important; border: none !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 14px !important;
  padding: 9px 18px !important;
  transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* danger button override via class trick */
.danger-btn > button {
  background: #1f0a0a !important;
  color: #f47 !important;
  border: 1px solid #3a1010 !important;
}
.secondary-btn > button {
  background: #0e0e1e !important;
  color: #9aa3c0 !important;
  border: 1px solid #1c1c30 !important;
}

/* ── speaker picker ── */
.speaker-chip {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 16px; border-radius: 20px;
  border: 1px solid #1e1e32; background: #0c0c18;
  font-size: 14px; color: #9aa3c0;
  cursor: pointer; margin: 4px;
  transition: all 0.15s;
}
.speaker-chip:hover { border-color: #7c6fef; color: #fff; }
.speaker-chip.selected { border-color: #7c6fef; background: #13132a; color: #fff; }

/* ── loading dots ── */
.typing-dots {
  display: inline-flex; gap: 4px; align-items: center; padding: 6px 0;
}
.typing-dots span {
  width: 7px; height: 7px; border-radius: 50%; background: #7c6fef;
  animation: bounce 1.2s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40%            { transform: scale(1.1); opacity: 1;   }
}

/* ── alert/info boxes ── */
.stAlert { border-radius: 10px !important; }
[data-testid="stNotification"] { border-radius: 10px !important; }

/* ── file uploader ── */
[data-testid="stFileUploader"] {
  background: #0c0c18 !important;
  border: 1px dashed #1e1e32 !important;
  border-radius: 12px !important;
}

/* ── expander ── */
.streamlit-expanderHeader {
  background: #0c0c18 !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
}

/* ── divider ── */
hr { border-color: #14142a !important; margin: 14px 0 !important; }

/* ── home feature grid ── */
.feature-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 20px; }
.feature-tile {
  padding: 18px 20px; border-radius: 14px;
  border: 1px solid #16162a; background: #0a0a16;
}
.feature-tile-icon { font-size: 24px; margin-bottom: 10px; }
.feature-tile-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 15px; color: #c8d0e8; margin-bottom: 6px; }
.feature-tile-desc  { font-size: 13px; color: #555c72; line-height: 1.5; }

/* ── step list ── */
.step-list { counter-reset: steps; list-style: none; padding: 0; margin: 0; }
.step-list li {
  counter-increment: steps;
  display: flex; align-items: flex-start; gap: 12px;
  padding: 10px 0; border-bottom: 1px solid #12121e; font-size: 14px; color: #9aa3c0;
}
.step-list li::before {
  content: counter(steps);
  min-width: 26px; height: 26px; border-radius: 50%;
  background: #13132a; border: 1px solid #2a2a45;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 12px; color: #7c6fef; flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def parse_speakers(raw_text: str) -> dict[str, list[str]]:
    """
    Parse a WhatsApp export and return {speaker_name: [message, ...]}
    Handles both Android and iOS formats, skips system messages.
    """
    # Android: DD/MM/YYYY, HH:MM - Name: msg
    # iOS:     [DD/MM/YYYY, HH:MM:SS] Name: msg
    pattern = re.compile(
        r"(?:\[)?(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}),?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP][mM])?(?:\])?[\s\-–]+([^:]+?):\s(.+)"
    )
    system_skip = re.compile(
        r"(Messages and calls are end-to-end|end-to-end encrypted|was added|left|changed the subject|changed this group|deleted this message|You deleted|media omitted|<Media omitted>)",
        re.IGNORECASE
    )
    speakers: dict[str, list[str]] = {}
    for line in raw_text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        speaker = m.group(2).strip()
        content = m.group(3).strip()
        if system_skip.search(content):
            continue
        if len(content.split()) < 2:
            continue
        speakers.setdefault(speaker, []).append(content)
    return speakers


def generate_persona(text_examples: str) -> str:
    if not text_examples or not genai_client:
        return ""
    prompt = f"""Take these example messages from a single person and write a 1-2 sentence persona description capturing their tone, slang, and typical phrases.

Examples:
{text_examples}

Return only the short persona description."""
    try:
        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=prompt,
        )
        text = getattr(resp, "text", None) or ""
        return text.strip().splitlines()[0][:240]
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def get_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_and_store_embeddings(username: str, bot_name: str, bot_text: str):
    lines = [l.strip() for l in bot_text.splitlines() if l.strip()]
    if not lines:
        lines = ["hello"]
    model = get_embed_model()
    embeddings = model.encode(lines, convert_to_numpy=True)
    save_embeddings(username, bot_name, lines, embeddings)


def retrieve_context(username: str, bot_name: str, user_msg: str, top_k: int = 20) -> str:
    model = get_embed_model()
    vec = model.encode([user_msg])[0]
    lines = search_embeddings(username, bot_name, vec, top_k=top_k)
    return "\n".join(lines[:12])[:3000]


def bot_color(name: str) -> str:
    colors = [
        "linear-gradient(135deg,#7c6fef,#5a4fcf)",
        "linear-gradient(135deg,#3ecfea,#2a9ab0)",
        "linear-gradient(135deg,#f47,#c02050)",
        "linear-gradient(135deg,#f8a14f,#d06020)",
        "linear-gradient(135deg,#4fe88a,#2a9a58)",
    ]
    return colors[sum(ord(c) for c in name) % len(colors)]


def bot_initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[:2].upper()


def build_prompt(bot_name: str, persona: str, recent_history: str, retrieved: str, user_msg: str, username: str) -> str:
    persona_block = f"Persona: {persona}\n\n" if persona else ""
    safe_username = username  # used only for history labeling, not injected as placeholder
    retrieved_examples = retrieved[:3000] if len(retrieved) > 3000 else retrieved
    recent = recent_history[-4000:] if len(recent_history) > 4000 else recent_history
    return f"""{persona_block}You are a real human being who has chatted with this user before.

RULES:
1) The 'Recent conversation' below is absolute truth — do NOT contradict any facts.
2) Determine your own real name from the examples (text before ":"). The label '{bot_name}' is NOT your real name unless examples confirm it.
3) If persona is empty, infer your personality from the examples and stick to it.
4) NEVER produce placeholders like {{name}} or [name]. Use only names actually in the data.
STRICT RULES:
- NEVER use placeholders like [User], [User's Name], or anything inside [], <>, {{}}.
- NEVER guess names. ONLY use names that exist in the real chat data.
- If you do NOT know a name, say "I don't know, you never told me."
- NEVER use markdown formatting like **bold**, __underline__, *, ~.
- Match emoji frequency from the examples. Don't exaggerate.
- Speak casually like in the chat data. Not like an assistant.

--- Recent conversation ---
{recent}

--- Real chat examples from export ---
{retrieved_examples}

Continue naturally.

User: {user_msg}
{bot_name}:"""


# ─────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────
defaults = {
    "logged_in": False,
    "username": "",
    "page": "home",
    "selected_bot": None,
    "upload_speakers": None,
    "upload_raw": None,
    "upload_filename": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Restore session from query param (survives page refresh) ──
if not st.session_state.logged_in:
    qp_user = st.query_params.get("user", "")
    if qp_user:
        st.session_state.logged_in = True
        st.session_state.username = qp_user
        if st.session_state.page == "home":
            st.session_state.page = "chat"


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="cd-logo">
      <div class="cd-logo-badge">CD</div>
      <div class="cd-logo-text">
        <div class="cd-logo-name">ChatDouble</div>
        <div class="cd-logo-tag">Personal bots from chat exports</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav
    pages = [
        ("home",   "🏠", "Home"),
        ("chat",   "💬", "Chat"),
        ("manage", "🧰", "Manage Bots"),
    ]
    st.markdown('<div class="nav-section"><div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    for pid, icon, label in pages:
        active_class = "active" if st.session_state.page == pid else ""
        if st.button(f"{icon}  {label}", key=f"nav_{pid}", use_container_width=True):
            st.session_state.page = pid
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Auth
    if not st.session_state.logged_in:
        st.markdown('<div class="nav-section"><div class="nav-label">Account</div>', unsafe_allow_html=True)
        mode = st.radio("", ["Login", "Register"], index=0, key="auth_mode", horizontal=True)
        u = st.text_input("Username", key="sb_user")
        p = st.text_input("Password", type="password", key="sb_pass")
        if st.button(mode, key="auth_btn", use_container_width=True):
            if not u.strip() or not p.strip():
                st.error("Fill both fields.")
            elif mode == "Login":
                with st.spinner("Logging in…"):
                    try:
                        ok = login_user(u, p)
                    except Exception as e:
                        st.error(f"Auth error: {e}")
                        ok = False
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.session_state.page = "chat"
                    # persist login in browser via query param
                    st.query_params["user"] = u
                    st.rerun()
                else:
                    st.error("Wrong credentials.")
            else:
                with st.spinner("Creating account…"):
                    try:
                        ok = register_user(u, p)
                    except Exception as e:
                        st.error(f"Register error: {e}")
                        ok = False
                if ok:
                    st.success("Registered! Now log in.")
                else:
                    st.error("Username already taken.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        initials = st.session_state.username[:2].upper()
        st.markdown(f"""
        <div class="user-pill">
          <div class="user-pill-avatar">{initials}</div>
          <div>
            <div class="user-pill-name">{st.session_state.username}</div>
            <div class="user-pill-role">Logged in</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Sign out", key="logout_btn", use_container_width=True):
            st.query_params.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ─────────────────────────────────────────────────────────
# ── PAGE: Home ──
# ─────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown("""
    <div class="page-header">
      <h1>Bring conversations back to life.</h1>
      <p>Upload a WhatsApp export, pick a person, and chat with their digital double.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
      <div class="feature-tile">
        <div class="feature-tile-icon">🗂️</div>
        <div class="feature-tile-title">Upload any export</div>
        <div class="feature-tile-desc">Works with WhatsApp .txt exports on both Android and iOS. Handles group chats too.</div>
      </div>
      <div class="feature-tile">
        <div class="feature-tile-icon">🧠</div>
        <div class="feature-tile-title">Smart context retrieval</div>
        <div class="feature-tile-desc">Every reply searches their real messages for the closest match before generating.</div>
      </div>
      <div class="feature-tile">
        <div class="feature-tile-icon">🎭</div>
        <div class="feature-tile-title">Auto persona</div>
        <div class="feature-tile-desc">AI reads their tone, slang, and patterns and locks it into the bot automatically.</div>
      </div>
      <div class="feature-tile">
        <div class="feature-tile-icon">🔒</div>
        <div class="feature-tile-title">Private by default</div>
        <div class="feature-tile-desc">Your bots and chat history are tied to your account only.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="cd-card"><div class="cd-card-title">How to get started</div>', unsafe_allow_html=True)
    st.markdown("""
    <ol class="step-list">
      <li>Register or log in using the sidebar</li>
      <li>Go to <b>Manage Bots</b> and upload a WhatsApp .txt export</li>
      <li>Choose which person (or both) to create a bot for</li>
      <li>Open <b>Chat</b> and start talking</li>
    </ol>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.info("👈  Register or log in from the sidebar to get started.")
    else:
        if st.button("Go to Chat →", key="home_to_chat"):
            st.session_state.page = "chat"
            st.rerun()


# ─────────────────────────────────────────────────────────
# ── PAGE: Chat ──
# ─────────────────────────────────────────────────────────
elif st.session_state.page == "chat":
    if not st.session_state.logged_in:
        st.warning("Please log in first.")
        st.stop()

    user = st.session_state.username

    with st.spinner("Loading your bots…"):
        try:
            user_bots = get_user_bots(user) or []
        except Exception as e:
            st.error(f"Could not load bots: {e}")
            user_bots = []

    if not user_bots:
        st.markdown("""
        <div class="page-header">
          <h1>Chat</h1>
          <p>No bots yet — create one first.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Create a bot →"):
            st.session_state.page = "manage"
            st.rerun()
        st.stop()

    # auto-select first bot
    if st.session_state.selected_bot not in [b["name"] for b in user_bots]:
        st.session_state.selected_bot = user_bots[0]["name"]

    # layout: bot list left | chat right
    col_bots, col_chat = st.columns([1, 2.6])

    with col_bots:
        st.markdown("<div style='padding-top:4px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:11px;font-weight:600;letter-spacing:1px;color:#353550;text-transform:uppercase;margin-bottom:10px'>Your Bots</div>", unsafe_allow_html=True)
        for b in user_bots:
            is_active = (b["name"] == st.session_state.selected_bot)
            active_cls = "active" if is_active else ""
            persona_preview = (b.get("persona") or "")[:50]
            initials = bot_initials(b["name"])
            color = bot_color(b["name"])
            st.markdown(f"""
            <div class="bot-card {active_cls}" onclick="">
              <div class="bot-avatar" style="background:{color}">{initials}</div>
              <div>
                <div class="bot-name">{b['name']}</div>
                <div class="bot-persona">{persona_preview or "No persona"}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {b['name']}", key=f"sel_{b['name']}", use_container_width=True):
                st.session_state.selected_bot = b["name"]
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chat:
        selected_bot = st.session_state.selected_bot
        bot_meta = next((b for b in user_bots if b["name"] == selected_bot), None)
        persona = bot_meta.get("persona", "") if bot_meta else ""

        # load bot file
        with st.spinner(f"Loading {selected_bot}'s data…"):
            try:
                res = get_bot_file(user, selected_bot)
                if isinstance(res, (list, tuple)):
                    bot_text = res[0]
                    persona = res[1] if len(res) > 1 else persona
                else:
                    bot_text = res or ""
            except Exception as e:
                st.error(f"Could not load bot data: {e}")
                st.stop()

        if not bot_text.strip():
            st.warning(f"{selected_bot} has no data. Re-upload in Manage Bots.")
            st.stop()

        # load chat history
        chat_key = f"chat_{selected_bot}_{user}"
        if chat_key not in st.session_state:
            with st.spinner("Loading chat history…"):
                st.session_state[chat_key] = load_chat_history_cloud(user, selected_bot) or []

        messages = st.session_state[chat_key]

        # chat header
        initials = bot_initials(selected_bot)
        color = bot_color(selected_bot)
        st.markdown(f"""
        <div class="chat-top">
          <div class="bot-avatar" style="background:{color};width:38px;height:38px;font-size:14px">{initials}</div>
          <div style="flex:1">
            <div class="chat-top-name">{selected_bot}</div>
            <div class="chat-top-persona">{persona or "No persona set"}</div>
          </div>
          <div class="chat-online"></div>
        </div>
        """, unsafe_allow_html=True)

        # build iframe HTML for chat bubbles
        clean_history = []
        for m in messages:
            if "user" in m:
                clean_history.append({"role": "user",  "content": m["user"]})
            if "bot"  in m:
                clean_history.append({"role": "bot",   "content": m["bot"]})

        is_typing = st.session_state.get(f"typing_{chat_key}", False)
        history_json = json.dumps(clean_history)

        iframe_html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
  height: 460px;
  max-height: 460px;
  background: #08080f;
  font-family: -apple-system, 'Segoe UI', sans-serif;
  overflow: hidden;
}}
.chat-box {{
  height: 460px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 14px 12px 10px 12px;
  scrollbar-width: thin;
  scrollbar-color: #2a2a45 transparent;
}}
.chat-box::-webkit-scrollbar {{ width: 5px; }}
.chat-box::-webkit-scrollbar-thumb {{ background: #2a2a45; border-radius: 10px; }}
.row {{ display: flex; }}
.row.user {{ justify-content: flex-end; }}
.row.bot  {{ justify-content: flex-start; }}
.bubble {{
  max-width: 75%; padding: 10px 14px; font-size: 14px;
  line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;
}}
.bubble.user {{
  background: linear-gradient(135deg, #7c6fef, #5a4fcf);
  color: #fff;
  border-radius: 18px 18px 4px 18px;
}}
.bubble.bot {{
  background: #1a1a2e; color: #dde4f0;
  border-radius: 18px 18px 18px 4px;
  border: 1px solid #2a2a45;
}}
.ts {{
  font-size: 10px; color: #444460;
  margin-top: 3px; text-align: right;
}}
.typing {{
  display: flex; gap: 5px; align-items: center;
  padding: 12px 14px; background: #1a1a2e;
  border-radius: 18px 18px 18px 4px;
  border: 1px solid #2a2a45;
  width: fit-content;
}}
.dot {{
  width: 7px; height: 7px; border-radius: 50%; background: #7c6fef;
  animation: bounce 1.2s infinite ease-in-out;
}}
.dot:nth-child(2) {{ animation-delay: 0.2s; }}
.dot:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes bounce {{
  0%,80%,100% {{ transform: scale(0.6); opacity: 0.4; }}
  40%          {{ transform: scale(1.1); opacity: 1;   }}
}}
</style></head><body>
<div id="chat" class="chat-box"></div>
<script>
const history = {history_json};
const isTyping = {'true' if is_typing else 'false'};

function renderChat() {{
  const box = document.getElementById('chat');
  box.innerHTML = '';
  history.forEach(t => {{
    const row = document.createElement('div');
    row.className = 'row ' + t.role;
    const wrap = document.createElement('div');
    const bub = document.createElement('div');
    bub.className = 'bubble ' + t.role;
    bub.textContent = t.content;
    wrap.appendChild(bub);
    if (t.ts) {{
      const ts = document.createElement('div');
      ts.className = 'ts'; ts.textContent = t.ts;
      wrap.appendChild(ts);
    }}
    row.appendChild(wrap);
    box.appendChild(row);
  }});
  if (isTyping) {{
    const row = document.createElement('div');
    row.className = 'row bot';
    const dots = document.createElement('div');
    dots.className = 'typing';
    for (let i=0;i<3;i++) {{ const d=document.createElement('div'); d.className='dot'; dots.appendChild(d); }}
    row.appendChild(dots);
    box.appendChild(row);
  }}
  box.scrollTop = box.scrollHeight;
  requestAnimationFrame(() => {{ box.scrollTop = box.scrollHeight; }});
  setTimeout(() => {{ box.scrollTop = box.scrollHeight; }}, 100);
}}
renderChat();
</script>
</body></html>"""

        from streamlit.components.v1 import html as components_html
        components_html(iframe_html, height=460, scrolling=False)

        # input bar
        if st.session_state.get("pending_clear"):
            st.session_state["chat_input_field"] = ""
            st.session_state["pending_clear"] = False

        # ── styled send button ──
        st.markdown("""
        <style>
        div[data-testid="column"]:last-child .stButton > button {
          background: linear-gradient(135deg,#7c6fef,#5a4fcf) !important;
          border-radius: 14px !important;
          height: 46px !important;
          font-size: 20px !important;
          padding: 0 !important;
          transition: transform 0.1s ease, box-shadow 0.1s ease !important;
          box-shadow: 0 4px 14px rgba(124,111,239,0.35) !important;
        }
        div[data-testid="column"]:last-child .stButton > button:hover {
          transform: scale(1.06) !important;
          box-shadow: 0 6px 20px rgba(124,111,239,0.5) !important;
        }
        div[data-testid="column"]:last-child .stButton > button:active {
          transform: scale(0.94) !important;
          box-shadow: 0 2px 8px rgba(124,111,239,0.3) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        inp_col, btn_col = st.columns([9, 1])
        with inp_col:
            user_msg = st.text_input("", key="chat_input_field", placeholder="Type a message…", label_visibility="collapsed")
        with btn_col:
            send = st.button("➤", key="send_btn", use_container_width=True)

        if send and user_msg.strip():
            ts = datetime.now().strftime("%I:%M %p")
            st.session_state[chat_key].append({"user": user_msg, "bot": "", "ts": ts})
            save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])
            st.session_state[f"typing_{chat_key}"] = True
            st.session_state["pending_clear"] = True
            st.rerun()

        # ── generate reply if last message has empty bot field ──
        msgs = st.session_state[chat_key]
        if msgs and msgs[-1].get("bot") == "" and msgs[-1].get("user"):
            pending_msg = msgs[-1]["user"]

            # retrieve context
            try:
                retrieved = retrieve_context(user, selected_bot, pending_msg)
            except Exception:
                retrieved = ""

            # build history string
            history_lines = []
            for entry in msgs[:-1]:
                if "user" in entry: history_lines.append(f"User: {entry['user']}")
                if "bot"  in entry and entry["bot"]: history_lines.append(f"{selected_bot}: {entry['bot']}")
            recent_history = "\n".join(history_lines)

            prompt = build_prompt(selected_bot, persona, recent_history, retrieved, pending_msg, user)

            # model fallback chain: 2.5-flash → 2.0-flash-exp → 2.0-flash → 1.5-flash
            MODEL_CHAIN = [
                "gemini-2.5-flash-preview-04-17",
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
            ]

            reply = None
            gen_error = None

            if not genai_client:
                gen_error = "Gemini API key not configured."
            else:
                for model_name in MODEL_CHAIN:
                    try:
                        resp = genai_client.models.generate_content(model=model_name, contents=prompt)
                        text = getattr(resp, "text", None)
                        if text and text.strip():
                            reply = text.strip()
                            break
                    except Exception as e:
                        gen_error = str(e)
                        continue  # try next model

            if reply:
                # success — store reply
                st.session_state[chat_key][-1]["bot"] = reply
                st.session_state[chat_key][-1]["ts"] = datetime.now().strftime("%I:%M %p")
                st.session_state[f"typing_{chat_key}"] = False
                save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])
            else:
                # all models failed — remove pending message, show toast, don't store junk
                st.session_state[chat_key].pop()
                st.session_state[f"typing_{chat_key}"] = False
                save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])
                st.session_state["gen_error"] = gen_error or "All models unavailable."

            st.rerun()

        # show generation error as toast (not in chat history)
        if st.session_state.get("gen_error"):
            st.error(f"⚠️ Couldn't get a reply — {st.session_state['gen_error'][:120]}. Try again in a moment.")
            del st.session_state["gen_error"]


# ─────────────────────────────────────────────────────────
# ── PAGE: Manage Bots ──
# ─────────────────────────────────────────────────────────
elif st.session_state.page == "manage":
    if not st.session_state.logged_in:
        st.warning("Please log in first.")
        st.stop()

    user = st.session_state.username

    st.markdown("""
    <div class="page-header">
      <h1>Manage Bots</h1>
      <p>Upload exports, pick speakers, rename or delete your bots.</p>
    </div>
    """, unsafe_allow_html=True)

    # load bots
    with st.spinner("Loading bots…"):
        try:
            user_bots = get_user_bots(user) or []
        except Exception as e:
            st.error(f"Could not load bots: {e}")
            user_bots = []

    bot_count = len(user_bots)

    # ── Upload section ──
    st.markdown('<div class="cd-card">', unsafe_allow_html=True)
    st.markdown('<div class="cd-card-title">📤 Upload Chat Export</div>', unsafe_allow_html=True)

    if bot_count >= 2:
        st.warning("You have 2 bots already. Delete one below before uploading a new one.")
    else:
        up_file = st.file_uploader("Choose a WhatsApp .txt export", type=["txt"], key="manage_upload")

        if up_file:
            raw = up_file.read().decode("utf-8", "ignore")
            # Only re-parse if new file
            if st.session_state.upload_filename != up_file.name:
                with st.spinner("Parsing speakers from export…"):
                    speakers = parse_speakers(raw)
                st.session_state.upload_speakers = speakers
                st.session_state.upload_raw = raw
                st.session_state.upload_filename = up_file.name

            speakers = st.session_state.upload_speakers or {}

            if not speakers:
                st.error("❌ Could not find any messages in this file. Make sure it's an exported WhatsApp chat (.txt).")
            else:
                st.success(f"Found **{len(speakers)}** speaker(s) in this export.")

                # speaker selection UI
                st.markdown("<div style='margin:14px 0 8px 0;font-size:13px;color:#9aa3c0'>Select who to create a bot for:</div>", unsafe_allow_html=True)

                selected_speakers = []
                cols = st.columns(min(len(speakers), 4))
                for i, (name, lines) in enumerate(speakers.items()):
                    with cols[i % len(cols)]:
                        checked = st.checkbox(
                            f"{name}  ({len(lines)} msgs)",
                            key=f"spk_{name}",
                            value=False,
                        )
                        if checked:
                            selected_speakers.append(name)

                slots_left = 2 - bot_count
                if len(selected_speakers) > slots_left:
                    st.warning(f"You can only add {slots_left} more bot(s). Uncheck one.")
                    selected_speakers = selected_speakers[:slots_left]

                if selected_speakers:
                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                    if st.button(f"Create bot(s) for: {', '.join(selected_speakers)}", key="upload_confirm_btn"):
                        for name in selected_speakers:
                            lines = speakers[name]
                            bot_text = "\n".join(lines)
                            display_name = name.strip().capitalize()

                            with st.spinner(f"Generating persona for {display_name}…"):
                                persona = generate_persona("\n".join(lines[:40]))

                            with st.spinner(f"Building embeddings for {display_name}…"):
                                try:
                                    add_bot(user, display_name, bot_text, persona=persona)
                                    build_and_store_embeddings(user, display_name, bot_text)
                                    st.success(f"✅ {display_name} created — persona: {persona or '(none)'}")
                                except Exception as e:
                                    st.error(f"Error creating {display_name}: {e}")

                        st.session_state.upload_speakers = None
                        st.session_state.upload_raw = None
                        st.session_state.upload_filename = None
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Existing bots ──
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="cd-card-title">🤖 Your Bots</div>', unsafe_allow_html=True)

    if not user_bots:
        st.markdown("<div style='color:#454560;font-size:14px;padding:10px 0'>No bots yet. Upload one above.</div>", unsafe_allow_html=True)
    else:
        for b in user_bots:
            bname = b["name"]
            initials = bot_initials(bname)
            color = bot_color(bname)

            with st.expander(f"  {bname}", expanded=False):
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:12px;margin-bottom:14px'>
                  <div class="bot-avatar" style="background:{color};width:44px;height:44px;font-size:17px">{initials}</div>
                  <div>
                    <div style='font-weight:700;font-size:15px;color:#d0d8f0'>{bname}</div>
                    <div style='font-size:12px;color:#555c72;margin-top:3px'>{b.get('persona') or 'No persona'}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    new_name = st.text_input("Rename to", key=f"rename_input_{bname}", placeholder="New name…")
                    if st.button("Rename", key=f"rename_btn_{bname}"):
                        if not new_name.strip():
                            st.error("Enter a new name.")
                        else:
                            with st.spinner("Renaming and migrating data…"):
                                try:
                                    # get old data
                                    res = get_bot_file(user, bname)
                                    old_text = res[0] if isinstance(res, (list, tuple)) else (res or "")
                                    old_persona = res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else b.get("persona", "")
                                    old_history = load_chat_history_cloud(user, bname) or []

                                    new_display = new_name.strip().capitalize()
                                    add_bot(user, new_display, old_text, persona=old_persona)
                                    build_and_store_embeddings(user, new_display, old_text)
                                    save_chat_history_cloud(user, new_display, old_history)
                                    delete_bot(user, bname)
                                    st.success(f"Renamed to {new_display}.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Rename error: {e}")

                with col_b:
                    if st.button("Clear history", key=f"clr_btn_{bname}"):
                        with st.spinner("Clearing…"):
                            try:
                                save_chat_history_cloud(user, bname, [])
                                chat_key = f"chat_{bname}_{user}"
                                if chat_key in st.session_state:
                                    st.session_state[chat_key] = []
                                st.success("History cleared.")
                            except Exception as e:
                                st.error(f"Clear error: {e}")

                with col_c:
                    st.markdown("<div class='danger-btn'>", unsafe_allow_html=True)
                    if st.button("🗑 Delete bot", key=f"del_btn_{bname}"):
                        with st.spinner("Deleting…"):
                            try:
                                delete_bot(user, bname)
                                if st.session_state.selected_bot == bname:
                                    st.session_state.selected_bot = None
                                st.warning(f"{bname} deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete error: {e}")
                    st.markdown("</div>", unsafe_allow_html=True)

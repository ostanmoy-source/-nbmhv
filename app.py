import os
from datetime import datetime
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.genai as genai

from supabase_db import (
    get_user_bots, add_bot, delete_bot, update_bot,
    register_user, login_user, get_bot_file,
    save_chat_history_cloud, load_chat_history_cloud,
    save_embeddings, search_embeddings
)

# ── page config ──────────────────────────────────────────
st.set_page_config(page_title="ChatDouble", page_icon="🤖", layout="centered")

API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
genai_client = genai.Client(api_key=API_KEY) if API_KEY else None

# ── session defaults ──────────────────────────────────────
for k, v in {"logged_in": False, "username": "", "active_bot": None, "page": "chat"}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, header, footer { visibility: hidden; }

[data-testid="stAppViewContainer"] {
  background: #0e0e12;
  color: #eaf0ff;
  font-family: Inter, system-ui, sans-serif;
}
[data-testid="stSidebar"] {
  background: #13131a;
  border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 860px; }

/* bot card in sidebar */
.bot-card {
  background: #1a1a26;
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 8px;
  cursor: pointer;
  border: 1px solid #252535;
  transition: border 0.2s;
}
.bot-card:hover { border-color: #6c63ff; }
.bot-card.active { border-color: #6c63ff; background: #1e1e30; }
.bot-name { font-weight: 700; font-size: 15px; color: #fff !important; }
.bot-persona { font-size: 12px; color: #7a8499 !important; margin-top: 2px; }

/* messages */
.msg-wrap { display: flex; flex-direction: column; gap: 8px; padding: 12px 0; }
.bubble-row-user { display: flex; justify-content: flex-end; }
.bubble-row-bot  { display: flex; justify-content: flex-start; }
.bubble {
  max-width: 72%;
  padding: 10px 14px;
  border-radius: 18px;
  font-size: 15px;
  line-height: 1.5;
  word-wrap: break-word;
}
.bubble.user {
  background: linear-gradient(135deg, #25D366, #128C7E);
  color: #fff;
  border-bottom-right-radius: 4px;
}
.bubble.bot {
  background: #1e1e2e;
  color: #eaf0ff;
  border-bottom-left-radius: 4px;
  border: 1px solid #2a2a3e;
}
.ts { font-size: 10px; color: #556; margin-top: 3px; text-align: right; }

/* thinking bubble */
.thinking { opacity: 0.5; font-style: italic; }

/* upload card */
.up-card {
  background: #13131a;
  border: 1px dashed #2a2a3e;
  border-radius: 14px;
  padding: 20px 24px;
  margin-bottom: 16px;
}

/* stButton tweaks */
div.stButton > button {
  border-radius: 10px;
  font-weight: 600;
}
div.stButton > button[kind="primary"] {
  background: #6c63ff;
  border: none;
  color: #fff;
}

/* tab styling */
[data-testid="stTabs"] button {
  font-weight: 600;
  font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_bot_lines(raw_text, bot_name):
    lines = []
    name_lower = bot_name.strip().lower()
    for line in raw_text.splitlines():
        if "-" not in line or ":" not in line:
            continue
        try:
            _, msg = line.split("-", 1)
            speaker, content = msg.split(":", 1)
            if speaker.strip().lower() == name_lower and len(content.split()) > 1:
                lines.append(content.strip())
        except:
            continue
    return "\n".join(lines)


def generate_persona(text_examples):
    if not text_examples or not genai_client:
        return ""
    prompt = f"Write a 1-sentence persona from these messages capturing tone and slang:\n{text_examples}\nReturn only the sentence."
    try:
        resp = genai_client.models.generate_content(model="gemini-2.5-flash-exp", contents=prompt)
        return (getattr(resp, "text", None) or "").strip()[:200]
    except:
        return ""


def retrieve_context(username, bot_name, user_msg, top_k=20):
    try:
        model = get_embed_model()
        vec = model.encode([user_msg])[0]
        lines = search_embeddings(username, bot_name, vec, top_k=top_k)
        return "\n".join(lines[:12])[:3000]
    except:
        return ""


def get_reply(prompt):
    if not genai_client:
        return "⚠️ No Gemini API key set."
    for model in ["gemini-2.0-flash-exp", "gemini-2.0-flash"]:
        try:
            resp = genai_client.models.generate_content(model=model, contents=prompt)
            return (getattr(resp, "text", None) or "").strip()
        except:
            continue
    return "⚠️ Offline, try again."


def build_prompt(bot_name, persona, context, history, user_msg, username):
    persona_block = f"Persona: {persona}\n\n" if persona else ""
    history_text = "\n".join(
        [f"User: {e['user']}\n{bot_name}: {e['bot']}" for e in history if e.get("bot")]
    )[-4000:]
    return f"""{persona_block}You are a real human (not AI) who chatted with this user before.
RULES: Never use placeholders. Never use markdown formatting. Match tone/slang from examples exactly. Never say you're an AI.

--- Recent conversation ---
{history_text}

--- Real chat examples ---
{context}

User: {user_msg}
{bot_name}:"""


def render_messages(messages):
    if not messages:
        st.markdown("<div style='color:#444;text-align:center;padding:40px 0'>No messages yet. Say hi 👋</div>", unsafe_allow_html=True)
        return
    html = "<div class='msg-wrap'>"
    for m in messages:
        if m.get("user"):
            html += f"<div class='bubble-row-user'><div class='bubble user'>{m['user']}<div class='ts'>{m.get('ts','')}</div></div></div>"
        if m.get("bot"):
            cls = "bot thinking" if m["bot"] == "..." else "bot"
            html += f"<div class='bubble-row-bot'><div class='bubble {cls}'>{m['bot']}<div class='ts'>{m.get('ts','')}</div></div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── sidebar: auth + bot list ──────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ChatDouble")
    st.markdown("---")

    if not st.session_state.logged_in:
        tab_l, tab_r = st.tabs(["Login", "Register"])
        with tab_l:
            u = st.text_input("Username", key="l_u")
            p = st.text_input("Password", type="password", key="l_p")
            if st.button("Login", use_container_width=True):
                if login_user(u, p):
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.rerun()
                else:
                    st.error("Wrong credentials")
        with tab_r:
            u2 = st.text_input("Username", key="r_u")
            p2 = st.text_input("Password", type="password", key="r_p")
            if st.button("Register", use_container_width=True):
                if register_user(u2, p2):
                    st.success("Done! Now login.")
                else:
                    st.error("Username taken")
    else:
        st.markdown(f"👋 **{st.session_state.username}**")
        if st.button("Logout", use_container_width=True):
            for k in ["logged_in", "username", "active_bot"]:
                st.session_state[k] = False if k == "logged_in" else ""
            st.rerun()

        st.markdown("---")
        st.markdown("**Your Bots**")
        bots = get_user_bots(st.session_state.username)
        if not bots:
            st.caption("No bots yet. Create one in Manage tab.")
        for b in bots:
            is_active = st.session_state.active_bot == b["name"]
            card_cls = "bot-card active" if is_active else "bot-card"
            st.markdown(
                f"<div class='{card_cls}'>"
                f"<div class='bot-name'>{b['name']}</div>"
                f"<div class='bot-persona'>{b.get('persona','') or 'No persona'}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            if st.button(f"Chat with {b['name']}", key=f"sel_{b['name']}", use_container_width=True):
                st.session_state.active_bot = b["name"]
                st.rerun()


# ── main content ──────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("## Welcome to ChatDouble")
    st.markdown("Create lifelike chatbots from your WhatsApp exports. **Login or register** in the sidebar to get started.")
    st.markdown("""
**How it works:**
1. Register & login
2. Upload a WhatsApp .txt export in **Manage** tab
3. Chat with the bot in **Chat** tab
""")
    st.stop()

tab_chat, tab_manage = st.tabs(["💬 Chat", "🧰 Manage Bots"])

# ── CHAT TAB ─────────────────────────────────────────────
with tab_chat:
    user = st.session_state.username
    bots = get_user_bots(user)

    if not bots:
        st.info("No bots yet. Go to **Manage Bots** tab to create one.")
        st.stop()

    # bot selector
    bot_names = [b["name"] for b in bots]
    if st.session_state.active_bot not in bot_names:
        st.session_state.active_bot = bot_names[0]

    selected = st.selectbox("", bot_names,
                            index=bot_names.index(st.session_state.active_bot),
                            key="chat_bot_select",
                            label_visibility="collapsed")
    if selected != st.session_state.active_bot:
        st.session_state.active_bot = selected
        st.rerun()

    bot_text, persona = get_bot_file(user, selected)
    if not bot_text.strip():
        st.warning("Bot has no data. Re-upload in Manage tab.")
        st.stop()

    # header
    st.markdown(f"<div style='padding:10px 0 4px 0'><span style='font-size:22px;font-weight:700'>{selected}</span> &nbsp;<span style='color:#666;font-size:13px'>{persona or ''}</span></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0 12px 0;border-color:#1e1e2e'>", unsafe_allow_html=True)

    chat_key = f"chat_{selected}_{user}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = load_chat_history_cloud(user, selected) or []

    # messages
    render_messages(st.session_state[chat_key])

    # input
    col_inp, col_btn = st.columns([9, 1])
    with col_inp:
        user_msg = st.text_input("", placeholder="Type a message…", key="chat_input", label_visibility="collapsed")
    with col_btn:
        send = st.button("➤", key="send_btn", use_container_width=True)

    if send and user_msg.strip():
        ts = datetime.now().strftime("%I:%M %p")
        # add user msg + thinking placeholder
        st.session_state[chat_key].append({"user": user_msg, "bot": "...", "ts": ts})
        save_chat_history_cloud(user, selected, st.session_state[chat_key])
        st.rerun()

    # generate if last bot msg is "..."
    msgs = st.session_state.get(chat_key, [])
    if msgs and msgs[-1].get("bot") == "...":
        last_user = msgs[-1].get("user", "")
        with st.spinner(f"{selected} is typing…"):
            context = retrieve_context(user, selected, last_user)
            prompt = build_prompt(selected, persona, context, msgs[:-1], last_user, user)
            reply = get_reply(prompt)
        st.session_state[chat_key][-1]["bot"] = reply
        st.session_state[chat_key][-1]["ts"] = datetime.now().strftime("%I:%M %p")
        save_chat_history_cloud(user, selected, st.session_state[chat_key])
        st.rerun()


# ── MANAGE TAB ───────────────────────────────────────────
with tab_manage:
    user = st.session_state.username
    bots = get_user_bots(user)

    # upload section
    st.markdown("<div class='up-card'>", unsafe_allow_html=True)
    st.markdown("#### Upload Chat Export")
    st.caption("WhatsApp .txt export — max 2 bots")

    up_file = st.file_uploader("Choose .txt file", type=["txt"], key="up_file")
    up_name = st.text_input("Bot name (e.g. John)", key="up_name")

    if st.button("⬆️ Upload Bot", use_container_width=True, type="primary"):
        if len(bots) >= 2:
            st.error("Max 2 bots. Delete one first.")
        elif not up_file:
            st.error("Please choose a file.")
        elif not up_name.strip():
            st.error("Please enter a bot name.")
        else:
            with st.spinner("Processing… this may take a moment"):
                raw = up_file.read().decode("utf-8", "ignore")
                bot_lines = extract_bot_lines(raw, up_name)
                if not bot_lines.strip():
                    bot_lines = "\n".join([l for l in raw.splitlines() if len(l.split()) > 1])
                persona = generate_persona("\n".join(bot_lines.splitlines()[:40]))
                bot_display = up_name.strip().capitalize()
                try:
                    add_bot(user, bot_display, bot_lines, persona=persona)
                    # store embeddings
                    lines_list = [l for l in bot_lines.splitlines() if l.strip()]
                    model = get_embed_model()
                    embeddings = model.encode(lines_list, convert_to_numpy=True)
                    save_embeddings(user, bot_display, lines_list, embeddings)
                    st.success(f"✅ Bot **{bot_display}** created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # existing bots
    st.markdown("#### Your Bots")
    bots = get_user_bots(user)
    if not bots:
        st.caption("No bots yet.")
    for b in bots:
        with st.expander(f"🤖 {b['name']}"):
            st.caption(f"Persona: {b.get('persona') or '—'}")
            c1, c2, c3 = st.columns(3)
            with c1:
                new_name = st.text_input("Rename to", key=f"ren_{b['name']}")
                if st.button("Rename", key=f"ren_btn_{b['name']}"):
                    if new_name.strip():
                        update_bot(user, b["name"], new_name.strip())
                        st.success("Renamed!")
                        st.rerun()
                    else:
                        st.error("Enter a name.")
            with c2:
                if st.button("🗑️ Delete", key=f"del_{b['name']}"):
                    delete_bot(user, b["name"])
                    if st.session_state.active_bot == b["name"]:
                        st.session_state.active_bot = None
                    st.warning("Deleted.")
                    st.rerun()
            with c3:
                if st.button("🧹 Clear History", key=f"clr_{b['name']}"):
                    save_chat_history_cloud(user, b["name"], [])
                    chat_key = f"chat_{b['name']}_{user}"
                    if chat_key in st.session_state:
                        st.session_state[chat_key] = []
                    st.success("History cleared.")

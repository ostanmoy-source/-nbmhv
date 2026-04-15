import bcrypt
import json
import streamlit as st
from supabase import create_client, Client

# =========================================================
# Supabase client
# =========================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================================================
# Auth
# =========================================================
def register_user(username: str, password: str) -> bool:
    existing = sb.table("users").select("username").eq("username", username).execute()
    if existing.data:
        return False
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8", "ignore")
    sb.table("users").insert({"username": username, "password": hashed}).execute()
    return True


def login_user(username: str, password: str) -> bool:
    if not username:
        return False
    res = sb.table("users").select("password").eq("username", username).execute()
    if not res.data:
        return False
    stored = res.data[0]["password"]
    return bcrypt.checkpw(password.encode(), stored.encode())


# =========================================================
# Bot Management
# =========================================================
def add_bot(username: str, name: str, file_text: str, persona: str = None) -> None:
    sb.table("bots").upsert({
        "username": username,
        "name": name,
        "file_text": file_text,
        "persona": persona or ""
    }, on_conflict="username,name").execute()


def get_user_bots(username: str):
    res = sb.table("bots").select("name, persona").eq("username", username).execute()
    return [{"name": r["name"], "file": r["name"].lower(), "persona": r.get("persona", "")} for r in res.data]


def get_bot_file(username: str, bot_name: str):
    res = sb.table("bots").select("file_text, persona").eq("username", username).eq("name", bot_name).execute()
    if not res.data:
        return "", ""
    return res.data[0]["file_text"], res.data[0].get("persona", "")


def update_bot(username: str, old_name: str, new_name: str, new_file_text: str = None) -> None:
    res = sb.table("bots").select("*").eq("username", username).eq("name", old_name).execute()
    if not res.data:
        return
    data = res.data[0]
    data["name"] = new_name
    if new_file_text:
        data["file_text"] = new_file_text
    sb.table("bots").delete().eq("username", username).eq("name", old_name).execute()
    sb.table("bots").insert(data).execute()


def delete_bot(username: str, bot_name: str) -> None:
    sb.table("bots").delete().eq("username", username).eq("name", bot_name).execute()
    sb.table("bot_embeddings").delete().eq("username", username).eq("bot_name", bot_name).execute()


def update_bot_persona(username: str, bot_name: str, persona_text: str) -> None:
    sb.table("bots").update({"persona": persona_text}).eq("username", username).eq("name", bot_name).execute()


# =========================================================
# Chat History
# =========================================================
def save_chat_history_cloud(user: str, bot: str, history: list) -> None:
    sb.table("chats").upsert({
        "username": user,
        "bot_name": bot,
        "history": history
    }, on_conflict="username,bot_name").execute()


def load_chat_history_cloud(user: str, bot: str) -> list:
    res = sb.table("chats").select("history").eq("username", user).eq("bot_name", bot).execute()
    if not res.data:
        return []
    return res.data[0]["history"] or []


# =========================================================
# pgvector embeddings (replaces FAISS)
# =========================================================
def save_embeddings(username: str, bot_name: str, lines: list, embeddings) -> None:
    """Store bot lines + embeddings in Supabase. Called once on bot upload."""
    # delete old
    sb.table("bot_embeddings").delete().eq("username", username).eq("bot_name", bot_name).execute()
    rows = []
    for line, emb in zip(lines, embeddings):
        rows.append({
            "username": username,
            "bot_name": bot_name,
            "line": line,
            "embedding": emb.tolist()
        })
    # batch insert in chunks of 100
    for i in range(0, len(rows), 100):
        sb.table("bot_embeddings").insert(rows[i:i+100]).execute()


def search_embeddings(username: str, bot_name: str, query_vector, top_k: int = 20) -> list:
    """pgvector similarity search. Returns list of matching lines."""
    vec = query_vector.tolist()
    # Use Supabase RPC for vector search
    res = sb.rpc("match_bot_lines", {
        "p_username": username,
        "p_bot_name": bot_name,
        "query_embedding": vec,
        "match_count": top_k
    }).execute()
    if not res.data:
        return []
    return [r["line"] for r in res.data]

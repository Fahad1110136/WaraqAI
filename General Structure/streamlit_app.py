import os
import re
import time
import shutil
import hashlib
from datetime import datetime
import streamlit as st
import config
from lang_utils import detect_language
import json
import uuid

# Page config
st.set_page_config(
    page_title="دردشة عربية | Arabic Chat",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Security / config constants
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_MB = 15
MIN_SECONDS_BETWEEN_MESSAGES = 1.5  
ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")

SESSION_SECRET_FILE = os.path.join(config.BASE_DIR, ".session_secret")

def _get_session_secret() -> str:
    if os.path.exists(SESSION_SECRET_FILE):
        with open(SESSION_SECRET_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    secret = os.urandom(32).hex()
    with open(SESSION_SECRET_FILE, "w", encoding="utf-8") as f:
        f.write(secret)
    return secret

SESSION_SECRET = _get_session_secret()

# ---- Multi-user auth (users.json, writable, replaces secrets.toml) ----
USERS_FILE = os.path.join(config.BASE_DIR, "users.json")
MIN_USERNAME_LEN = 3
MIN_PASSWORD_LEN = 6

def sanitize_username(username: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", username.strip().lower())

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users: dict) -> None:
    tmp_path = USERS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, USERS_FILE)

def _make_salt() -> str:
    return os.urandom(16).hex()

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def register_user(username: str, password: str, confirm_password: str):
    clean_username = sanitize_username(username)

    if not clean_username or len(clean_username) < MIN_USERNAME_LEN:
        return False, f"Username must be at least {MIN_USERNAME_LEN} characters."
    if not password or len(password) < MIN_PASSWORD_LEN:
        return False, f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if password != confirm_password:
        return False, "Passwords do not match."

    users = _load_users()
    if clean_username in users:
        return False, "This username is already taken."

    salt = _make_salt()
    users[clean_username] = {
        "password_hash": _hash_password(password, salt),
        "salt": salt,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_users(users)
    return True, clean_username

def verify_user(username: str, password: str):
    clean_username = sanitize_username(username)
    users = _load_users()
    entry = users.get(clean_username)
    if not entry:
        return False, "No account found with this username."

    salt = entry.get("salt", "")
    if _hash_password(password, salt) != entry.get("password_hash"):
        return False, "Incorrect password."

    return True, clean_username

def make_session_token(username: str) -> str:
    return hashlib.sha256((SESSION_SECRET + username).encode("utf-8")).hexdigest()

def verify_session_token(username: str, token: str) -> bool:
    if not username or not token:
        return False
    users = _load_users()
    if sanitize_username(username) not in users:
        return False
    return make_session_token(sanitize_username(username)) == token

# Helpers
def is_arabic(text: str) -> bool:
    return bool(ARABIC_RANGE.search(text or ""))

def secure_filename(filename: str) -> str:
    # Strip path separators and any character that isn't safe for a filename
    filename = os.path.basename(filename)
    filename = re.sub(r"[^A-Za-z0-9_.\-]", "_", filename)
    return filename or f"upload_{int(time.time())}"

def indexes_exist() -> bool:
    return all(
        os.path.exists(p)
        for p in (config.FAISS_INDEX_PATH, config.BM25_PATH, config.CHUNKS_PATH)
    )

def groq_key_configured() -> bool:
    return bool(os.environ.get(config.GROQ_API_KEY))

# @st.cache_resource(show_spinner="Loading embedding & reranker models and FAISS indexes ...")
def load_retriever():
    from retriever import HybridRetriever
    return HybridRetriever()

def save_uploaded_files(uploaded_files) -> list:
    os.makedirs(config.DOCS_DIR, exist_ok=True)
    saved = []
    for uf in uploaded_files:
        ext = os.path.splitext(uf.name)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            st.warning(f"Skipped '{uf.name}': unsupported file type ({ext}).")
            continue

        size_mb = len(uf.getvalue()) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.warning(f"Skipped '{uf.name}': exceeds {MAX_UPLOAD_MB} MB limit.")
            continue

        safe_name = secure_filename(uf.name)
        dest_path = os.path.join(config.DOCS_DIR, safe_name)

        # avoid silently overwriting an existing document
        if os.path.exists(dest_path):
            stem, ext2 = os.path.splitext(safe_name)
            safe_name = f"{stem}_{int(time.time())}{ext2}"
            dest_path = os.path.join(config.DOCS_DIR, safe_name)

        with open(dest_path, "wb") as f:
            f.write(uf.getvalue())
        saved.append(safe_name)
    return saved

def rebuild_index():
    import ingest
    ingest.main()
    load_retriever.clear()  # force reload with fresh indexes on next use

# ---- Chat history persistence (per-user) ----
CHAT_HISTORY_DIR = os.path.join(config.BASE_DIR, "chat_history")

def _user_history_path(username: str) -> str:
    return os.path.join(CHAT_HISTORY_DIR, f"{sanitize_username(username)}.json")

def _load_all_chats(username: str) -> dict:
    path = _user_history_path(username)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_all_chats(username: str, chats: dict) -> None:
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    path = _user_history_path(username)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def _make_chat_title(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = m["content"].strip()
            return text[:40] + ("..." if len(text) > 40 else "")
    return "New chat"

def get_all_chats() -> dict:
    if "_all_chats_cache" not in st.session_state:
        st.session_state._all_chats_cache = _load_all_chats(st.session_state.username)
    return st.session_state._all_chats_cache

def save_current_chat() -> None:
    if not st.session_state.messages:
        return
    chats = get_all_chats()
    chats[st.session_state.current_chat_id] = {
        "title": _make_chat_title(st.session_state.messages),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
    }
    st.session_state._all_chats_cache = chats
    _save_all_chats(st.session_state.username, chats)

def load_chat(chat_id: str) -> None:
    chats = get_all_chats()
    entry = chats.get(chat_id)
    if not entry:
        return
    st.session_state.messages = entry.get("messages", [])
    st.session_state.current_chat_id = chat_id
    st.session_state._had_messages_before = bool(st.session_state.messages)
    st.query_params["chat_id"] = chat_id

# ---- Edit/delete chat ----
def rename_chat(chat_id: str, new_title: str) -> None:
    chats = get_all_chats()
    if chat_id in chats:
        new_title = new_title.strip() or "New chat"
        chats[chat_id]["title"] = new_title[:40] + ("..." if len(new_title) > 40 else "")
        st.session_state._all_chats_cache = chats
        _save_all_chats(st.session_state.username, chats)

def delete_chat(chat_id: str) -> None:
    chats = get_all_chats()
    if chat_id in chats:
        del chats[chat_id]
        st.session_state._all_chats_cache = chats
        _save_all_chats(st.session_state.username, chats)

    if st.session_state.current_chat_id == chat_id:
        st.session_state.messages = []
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.query_params["chat_id"] = st.session_state.current_chat_id
        st.session_state._had_messages_before = False

# ---- Auth page (register / login) ----
def _reset_session_for_new_user(username: str) -> None:
    st.session_state.authenticated = True
    st.session_state.username = username
    st.query_params["user"] = username
    st.query_params["token"] = make_session_token(username)
    st.session_state.messages = []
    st.session_state._all_chats_cache = _load_all_chats(username)
    st.session_state._had_messages_before = False

    qp_chat_id = st.query_params.get("chat_id")
    if qp_chat_id and qp_chat_id in st.session_state._all_chats_cache:
        st.session_state.current_chat_id = qp_chat_id
        st.session_state.messages = st.session_state._all_chats_cache[qp_chat_id].get("messages", [])
    else:
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.query_params["chat_id"] = st.session_state.current_chat_id

def show_auth_page() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>📖 Arabic Chat / دردشة عربية</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align:center;'>Login or create an account to continue / سجل الدخول أو أنشئ حسابًا للمتابعة</h4>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔑 Login / تسجيل الدخول", "🆕 Register / إنشاء حساب"])

        with tab_login:
            login_username = st.text_input("Username / اسم المستخدم", key="login_username")
            login_password = st.text_input("Password / كلمة المرور", type="password", key="login_password")
            if st.button("Login / دخول", use_container_width=True, key="login_btn"):
                ok, result = verify_user(login_username, login_password)
                if ok:
                    _reset_session_for_new_user(result)
                    st.rerun()
                else:
                    st.error(result)

        with tab_register:
            reg_username = st.text_input("Choose a username / اختر اسم مستخدم", key="reg_username")
            reg_password = st.text_input("Choose a password / اختر كلمة مرور", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm password / تأكيد كلمة المرور", type="password", key="reg_confirm")
            if st.button("Create account / إنشاء حساب", use_container_width=True, key="register_btn"):
                ok, result = register_user(reg_username, reg_password, reg_confirm)
                if ok:
                    _reset_session_for_new_user(result)
                    st.success(f"Account created. Welcome, {result}!")
                    st.rerun()
                else:
                    st.error(result)

# Auth gate
# Auth gate
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _qp_user = st.query_params.get("user")
    _qp_token = st.query_params.get("token")
    if _qp_user and verify_session_token(_qp_user, _qp_token):
        _reset_session_for_new_user(sanitize_username(_qp_user))
    else:
        show_auth_page()
        st.stop()

# Global styling — ChatGPT-like bubbles + RTL support for Arabic text
st.markdown(
    """
    <style>
    section[data-testid="stChatMessageContent"] p { font-size: 1.02rem; line-height: 1.7; }
    .source-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .rtl-text { direction: rtl; text-align: right; unicode-bidi: plaintext; margin: 0; }
    .ltr-text { direction: ltr; text-align: left; unicode-bidi: plaintext; margin: 0; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; margin-right: 6px;
    }
    .badge-green { background: #16412f; color: #6ee7a8; }
    .badge-amber { background: #4a3410; color: #f7c873; }
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, chunks, translation, context_found, lang}
if "last_msg_time" not in st.session_state:
    st.session_state.last_msg_time = 0.0

# ---- Chat history session init ----
if "_all_chats_cache" not in st.session_state:
    st.session_state._all_chats_cache = _load_all_chats(st.session_state.username)

if "current_chat_id" not in st.session_state:
    _qp_chat_id = st.query_params.get("chat_id")
    if _qp_chat_id and _qp_chat_id in st.session_state._all_chats_cache:
        st.session_state.current_chat_id = _qp_chat_id
        st.session_state.messages = st.session_state._all_chats_cache[_qp_chat_id].get("messages", [])
    else:
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.query_params["chat_id"] = st.session_state.current_chat_id

if "_had_messages_before" not in st.session_state:
    st.session_state._had_messages_before = False

if not st.session_state.messages and st.session_state._had_messages_before:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.query_params["chat_id"] = st.session_state.current_chat_id
    st.session_state._had_messages_before = False

# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:

    st.markdown("## 📖 Arabic Chat / دردشة عربية")
    st.caption("Retrivel · Rerank · Response · Result")

    if st.button("🆕 New chat / محادثة جديدة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.query_params["chat_id"] = st.session_state.current_chat_id
        st.rerun()

    st.divider()

# ---- Chats History (per-user) ----
    st.markdown("### 🕘 Chat History / سجل المحادثات")
    _chats = get_all_chats()
    if "_renaming_chat_id" not in st.session_state:
        st.session_state._renaming_chat_id = None

    if _chats:
        _sorted_chats = sorted(
            _chats.items(),
            key=lambda kv: kv[1].get("updated_at", ""),
            reverse=True,
        )
        for _chat_id, _entry in _sorted_chats:
            _label = _entry.get("title", "New chat")
            _is_current = _chat_id == st.session_state.current_chat_id

            if st.session_state._renaming_chat_id == _chat_id:
                _new_title = st.text_input(
                    "Rename / إعادة تسمية",
                    value=_label,
                    key=f"rename_input_{_chat_id}",
                    label_visibility="collapsed",
                )
                _rcol1, _rcol2 = st.columns([1, 1])
                with _rcol1:
                    if st.button("✅ Save / حفظ", key=f"rename_save_{_chat_id}", use_container_width=True):
                        rename_chat(_chat_id, _new_title)
                        st.session_state._renaming_chat_id = None
                        st.rerun()
                with _rcol2:
                    if st.button("❌ Cancel / إلغاء", key=f"rename_cancel_{_chat_id}", use_container_width=True):
                        st.session_state._renaming_chat_id = None
                        st.rerun()
            else:
                _button_label = f"● {_label}" if _is_current else f"○ {_label}"
                _c1, _c2, _c3 = st.columns([5, 2, 2])
                with _c1:
                    if st.button(_button_label, key=f"hist_{_chat_id}", use_container_width=True):
                        load_chat(_chat_id)
                        st.rerun()
                with _c2:
                    if st.button("✏️", key=f"rename_{_chat_id}", use_container_width=True):
                        st.session_state._renaming_chat_id = _chat_id
                        st.rerun()
                with _c3:
                    if st.button("🗑️", key=f"delete_{_chat_id}", use_container_width=True):
                        delete_chat(_chat_id)
                        st.rerun()
    else:
        st.caption("No saved chats / لا توجد محادثات محفوظة")

    
    st.divider()

    if indexes_exist():
        n_docs = 0
        try:
            n_docs = len([
                f for f in os.listdir(config.DOCS_DIR)
                if f.lower().endswith((".pdf", ".txt", ".md"))
            ])
        except Exception:
            pass
        st.markdown('<span class="badge badge-green">· Index ready / الفهرس جاهز ·</span>', unsafe_allow_html=True)
        st.caption(f"{n_docs} document(s) uploaded")
    else:
        st.markdown('<span class="badge badge-amber">· No index found / لم يتم العثور على فهرس ·</span>', unsafe_allow_html=True)
        st.caption("Upload documents and build the index")

    st.divider()

    # --- document upload / ingestion ---
    st.markdown("### 📚 Documents")
    uploaded = st.file_uploader(
        "Upload / تحميل (.pdf / .txt / .md )",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("💾 Save / حفظ ?", use_container_width=True):
        saved = save_uploaded_files(uploaded)
        if saved:
            st.success(f"Saved {len(saved)} file(s) to documents folder.")


    # ---- View uploaded files ----
    try:
        os.makedirs(config.DOCS_DIR, exist_ok=True)
        _existing_files = sorted(
            f for f in os.listdir(config.DOCS_DIR)
            if f.lower().endswith((".pdf", ".txt", ".md"))
        )
    except Exception:
        _existing_files = []

    if "_viewing_file" not in st.session_state:
        st.session_state._viewing_file = None

    if _existing_files:
        with st.expander(f"📂 View uploaded files / عرض الملفات المرفوعة ({len(_existing_files)})"):
            for _f in _existing_files:
                _fpath = os.path.join(config.DOCS_DIR, _f)
                try:
                    _size_kb = os.path.getsize(_fpath) / 1024
                    _size_label = f"{_size_kb:.1f} KB" if _size_kb < 1024 else f"{_size_kb/1024:.1f} MB"
                except Exception:
                    _size_label = "—"

                _is_open = st.session_state._viewing_file == _f

                _col1, _col2 = st.columns([3, 3])
                with _col1:
                    st.markdown(f"📄 **{_f}**  \n`{_size_label}`")
                with _col2:
                    _btn_label = "❌ Close / إغلاق" if _is_open else "🔎 View / عرض"
                    if st.button(_btn_label, key=f"toggle_{_f}", use_container_width=True):
                        st.session_state._viewing_file = None if _is_open else _f
                        st.rerun()

                if _is_open:
                    _ext = os.path.splitext(_f)[1].lower()
                    try:
                        if _ext == ".pdf":
                            import base64
                            with open(_fpath, "rb") as _fh:
                                _b64 = base64.b64encode(_fh.read()).decode("utf-8")
                            st.markdown(
                                f'<iframe src="data:application/pdf;base64,{_b64}" '
                                f'width="100%" height="500" style="border:1px solid rgba(255,255,255,0.1); border-radius:8px;"></iframe>',
                                unsafe_allow_html=True,
                            )
                        else:  # .txt / .md
                            with open(_fpath, "r", encoding="utf-8", errors="replace") as _fh:
                                _content = _fh.read()
                            st.text_area(
                                f"Content of {_f}",
                                value=_content,
                                height=300,
                                key=f"content_{_f}",
                                disabled=True,
                            )
                    except Exception as _e:
                        st.error(f"Could not preview file: {_e}")

                st.divider()
    else:
        st.caption("No documents uploaded yet / لا توجد ملفات مرفوعة بعد")

    if st.button("⚙️ (Re)build index / (إعادة) بناء الفهرس", use_container_width=True):
        if not groq_key_configured():
            st.warning("You can still build the index without a Groq key — you'll only need the key to generate answers.")
        with st.spinner("Chunking, embedding, and indexing documents ..."):
            try:
                rebuild_index()
                st.success("Index rebuilt successfully / تم إعادة بناء الفهرس بنجاح !!")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    with st.expander("⚠️ Danger zone / منطقة الخطر"):
        if st.button("🗑️ Delete index / حذف الفهرس", use_container_width=True):
            for p in (config.FAISS_INDEX_PATH, config.BM25_PATH, config.CHUNKS_PATH):
                if os.path.exists(p):
                    os.remove(p)
            load_retriever.clear()
            st.success("Index deleted / تم حذف الفهرس")

    st.divider()

    # --- display / retrieval settings ---
    st.markdown("### 🛠️ Display Settings / الإعدادات")
    show_sources = st.toggle("Show retrieved sources / إظهار المصادر المسترجعة", value=True)
    show_translation = st.toggle("Show English translation / إظهار الترجمة الإنجليزية", value=True)

    with st.expander("Configuration / إعدادات الاسترجاع (Read only)"):
        st.write(f"Search Modes: **Hybrid / Dense**")
        st.write(f"Dense Top k: **{config.DENSE_TOP_K}**")
        st.write(f"Sparse Top k: **{config.SPARSE_TOP_K}**")
        st.write(f"Final Top k: **{config.FINAL_TOP_K}**")
        st.write(f"Relevance Threshold: **{config.RELEVANCE_SCORE_THRESHOLD}**")
        st.write(f"Embedding Model: `{config.EMBEDDING_MODEL_NAME}`")
        st.write(f"Reranker Model: `{config.RERANKER_MODEL_NAME}`")
        st.write(f"LLM: `{config.GROQ_MODEL_NAME}`")

    st.divider()
    st.markdown(f"👤 **{st.session_state.username}**")
    if st.button("🚪 Logout / تسجيل خروج", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.messages = []
        st.session_state.pop("_all_chats_cache", None)
        st.session_state.pop("current_chat_id", None)
        st.session_state._had_messages_before = False
        st.query_params.clear()
        st.rerun()

# Main chat area
st.markdown(
    "<h1 style='text-align:center;'>📖 Arabic Chat / دردشة عربية</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align:center;'>💬 اسأل مستنداتك | Ask your documents</h3>",
    unsafe_allow_html=True,
)

# render history
for msg in st.session_state.messages:
    avatar = "💬" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        css_class = "rtl-text" if msg.get("lang") == "ar" or is_arabic(msg["content"]) else "ltr-text"
        st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)

        if msg["role"] == "assistant":
            badge = (
                '<span class="badge badge-green"> · </span>'
                if msg.get("context_found")
                else '<span class="badge badge-amber"> · </span>'
            )
            st.markdown(badge, unsafe_allow_html=True)

            if show_translation and msg.get("translation"):
                st.markdown("**AI Translation / الترجمة بالذكاء الاصطناعي:**")
                st.markdown(f'<div class="ltr-text">{msg["translation"]}</div>', unsafe_allow_html=True)

            if show_sources and msg.get("chunks"):
                with st.expander(f"📎 Sources / المصادر ({len(msg['chunks'])})"):
                    for i, c in enumerate(msg["chunks"], start=1):
                        st.markdown(
                            f'<div class="source-box"><b>{i}. {c["source"]}</b> '
                            f'(score: {c["rerank_score"]:.3f})<br>{c["text"][:400]}...</div>',
                            unsafe_allow_html=True,
                        )

# Chat input
prompt = st.chat_input("Type your Question here / اكتب سؤالك هنا")

if prompt:
    prompt = prompt.strip()

    # basic client-side rate limiting
    now = time.time()
    if now - st.session_state.last_msg_time < MIN_SECONDS_BETWEEN_MESSAGES:
        st.toast("Please wait a moment / يرجى الانتظار قليلاً ...")
        st.stop()
    st.session_state.last_msg_time = now

    if not groq_key_configured():
        st.error("GROQ_API_KEY is not set. Add it to your `.env` file and restart the app.")
        st.stop()

    if not indexes_exist():
        st.error("No index found yet !!\n\nUpload documents and click **(Re)build index** in the sidebar first.")
        st.stop()

    detected_lang = detect_language(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "lang": detected_lang})
    with st.chat_message("user", avatar="💬"):
        css_class = "rtl-text" if is_arabic(prompt) else "ltr-text"
        st.markdown(f'<div class="{css_class}">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Searching & Generating answer / جاري البحث وتوليد الإجابة ..."):
            try:
                retriever = load_retriever()
                chunks = retriever.search(prompt)
                context_found = bool(chunks) and chunks[0]["rerank_score"] >= config.RELEVANCE_SCORE_THRESHOLD

                from llm import answer_query, translate_to_english
                answer = answer_query(prompt, chunks, context_found=context_found)

                translation = None
                if show_translation:
                    try:
                        translation = translate_to_english(answer)
                    except Exception:
                        translation = None

            except Exception as e:
                answer = f"عذرًا، حدث خطأ أثناء معالجة سؤالك.\n\n_Error: {e}_"
                chunks, context_found, translation = [], False, None

        st.markdown(f'<div class="rtl-text">{answer}</div>', unsafe_allow_html=True)
        badge = (
            '<span class="badge badge-green"> · </span>'
            if context_found
            else '<span class="badge badge-amber"> · </span>'
        )
        st.markdown(badge, unsafe_allow_html=True)

        if show_translation and translation:
            st.markdown("**AI Translation / الترجمة بالذكاء الاصطناعي:**")
            st.markdown(f'<div class="ltr-text">{translation}</div>', unsafe_allow_html=True)

        if show_sources and chunks:
            with st.expander(f"📎 Sources / المصادر ({len(chunks)})"):
                for i, c in enumerate(chunks, start=1):
                    st.markdown(
                        f'<div class="source-box"><b>{i}. {c["source"]}</b> '
                        f'(score: {c["rerank_score"]:.3f})<br>{c["text"][:400]}...</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "translation": translation,
        "context_found": context_found,
        "lang": "ar",
    })

# ---- persist chat history ----
if st.session_state.messages:
    st.session_state._had_messages_before = True
    save_current_chat()
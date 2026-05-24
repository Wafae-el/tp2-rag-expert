import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import requests
import uuid

# --- 1. CONFIGURATION INTERFACE LUXE ---
st.set_page_config(page_title="AI Expert", layout="wide", page_icon="⚫")

# CSS pour transformer l'app en interface type "ChatGPT / Apple"
st.markdown("""
    <style>
    /* Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #ffffff; }

    /* Sidebar - Style Dark Mode pur */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }

    /* Bouton Nouvelle Discussion - Minimaliste blanc */
    div.stButton > button:first-child {
        background-color: #ffffff;
        color: #000000;
        border: none;
        border-radius: 5px;
        width: 100%;
        font-weight: 600;
        height: 45px;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover { background-color: #cccccc; }

    /* Historique - Liens discrets */
    div[data-testid="stSidebar"] button[key*="chat"] {
        background-color: transparent;
        border: none;
        text-align: left;
        color: #888888 !important;
        font-size: 14px;
        transition: 0.2s;
    }
    div[data-testid="stSidebar"] button[key*="chat"]:hover { color: #ffffff !important; }

    /* Zone de Chat */
    .stChatMessage {
        background-color: transparent !important;
        border-bottom: 1px solid #f0f0f0;
        border-radius: 0px;
    }
    
    /* Support Arabe */
    .rtl-text { direction: rtl; text-align: right; font-size: 1.1rem; line-height: 1.6; }

    /* Badge sources */
    .source-badge {
        font-size: 10px;
        color: #999;
        border: 1px solid #ddd;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIQUE TECHNIQUE (Inchangée mais optimisée) ---
def get_embeddings_api(text):
    api_url = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    try:
        res = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=10)
        return res.json() if res.status_code == 200 else None
    except: return None

@st.cache_resource
def init_clients():
    return QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"]), Groq(api_key=st.secrets["G_API"])

client_q, client_g = init_clients()

# --- 3. GESTION DES DISCUSSIONS ---
if "all_chats" not in st.session_state: st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='font-weight:600;'>AI Expert</h2>", unsafe_allow_html=True)
    if st.button("＋ Nouvelle discussion"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("<br><p style='color:#555; font-size:12px; font-weight:600;'>HISTORIQUE</p>", unsafe_allow_html=True)
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        if st.button(f"• {data['title'][:20]}", key=f"chat_{cid}"):
            st.session_state.current_chat_id = cid
            st.rerun()

# --- 5. ZONE DE CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

# En-tête de la discussion
st.markdown(f"<h3 style='font-weight:600; color:#111;'>{chat_data['title']}</h3>", unsafe_allow_html=True)
st.markdown("---")

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else: st.markdown(msg["content"])

if prompt := st.chat_input("Écrivez un message..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion": chat_data["title"] = prompt[:30]
    
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner(""): # Spinner invisible pour plus de classe
        vector = get_embeddings_api(prompt)
        if vector:
            try:
                search = client_q.query_points(collection_name="ma_base_expert", query=vector, limit=3).points
                context = "\n".join([f"- {r.payload['text']}" for r in search])
                sources = list(set([r.payload['source'] for r in search]))

                msgs = [{"role": "system", "content": f"Réponds avec ce contexte : {context}"}]
                for m in chat_data["messages"][-5:]: msgs.append({"role": m["role"], "content": m["content"]})

                res = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
                ans = res.choices[0].message.content
                
                with st.chat_message("assistant"):
                    st.markdown(ans)
                    src_html = " ".join([f'<span class="source-badge">{s}</span>' for s in sources])
                    st.markdown(src_html, unsafe_allow_html=True)
                
                chat_data["messages"].append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e: st.error("L'IA est occupée.")
        else: st.error("Connexion perdue. Réessayez.")

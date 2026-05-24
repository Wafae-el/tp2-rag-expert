import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import requests
import uuid

# --- 1. CONFIGURATION INTERFACE "CONFORT VISUEL" ---
st.set_page_config(page_title="IA Expert - Soft Dark", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    /* Fond général Ardoise Profonde (très reposant) */
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }

    /* Sidebar Grise subtile */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Titres sidebar */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p {
        color: #8b949e !important;
    }

    /* Bouton Nouvelle Discussion - Style "Pill" Moderne */
    div.stButton > button:first-child {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 20px;
        width: 100%;
        height: 40px;
        font-size: 14px;
        transition: 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #30363d;
        border-color: #8b949e;
    }

    /* Boutons Historique - Texte discret */
    div[data-testid="stSidebar"] button[key*="chat"] {
        background-color: transparent;
        border: none;
        color: #8b949e !important;
        text-align: left;
        font-size: 14px;
        padding: 5px 10px;
    }
    div[data-testid="stSidebar"] button[key*="chat"]:hover {
        color: #58a6ff !important;
        background-color: #1f242c;
    }

    /* Bulles de Chat - Style Épuré */
    .stChatMessage {
        background-color: #0e1117 !important;
        border: none !important;
        padding-top: 20px;
        padding-bottom: 20px;
    }
    
    /* Séparateur de message */
    .stChatMessage + .stChatMessage {
        border-top: 1px solid #21262d !important;
    }

    /* Support Arabe */
    .rtl-text { 
        direction: rtl; 
        text-align: right; 
        color: #c9d1d9;
        font-size: 1.1rem;
        line-height: 1.8;
    }

    /* Sources style Tags */
    .source-badge {
        font-size: 11px;
        color: #58a6ff;
        background-color: rgba(56, 139, 253, 0.1);
        padding: 3px 10px;
        border-radius: 12px;
        margin-right: 8px;
        border: 1px solid rgba(56, 139, 253, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIQUE TECHNIQUE API ---
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
    st.markdown("<h2 style='font-size: 20px; margin-bottom: 20px;'>Assistant Expert</h2>", unsafe_allow_html=True)
    if st.button("＋ Nouvelle discussion"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("<p style='margin-top: 30px; margin-bottom: 10px; font-size: 12px;'>HISTORIQUE</p>", unsafe_allow_html=True)
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        active_color = "#58a6ff" if cid == st.session_state.current_chat_id else "#8b949e"
        if st.button(f"• {data['title'][:22]}", key=f"chat_{cid}"):
            st.session_state.current_chat_id = cid
            st.rerun()

# --- 5. ZONE DE CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

# En-tête discret
st.markdown(f"<h4 style='color:#c9d1d9; font-weight:400;'>{chat_data['title']}</h4>", unsafe_allow_html=True)

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else: 
            st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion": chat_data["title"] = prompt[:30]
    
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("Analyse..."):
        vector = get_embeddings_api(prompt)
        if vector:
            try:
                search = client_q.query_points(collection_name="ma_base_expert", query=vector, limit=3).points
                context = "\n".join([f"- {r.payload['text']}" for r in search])
                sources = list(set([r.payload['source'] for r in search]))

                msgs = [{"role": "system", "content": f"Tu es un expert. Réponds avec ce contexte : {context}"}]
                for m in chat_data["messages"][-5:]: msgs.append({"role": m["role"], "content": m["content"]})

                res = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
                ans = res.choices[0].message.content
                
                with st.chat_message("assistant"):
                    st.markdown(ans)
                    src_html = " ".join([f'<span class="source-badge">{s}</span>' for s in sources])
                    st.markdown(src_html, unsafe_allow_html=True)
                
                chat_data["messages"].append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e: st.error("Délai d'attente dépassé. Réessayez.")
        else: st.error("Service d'analyse en veille. Réessayez dans 10 secondes.")

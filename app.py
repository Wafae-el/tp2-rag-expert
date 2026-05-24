import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import requests
import uuid
import time

# --- 1. CONFIGURATION & DESIGN ADOUCI ---
st.set_page_config(page_title="Assistant IA - Soft", layout="wide", page_icon="🌿")

st.markdown("""
    <style>
    /* Fond principal très doux */
    .main { background-color: #fcfcfc; }
    
    /* Sidebar Gris Anthracite (pas noir pur) */
    [data-testid="stSidebar"] {
        background-color: #2d3436;
        color: #dfe6e9;
    }
    [data-testid="stSidebar"] * { color: #dfe6e9 !important; }
    
    /* Bulles de chat grises et blanches (Style épuré) */
    .stChatMessage {
        border-radius: 10px;
        border: 1px solid #f1f1f1;
        background-color: white !important;
    }
    
    /* Bouton Nouvelle Conversation - Vert émeraude doux */
    div.stButton > button:first-child {
        background-color: #55efc4;
        color: #2d3436;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Boutons de l'historique - Gris neutre */
    div[data-testid="stSidebar"] button {
        background-color: transparent;
        border: 1px solid #636e72;
        text-align: left;
        font-size: 13px;
    }

    /* Texte Arabe */
    .rtl-text { direction: rtl; text-align: right; font-size: 18px; }
    
    /* Badges de sources - Gris perle */
    .source-badge {
        background-color: #f1f2f6;
        color: #2f3542;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 11px;
        margin-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FONCTIONS TECHNIQUES ---
def get_embeddings_api(text):
    api_url = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=15)
        if response.status_code == 200: return response.json()
        return None
    except: return None

@st.cache_resource
def init_clients():
    return QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"]), Groq(api_key=st.secrets["G_API"])

client_q, client_g = init_clients()

# --- 3. SESSION ---
if "all_chats" not in st.session_state: st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("### 🌿 Menu")
    if st.button("➕ Nouvelle discussion", use_container_width=True):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("---")
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        label = f"• {data['title']}"
        if st.button(label, key=cid, use_container_width=True):
            st.session_state.current_chat_id = cid
            st.rerun()

# --- 5. CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

st.subheader(f"💬 {chat_data['title']}")

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else: st.markdown(msg["content"])

if prompt := st.chat_input("Écrivez ici..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion": chat_data["title"] = prompt[:25]
    
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("Recherche..."):
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
                    src_html = "".join([f'<span class="source-badge">{s}</span>' for s in sources])
                    st.markdown(src_html, unsafe_allow_html=True)
                
                chat_data["messages"].append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e: st.error(f"Erreur technique : {e}")
        else: st.error("Le service d'analyse est en pause. Réessayez dans 10 secondes.")

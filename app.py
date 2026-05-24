import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import requests
import uuid
import time

# --- 1. CONFIGURATION VISUELLE (SOFT DARK) ---
st.set_page_config(page_title="IA Expert - 1024 Pro", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #8b949e !important; }
    div.stButton > button:first-child {
        background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d;
        border-radius: 20px; width: 100%; transition: 0.2s;
    }
    div.stButton > button:first-child:hover { border-color: #58a6ff; }
    .stChatMessage { background-color: #0d1117 !important; border-bottom: 1px solid #21262d !important; }
    .rtl-text { direction: rtl; text-align: right; color: #c9d1d9; font-size: 1.1rem; }
    .source-badge {
        font-size: 11px; color: #58a6ff; background-color: rgba(56, 139, 253, 0.1);
        padding: 3px 10px; border-radius: 12px; margin-right: 8px; border: 1px solid rgba(56, 139, 253, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FONCTION D'EMBEDDING (1024 DIMENSIONS VIA API) ---
def get_embeddings_1024(text):
    api_url = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    # Tentatives automatiques si le modèle dort
    for i in range(5):
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=15)
            result = response.json()
            if response.status_code == 200:
                return result
            elif response.status_code == 503: # Modèle en cours de chargement
                st.info(f"⏳ Le moteur d'analyse se réveille... Patientez 5 secondes (Tentative {i+1}/5)")
                time.sleep(5)
            else:
                st.error(f"Erreur API : {result}")
                return None
        except:
            time.sleep(2)
    return None

# --- 3. INITIALISATION ---
@st.cache_resource
def init_clients():
    q = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    g = Groq(api_key=st.secrets["G_API"])
    return q, g

client_q, client_g = init_clients()

if "all_chats" not in st.session_state: st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='font-size: 20px;'>Assistant Expert</h2>", unsafe_allow_html=True)
    if st.button("＋ Nouvelle discussion"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("<p style='margin-top: 30px; font-size: 12px;'>HISTORIQUE</p>", unsafe_allow_html=True)
    
    to_del = []
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"• {data['title'][:20]}", key=f"s_{cid}", use_container_width=True):
                st.session_state.current_chat_id = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"d_{cid}"): to_del.append(cid)

    for i in to_del:
        del st.session_state.all_chats[i]
        if i == st.session_state.current_chat_id:
            st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else str(uuid.uuid4())[:8]
        st.rerun()

# --- 5. CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

st.markdown(f"<h4 style='color: #8b949e;'>{chat_data['title']}</h4>", unsafe_allow_html=True)

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else: st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion": chat_data["title"] = prompt[:30]
    with st.chat_message("user"): st.markdown(prompt)

    # Récupération de l'embedding 1024 via API
    vector = get_embeddings_1024(prompt)
    
    if vector:
        with st.spinner("Analyse des documents..."):
            try:
                # RECHERCHE (Maintenant on envoie bien 1024 dimensions !)
                search = client_q.query_points(collection_name="ma_base_expert", query=vector, limit=3).points
                context = "\n".join([f"- {r.payload['text']}" for r in search])
                sources = list(set([r.payload['source'] for r in search]))

                # GÉNÉRATION
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
            except Exception as e: st.error(f"Erreur technique : {e}")

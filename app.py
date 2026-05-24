import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import requests
import uuid
import time

# --- 1. CONFIGURATION INTERFACE SOFT DARK ---
st.set_page_config(page_title="IA Expert Pro", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #8b949e !important; }
    div.stButton > button:first-child {
        background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d;
        border-radius: 20px; width: 100%; height: 40px; transition: 0.2s;
    }
    div.stButton > button:first-child:hover { background-color: #30363d; border-color: #8b949e; }
    .stChatMessage { background-color: #0e1117 !important; border-bottom: 1px solid #21262d !important; border-radius: 0; }
    .rtl-text { direction: rtl; text-align: right; color: #c9d1d9; font-size: 1.1rem; line-height: 1.8; }
    .source-badge {
        font-size: 11px; color: #58a6ff; background-color: rgba(56, 139, 253, 0.1);
        padding: 3px 10px; border-radius: 12px; margin-right: 8px; border: 1px solid rgba(56, 139, 253, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FONCTION D'EMBEDDING AVEC RÉVEIL AUTO ---
def get_embeddings_with_retry(text):
    api_url = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    max_retries = 10  # On essaie pendant environ 50 secondes
    for i in range(max_retries):
        try:
            res = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=15)
            data = res.json()
            
            if res.status_code == 200:
                return data
            elif res.status_code == 503 or "estimated_time" in str(data):
                # Le modèle est en train de charger
                wait_time = 5
                st.info(f"⏳ Le moteur d'analyse se réveille (Tentative {i+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                st.error(f"Erreur API : {data}")
                return None
        except Exception as e:
            time.sleep(2)
    return None

# --- 3. INITIALISATION ---
@st.cache_resource
def init_clients():
    return QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"]), Groq(api_key=st.secrets["G_API"])

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
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        if st.button(f"• {data['title'][:22]}", key=f"chat_{cid}"):
            st.session_state.current_chat_id = cid
            st.rerun()

# --- 5. CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else: st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion": chat_data["title"] = prompt[:30]
    with st.chat_message("user"): st.markdown(prompt)

    # Récupération des embeddings (avec patience)
    vector = get_embeddings_with_retry(prompt)
    
    if vector:
        with st.spinner("Analyse des documents..."):
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
            except Exception as e: st.error(f"Erreur technique : {e}")
    else:
        st.error("❌ Le service d'analyse n'a pas pu démarrer. Réessayez dans une minute.")

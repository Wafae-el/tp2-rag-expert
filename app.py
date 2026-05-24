import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import requests
import uuid
import time

# --- 1. CONFIGURATION & DESIGN PREMIUM ---
st.set_page_config(page_title="IA Expert Pro", layout="wide", page_icon="💎")

# CSS personnalisé pour un look haut de gamme
st.markdown("""
    <style>
    /* Fond général et police */
    .main { background-color: #f8f9fa; }
    
    /* Sidebar foncée style pro */
    [data-testid="stSidebar"] {
        background-color: #1a1c24;
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    
    /* Bulles de chat stylisées */
    .stChatMessage {
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Bouton "Nouvelle Discussion" en bleu électrique */
    div.stButton > button:first-child {
        background-color: #0078ff;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }

    /* Support pour l'Arabe */
    .rtl-text { direction: rtl; text-align: right; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Sources style badges */
    .source-badge {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FONCTION D'EMBEDDING (CORRIGÉE & ROBUSTE) ---
def get_embeddings_api(text):
    # Utilisation d'une session pour éviter les ConnectionErrors
    session = requests.Session()
    api_url = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    try:
        response = session.post(api_url, headers=headers, json={"inputs": text}, timeout=20)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.warning("⏳ Le moteur d'analyse démarre... réessayez dans 5 secondes.")
            return None
        else:
            st.error(f"Erreur API ({response.status_code})")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
        return None

# --- 3. CONNEXION SERVICES ---
@st.cache_resource
def init_clients():
    q_client = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    g_client = Groq(api_key=st.secrets["G_API"])
    return q_client, g_client

client_q, client_g = init_clients()

# --- 4. GESTION DES SESSIONS ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 5. BARRE LATÉRALE ---
with st.sidebar:
    st.markdown("### 💎 IA EXPERT PRO")
    if st.button("➕ NOUVELLE CONVERSATION", use_container_width=True):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("---")
    st.write("📂 HISTORIQUE")
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        active = "🔹 " if cid == st.session_state.current_chat_id else "  "
        if st.button(f"{active}{data['title']}", key=cid, use_container_width=True):
            st.session_state.current_chat_id = cid
            st.rerun()

# --- 6. ZONE DE CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

st.title(f"💬 {chat_data['title']}")

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question ici..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion":
        chat_data["title"] = (prompt[:25] + '...') if len(prompt) > 25 else prompt
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyse intelligente..."):
        vector = get_embeddings_api(prompt)
        
        if vector is not None:
            try:
                # Recherche Qdrant
                search = client_q.query_points(collection_name="ma_base_expert", query=vector, limit=3).points
                context = "\n".join([f"- {r.payload['text']}" for r in search])
                sources = list(set([r.payload['source'] for r in search]))

                # Groq LLM
                msgs = [{"role": "system", "content": f"Tu es un expert pro. Réponds avec : {context}"}]
                for m in chat_data["messages"][-5:]:
                    msgs.append({"role": m["role"], "content": m["content"]})

                res = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
                ans = res.choices[0].message.content
                
                with st.chat_message("assistant"):
                    st.markdown(ans)
                    # Affichage des sources stylisé
                    src_html = "".join([f'<span class="source-badge">{s}</span>' for s in sources])
                    st.markdown(src_html, unsafe_allow_html=True)
                
                chat_data["messages"].append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e:
                st.error(f"Erreur technique : {e}")

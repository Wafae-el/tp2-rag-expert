import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import uuid

# --- 1. CONFIGURATION VISUELLE (SOFT DARK GITHUB) ---
st.set_page_config(page_title="IA Expert TP2", layout="wide", page_icon="🧠")

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

# --- 2. INITIALISATION (CONFORME AU TP : QDRANT + BGE-M3 LOCAL) ---
@st.cache_resource
def init_all():
    # Connexion Qdrant Cloud
    client_q = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    
    # Activation du modèle BGE-M3 (1024 dim) en mode FastEmbed (Local au serveur)
    # C'est ce que le prof a demandé pour le déploiement
    client_q.set_model("BAAI/bge-m3") 
    
    # Connexion Groq
    client_g = Groq(api_key=st.secrets["G_API"])
    return client_q, client_g

client_q, client_g = init_all()

# --- 3. GESTION DES DISCUSSIONS ---
if "all_chats" not in st.session_state: st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 4. SIDEBAR (HISTORIQUE ET SUPPRESSION) ---
with st.sidebar:
    st.markdown("<h2 style='font-size: 20px;'>Assistant Expert</h2>", unsafe_allow_html=True)
    if st.button("＋ Nouvelle discussion"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("<p style='margin-top: 30px; font-size: 12px;'>HISTORIQUE</p>", unsafe_allow_html=True)
    
    ids_to_del = []
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"• {data['title'][:20]}", key=f"s_{cid}", use_container_width=True):
                st.session_state.current_chat_id = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"d_{cid}"): ids_to_del.append(cid)

    for i in ids_to_del:
        del st.session_state.all_chats[i]
        if i == st.session_state.current_chat_id:
            st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else str(uuid.uuid4())[:8]
        st.rerun()

# --- 5. ZONE DE CHAT ---
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

    with st.spinner("Recherche immédiate..."):
        try:
            # RECHERCHE LOCALE (FastEmbed génère le vecteur 1024 et cherche dans Qdrant)
            search = client_q.query(
                collection_name="ma_base_expert",
                query_text=prompt,
                limit=3
            )

            context = "\n".join([f"- {r.metadata['text']}" for r in search])
            sources = list(set([r.metadata['source'] for r in search]))

            # GÉNÉRATION LLM (GROQ)
            msgs = [{"role": "system", "content": f"Tu es un expert. Réponds avec ce contexte : {context}. Réponds dans la langue de l'utilisateur."}]
            for m in chat_data["messages"][-5:]: msgs.append({"role": m["role"], "content": m["content"]})

            res = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
            ans = res.choices[0].message.content
            
            with st.chat_message("assistant"):
                st.markdown(ans)
                src_html = " ".join([f'<span class="source-badge">{s}</span>' for s in sources])
                st.markdown(src_html, unsafe_allow_html=True)
            
            chat_data["messages"].append({"role": "assistant", "content": ans})
            st.rerun()
        except Exception as e:
            st.error(f"Erreur technique : {e}")

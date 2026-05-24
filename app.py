import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import uuid

# --- 1. CONFIGURATION VISUELLE REPOSANTE ---
st.set_page_config(page_title="IA Expert Pro", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    
    /* Bulles de chat */
    .stChatMessage { background-color: #1e293b !important; border-radius: 10px; border: 1px solid #334155; margin-bottom: 10px; }
    
    /* Bouton Nouvelle Discussion */
    div.stButton > button:first-child {
        background-color: #38bdf8; color: #0f172a; border: none;
        border-radius: 8px; width: 100%; font-weight: bold;
    }
    
    /* Boutons de suppression (poubelle) */
    .del-btn { color: #f87171 !important; font-size: 12px; cursor: pointer; }
    
    .rtl-text { direction: rtl; text-align: right; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALISATION (SANS FASTEXT POUR ÉVITER LE CRASH) ---
@st.cache_resource
def load_resources():
    # Modèle multilingue léger (tient dans 1Go de RAM et répond immédiatement)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    q_client = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    g_client = Groq(api_key=st.secrets["G_API"])
    return model, q_client, g_client

try:
    model_emb, client_q, client_g = load_resources()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")

# --- 3. GESTION DES SESSIONS ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 4. SIDEBAR : HISTORIQUE ET SUPPRESSION ---
with st.sidebar:
    st.markdown("### 🧠 Mes Discussions")
    if st.button("➕ NOUVELLE DISCUSSION"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("---")
    
    # Liste des discussions avec bouton de suppression
    ids_to_delete = []
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        col1, col2 = st.columns([0.8, 0.2])
        
        # Bouton pour sélectionner la discussion
        with col1:
            is_active = (cid == st.session_state.current_chat_id)
            label = f"{'🔹 ' if is_active else ''}{data['title'][:20]}"
            if st.button(label, key=f"select_{cid}", use_container_width=True):
                st.session_state.current_chat_id = cid
                st.rerun()
        
        # Bouton pour supprimer (Poubelle)
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                ids_to_delete.append(cid)

    # Exécution de la suppression
    for i in ids_to_delete:
        del st.session_state.all_chats[i]
        # Si on supprime la discussion actuelle, on bascule sur une autre
        if i == st.session_state.current_chat_id:
            if st.session_state.all_chats:
                st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[0]
            else:
                nid = str(uuid.uuid4())[:8]
                st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
                st.session_state.current_chat_id = nid
        st.rerun()

# --- 5. ZONE DE CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

st.title(chat_data["title"])

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else: st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion":
        chat_data["title"] = (prompt[:25] + '...') if len(prompt) > 25 else prompt
    
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("Réponse immédiate..."):
        try:
            # 1. Embedding local (Immédiat, pas besoin de HF API)
            vector = model_emb.encode(prompt).tolist()

            # 2. Recherche Qdrant
            search = client_q.query_points(collection_name="ma_base_expert", query=vector, limit=3).points
            context = "\n".join([f"- {r.payload['text']}" for r in search])
            sources = list(set([r.payload['source'] for r in search]))

            # 3. Groq (Llama 3.3)
            history = chat_data["messages"][-5:]
            msgs = [{"role": "system", "content": f"Tu es un expert. Réponds avec ce contexte : {context}"}]
            for m in history: msgs.append({"role": m["role"], "content": m["content"]})

            res = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
            ans = res.choices[0].message.content
            
            with st.chat_message("assistant"):
                st.markdown(ans)
                st.caption(f"Sources : {', '.join(sources)}")
            
            chat_data["messages"].append({"role": "assistant", "content": ans})
            st.rerun()
        except Exception as e:
            st.error(f"Erreur technique : {e}")

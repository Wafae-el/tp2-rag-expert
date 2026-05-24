import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import uuid

# --- CONFIGURATION ---
st.set_page_config(page_title="Mon IA - Style ChatGPT", layout="wide", page_icon="💬")

# CSS pour un look moderne et support Arabe
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin: 5px 0; }
    .rtl-text { direction: rtl; text-align: right; font-family: 'Arial'; }
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    .sidebar-btn { text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer('BAAI/bge-m3')
    q_client = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    g_client = Groq(api_key=st.secrets["G_API"])
    return model, q_client, g_client

model_emb, client_q, client_g = load_resources()

# --- GESTION DES SESSIONS ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {} # Format: {id: {"title": "...", "messages": []}}

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())[:8]
    st.session_state.all_chats[new_id] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = new_id

# --- BARRE LATÉRALE (STYLE CHATGPT) ---
with st.sidebar:
    st.title("🤖 Mon IA")
    
    if st.button("➕ Nouvelle discussion", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.all_chats[new_id] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.markdown("### Historique")
    
    # Affichage des titres des discussions
    for chat_id, chat_data in reversed(list(st.session_state.all_chats.items())):
        # Style du bouton (en gras si c'est l'actuel)
        is_active = (chat_id == st.session_state.current_chat_id)
        btn_label = f"💬 {chat_data['title']}"
        
        if st.button(btn_label, key=chat_id, use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()

# --- ZONE DE CHAT PRINCIPALE ---
current_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[current_id]

st.header(chat_data["title"])

# Afficher les messages
for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# --- INTERACTION ---
if prompt := st.chat_input("Envoyez un message..."):
    # 1. Sauvegarder le message utilisateur
    chat_data["messages"].append({"role": "user", "content": prompt})
    
    # 2. Si c'est le premier message, on met à jour le titre de la discussion
    if chat_data["title"] == "Nouvelle discussion":
        # On prend les 30 premiers caractères pour le titre
        new_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
        chat_data["title"] = new_title

    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Processus RAG
    with st.spinner("L'IA réfléchit..."):
        try:
            # Encodage et recherche
            query_vector = model_emb.encode(prompt).tolist()
            search_results = client_q.query_points(
                collection_name="ma_base_expert",
                query=query_vector,
                limit=3
            ).points

            context = "\n".join([f"- {r.payload['text']}" for r in search_results])
            sources = list(set([r.payload['source'] for r in search_results]))

            # Groq avec mémoire locale
            history = chat_data["messages"][-6:] # On donne les 6 derniers messages
            msgs = [{"role": "system", "content": f"Réponds avec ce contexte : {context}"}]
            for m in history:
                msgs.append({"role": m["role"], "content": m["content"]})

            completion = client_g.chat.completions.create(
                messages=msgs,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            answer = completion.choices[0].message.content
            
            # Affichage et sauvegarde
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Sources : {', '.join(sources)}")
            
            chat_data["messages"].append({"role": "assistant", "content": answer})
            st.rerun() # Pour rafraîchir le titre dans la sidebar immédiatement

        except Exception as e:
            st.error(f"Erreur : {e}")

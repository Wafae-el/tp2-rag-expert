import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import time
import uuid # Pour créer des identifiants uniques de discussion

# --- CONFIGURATION ---
st.set_page_config(page_title="IA Expert - Multi-Chats", layout="wide", page_icon="🧠")

# CSS pour un look pro
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; }
    .rtl-text { direction: rtl; text-align: right; }
    .sidebar-chat-btn { margin-bottom: 5px; }
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

# --- GESTION DE L'HISTORIQUE GLOBAL ---
# 'chats' contiendra toutes nos discussions {id: [messages]}
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}
# 'current_chat_id' est la discussion affichée actuellement
if "current_chat_id" not in st.session_state:
    first_id = str(uuid.uuid4())[:8]
    st.session_state.all_chats[first_id] = []
    st.session_state.current_chat_id = first_id

# --- BARRE LATÉRALE (GESTION DES DISCUSSIONS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=60)
    st.title("Mes Discussions")
    
    if st.button("➕ Nouvelle Discussion", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.all_chats[new_id] = []
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.markdown("---")
    # Liste des anciennes discussions
    for chat_id in st.session_state.all_chats.keys():
        label = f"💬 Discussion {chat_id}"
        # On met en gras la discussion actuelle
        if chat_id == st.session_state.current_chat_id:
            label = f"👉 **{label}**"
        
        if st.button(label, key=chat_id, use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()

# --- INTERFACE PRINCIPALE ---
current_id = st.session_state.current_chat_id
st.title(f"🧠 Assistant Intelligent ({current_id})")

# Affichage des messages de la discussion SELECTIONNÉE
for msg in st.session_state.all_chats[current_id]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# --- LOGIQUE DE RÉPONSE ---
if prompt := st.chat_input("Posez votre question..."):
    # Ajouter à la discussion en cours
    st.session_state.all_chats[current_id].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Recherche et réflexion..."):
        try:
            # 1. RAG (Recherche Qdrant)
            query_vector = model_emb.encode(prompt).tolist()
            search_results = client_q.query_points(
                collection_name="ma_base_expert",
                query=query_vector,
                limit=3
            ).points

            context = "\n".join([f"- {r.payload['text']}" for r in search_results])
            sources = list(set([r.payload['source'] for r in search_results]))

            # 2. Construction de la mémoire locale (historique de ce chat précis)
            history = st.session_state.all_chats[current_id][-5:]
            messages_for_groq = [{"role": "system", "content": f"Réponds avec ce contexte : {context}"}]
            for m in history:
                messages_for_groq.append({"role": m["role"], "content": m["content"]})

            # 3. Groq
            completion = client_g.chat.completions.create(
                messages=messages_for_groq,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            response = completion.choices[0].message.content
            
            # 4. Affichage et sauvegarde
            with st.chat_message("assistant"):
                st.markdown(response)
                st.caption(f"Sources : {', '.join(sources)}")

            st.session_state.all_chats[current_id].append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Erreur : {e}")

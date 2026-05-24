import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import time

# --- CONFIGURATION PRO ---
st.set_page_config(page_title="IA Expert RAG + Mémoire", layout="wide", page_icon="🧠")

# CSS pour un look "ChatGPT" et support Arabe
st.markdown("""
    <style>
    .stChatMessage { border-radius: 20px; margin: 10px 0; }
    .rtl-text { direction: rtl; text-align: right; font-family: 'Arial'; }
    .source-tag { background: #f0f2f6; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; color: #555; margin-right: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_all():
    model = SentenceTransformer('BAAI/bge-m3')
    q_client = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    g_client = Groq(api_key=st.secrets["G_API"])
    return model, q_client, g_client

model_emb, client_q, client_g = load_all()

# --- GESTION DE LA MÉMOIRE (DISCUSSION) ---
# On initialise l'historique s'il n'existe pas
if "messages" not in st.session_state:
    st.session_state.messages = []

# BARRE LATÉRALE
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=70)
    st.title("IA Expert")
    st.info(f"Modèle : Llama 3.3 70B\nDocuments indexés : Qdrant Cloud")
    
    if st.button("🗑️ Effacer la discussion"):
        st.session_state.messages = []
        st.rerun()

# --- INTERFACE PRINCIPALE ---
st.title("🧠 Assistant Intelligent avec Mémoire")

# Affichage des anciens messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# --- LOGIQUE DE RÉPONSE ---
if prompt := st.chat_input("Posez votre question..."):
    # 1. On affiche et on sauvegarde la question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Réflexion en cours..."):
        try:
            # 2. RECHERCHE VECTORIELLE (RAG)
            query_vector = model_emb.encode(prompt).tolist()
            search_results = client_q.query_points(
                collection_name="ma_base_expert",
                query=query_vector,
                limit=3
            ).points

            # Préparation du contexte
            context = "\n".join([f"- {r.payload['text']} (Source: {r.payload['source']})" for r in search_results])
            sources = list(set([r.payload['source'] for r in search_results]))

            # 3. MÉMOIRE DE LA CONVERSATION
            # On prend les 5 derniers messages pour donner du contexte à l'IA
            history = st.session_state.messages[-5:] 
            
            # Construction du prompt "Intelligent"
            messages_for_groq = [
                {
                    "role": "system", 
                    "content": f"""Tu es un assistant IA ultra-intelligent. 
                    Utilise le CONTEXTE suivant pour répondre. 
                    Prends aussi en compte l'HISTORIQUE de notre discussion pour comprendre les questions de suivi.
                    Si l'info n'est pas dans le contexte, dis-le.
                    
                    CONTEXTE : {context}"""
                }
            ]
            # On ajoute l'historique
            for m in history:
                messages_for_groq.append({"role": m["role"], "content": m["content"]})

            # 4. GÉNÉRATION VIA GROQ
            start_time = time.time()
            completion = client_g.chat.completions.create(
                messages=messages_for_groq,
                model="llama-3.3-70b-versatile",
                temperature=0.3, # Un peu de créativité mais reste précis
            )
            
            response = completion.choices[0].message.content
            duration = round(time.time() - start_time, 2)

            # 5. AFFICHAGE ET SAUVEGARDE
            with st.chat_message("assistant"):
                st.markdown(response)
                # Affichage propre des sources
                source_html = "".join([f'<span class="source-tag">📄 {s}</span>' for s in sources])
                st.markdown(source_html, unsafe_allow_html=True)
                st.caption(f"Calculé en {duration}s")

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Erreur : {e}")

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuration
st.set_page_config(page_title="IA Expert RAG", layout="centered")

# --- CHARGEMENT DU MODÈLE (On garde le cache ici car c'est lourd) ---
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('BAAI/bge-m3')

model_emb = load_embed_model()

# --- INITIALISATION DES CLIENTS (Sans cache pour éviter ton erreur) ---
def get_clients():
    q_client = QdrantClient(
        url=st.secrets["Q_URL"], 
        api_key=st.secrets["Q_API"]
    )
    g_client = Groq(api_key=st.secrets["G_API"])
    return q_client, g_client

client_q, client_g = get_clients()

st.title("🤖 Mon Assistant IA Expert")

# Historique
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Recherche dans la base vectorielle..."):
        try:
            # 1. Encodage de la question
            query_vector = model_emb.encode(prompt).tolist()

            # 2. Recherche dans Qdrant (Correction de l'erreur search)
            # On utilise le client fraichement créé
            search_results = client_q.search(
                collection_name="ma_base_expert",
                query_vector=query_vector,
                limit=3
            )

            # 3. Préparation du contexte
            context = ""
            sources = []
            for res in search_results:
                context += f"\n- {res.payload['text']}"
                sources.append(res.payload['source'])

            # 4. Génération avec Groq Llama 3.3
            full_prompt = f"Utilise ce contexte pour répondre : {context}\n\nQuestion : {prompt}"
            
            completion = client_g.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model="llama-3.3-70b-versatile",
            )
            
            ans = completion.choices[0].message.content
            final_ans = f"{ans}\n\n**Sources :** {', '.join(list(set(sources)))}"

            with st.chat_message("assistant"):
                st.markdown(final_ans)
            st.session_state.messages.append({"role": "assistant", "content": final_ans})
            
        except Exception as e:
            st.error(f"Détail de l'erreur : {e}")

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# Config de la page
st.set_page_config(page_title="IA Expert RAG", layout="centered")

# --- INITIALISATION ---
@st.cache_resource
def load_all():
    # Chargement des clés depuis les secrets
    q_url = st.secrets["Q_URL"]
    q_api = st.secrets["Q_API"]
    g_api = st.secrets["G_API"]
    
    # Création des clients
    q_client = QdrantClient(url=q_url, api_key=q_api)
    model = SentenceTransformer('BAAI/bge-m3')
    groq_client = Groq(api_key=g_api)
    
    return q_client, model, groq_client

# On récupère les services
client_q, model_emb, client_g = load_all()

st.title("🤖 Mon Assistant IA Expert")

# Historique de chat
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

    with st.spinner("Recherche et génération..."):
        try:
            # 1. Vectoriser
            vector = model_emb.encode(prompt).tolist()

            # 2. Chercher dans Qdrant
            # On vérifie bien le nom de la collection utilisée dans Colab
            search_results = client_q.search(
                collection_name="ma_base_expert",
                query_vector=vector,
                limit=3
            )

            context = ""
            sources = []
            for res in search_results:
                context += f"\n- {res.payload['text']}"
                sources.append(res.payload['source'])

            # 3. Réponse Groq
            full_prompt = f"Réponds à la question en utilisant ce contexte :\n{context}\n\nQuestion : {prompt}"
            
            completion = client_g.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model="llama-3.3-70b-versatile",
            )
            
            ans = completion.choices[0].message.content
            final_ans = f"{ans}\n\n**Sources :** {', '.join(list(set(sources)))}"

            with st.chat_message("assistant"):
                st.markdown(final_ans)
            st.sess

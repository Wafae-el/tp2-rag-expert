import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuration de la page
st.set_page_config(page_title="RAG Multilingue Expert", page_icon="🤖", layout="centered")

st.title("🤖 Mon Assistant IA Expert")
st.markdown("Ce système utilise **Llama 3.3 70B** et une base vectorielle **Qdrant**.")

# --- INITIALISATION ---
@st.cache_resource
def init_services():
    # Connexion à Qdrant
    q_client = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    # Modèle d'embeddings
    embed_model = SentenceTransformer('BAAI/bge-m3')
    # Client Groq
    g_client = Groq(api_key=st.secrets["G_API"])
    return q_client, embed_model, g_client

# On essaie de charger les services, sinon on affiche une erreur propre
try:
    client_q, model_emb, client_g = init_services()
except Exception as e:
    st.error(f"Erreur de configuration : {e}")
    st.stop()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("L'IA réfléchit..."):
        # 1. On transforme la question en vecteur
        query_vector = model_emb.encode(prompt).tolist()

        # 2. On cherche dans Qdrant
        search_results = client_q.search(
            collection_name="ma_base_expert",
            query_vector=query_vector,
            limit=3
        )

        # 3. On prépare le contexte pour le LLM
        context = ""
        sources = []
        for res in search_results:
            context += f"\n- {res.payload['text']}"
            sources.append(res.payload['source'])

        # 4. Génération de la réponse avec Groq
        full_prompt = f"""Tu es un assistant intelligent. Utilise le contexte suivant pour répondre précisément à la question.
        Si l'information n'est pas dans le contexte, dis poliment que tu ne sais pas.
        
        CONTEXTE: {context}
        QUESTION: {prompt}
        """

        chat_completion = client_g.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="llama-3.3-70b-versatile",
        )
        
        response_text = chat_completion.choices[0].message.content
        source_display = f"\n\n**Sources utilisées :** {', '.join(list(set(sources)))}"
        final_answer = response_text + source_display

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

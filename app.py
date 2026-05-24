st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
st.sidebar.title("Paramètres")
st.sidebar.info("Modèle : Llama 3.3 70B\nEmbeddings : BGE-M3\nBase : Qdrant Cloud")
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuration de la page
st.set_page_config(page_title="IA Expert RAG", layout="centered")

# --- CHARGEMENT DU MODÈLE D'EMBEDDING ---
@st.cache_resource
def get_model():
    return SentenceTransformer('BAAI/bge-m3')

model_emb = get_model()

# --- CONNEXION AUX SERVICES ---
# On crée les clients directement pour éviter les erreurs d'attributs
q_url = st.secrets["Q_URL"]
q_api = st.secrets["Q_API"]
g_api = st.secrets["G_API"]

client_q = QdrantClient(url=q_url, api_key=q_api)
client_g = Groq(api_key=g_api)

st.title("🤖 Mon Assistant IA Expert")
st.write("Posez une question sur les documents indexés.")

# Historique
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrée utilisateur
if prompt := st.chat_input("Votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Recherche..."):
        try:
            # 1. Créer le vecteur
            query_vector = model_emb.encode(prompt).tolist()

            # 2. Recherche (On essaie 'query_points' qui est la méthode moderne)
            # Si 'search' ne marche pas, 'query_points' est plus robuste
            search_results = client_q.query_points(
                collection_name="ma_base_expert",
                query=query_vector,
                limit=3
            ).points

            # 3. Préparer le contexte
            context_text = ""
            sources = []
            for res in search_results:
                context_text += f"\n- {res.payload['text']}"
                sources.append(res.payload['source'])

            # 4. Générer la réponse avec Groq
            sys_prompt = f"Tu es un assistant expert. Réponds en utilisant ce contexte : {context_text}"
            
            chat_completion = client_g.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            
            reponse = chat_completion.choices[0].message.content
            full_response = f"{reponse}\n\n**Sources :** {', '.join(list(set(sources)))}"

            with st.chat_message("assistant"):
                st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Erreur technique : {e}")
            st.info("Conseil : Vérifiez que la collection 'ma_base_expert' existe bien dans votre Qdrant Cloud.")

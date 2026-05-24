import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. Configuration de la page (DOIT être la première commande Streamlit)
st.set_page_config(page_title="IA Expert RAG", layout="centered", page_icon="🤖")

# 2. Barre latérale (Sidebar) - Ton ajout extraordinaire
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
st.sidebar.title("Configuration")
st.sidebar.info("""
**Tech Stack :**
- **LLM :** Llama 3.3 70B (Groq)
- **Embeddings :** BGE-M3 (Multilingue)
- **Vecteur DB :** Qdrant Cloud
""")

# Petit bouton bonus pour effacer le chat
if st.sidebar.button("🗑️ Effacer la discussion"):
    st.session_state.messages = []
    st.rerun()

# 3. Chargement des modèles et clients
@st.cache_resource
def get_model():
    return SentenceTransformer('BAAI/bge-m3')

model_emb = get_model()

# Connexion sécurisée via secrets
client_q = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
client_g = Groq(api_key=st.secrets["G_API"])

# 4. Interface principale
st.title("🤖 Mon Assistant IA Expert")
st.write("Posez une question en **Français, Anglais ou Arabe**.")

# Historique de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Logique de Question/Réponse
if prompt := st.chat_input("Votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Recherche dans la base de connaissances..."):
        try:
            # Encodage de la question
            query_vector = model_emb.encode(prompt).tolist()

            # Recherche vectorielle
            search_results = client_q.query_points(
                collection_name="ma_base_expert",
                query=query_vector,
                limit=3
            ).points

            # Construction du contexte
            context_text = ""
            sources = []
            for res in search_results:
                context_text += f"\n- {res.payload['text']}"
                sources.append(res.payload['source'])

            # Génération de la réponse
            sys_prompt = f"""Tu es un assistant expert. Réponds de façon précise en utilisant ce contexte :
            {context_text}
            Si l'info n'est pas là, dis que tu ne sais pas.
            """
            
            chat_completion = client_g.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            
            reponse = chat_completion.choices[0].message.content
            final_ans = f"{reponse}\n\n**Sources consultées :** {', '.join(list(set(sources)))}"

            with st.chat_message("assistant"):
                st.markdown(final_ans)
            st.session_state.messages.append({"role": "assistant", "content": final_ans})

        except Exception as e:
            st.error(f"Erreur : {e}")

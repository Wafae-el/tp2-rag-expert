import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import requests
import uuid
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="IA Expert - Cloud Optimized", layout="wide", page_icon="💬")

# CSS pour le look ChatGPT et support Arabe
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin: 5px 0; }
    .rtl-text { direction: rtl; text-align: right; font-family: 'Arial'; }
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTION D'EMBEDDING VIA API (Consomme 0 RAM) ---
def get_embeddings_api(text):
    api_url = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    # Tentatives en cas de "Modèle en cours de chargement"
    for _ in range(3):
        response = requests.post(api_url, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503: # Modèle en train de démarrer
            time.sleep(10)
            continue
        else:
            return None
    return None

# --- CONNEXION AUX SERVICES ---
client_q = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
client_g = Groq(api_key=st.secrets["G_API"])

# --- GESTION DES DISCUSSIONS (Mémoire) ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())[:8]
    st.session_state.all_chats[new_id] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = new_id

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.title("🤖 Mon IA Expert")
    if st.button("➕ Nouvelle discussion", use_container_width=True, type="primary"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("### Historique")
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        is_active = (cid == st.session_state.current_chat_id)
        label = f"💬 {data['title']}"
        if st.button(label, key=cid, use_container_width=True, type="secondary" if not is_active else "primary"):
            st.session_state.current_chat_id = cid
            st.rerun()

# --- ZONE DE CHAT PRINCIPALE ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]
st.header(chat_data["title"])

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion":
        chat_data["title"] = (prompt[:30] + '...') if len(prompt) > 30 else prompt
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyse et recherche..."):
        # 1. Embeddings via Hugging Face API
        vector = get_embeddings_api(prompt)
        
        if vector is not None:
            try:
                # 2. Recherche Qdrant
                search_results = client_q.query_points(
                    collection_name="ma_base_expert",
                    query=vector,
                    limit=3
                ).points

                context = "\n".join([f"- {r.payload['text']}" for r in search_results])
                sources = list(set([r.payload['source'] for r in search_results]))

                # 3. Groq (Llama 3.3 70B)
                history = chat_data["messages"][-5:]
                msgs = [{"role": "system", "content": f"Tu es un expert. Réponds avec ce contexte : {context}. Réponds dans la langue de la question."}]
                for m in history:
                    msgs.append({"role": m["role"], "content": m["content"]})

                completion = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
                answer = completion.choices[0].message.content
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    st.caption(f"Sources : {', '.join(sources)}")
                
                chat_data["messages"].append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")
        else:
            st.error("Le service d'analyse (Hugging Face) est temporairement indisponible. Réessayez dans un instant.")

import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import requests
import uuid
import time

# --- 1. DESIGN ---
st.set_page_config(page_title="IA Expert Pro", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .rtl-text { direction: rtl; text-align: right; color: #c9d1d9; font-size: 1.1rem; }
    .source-badge {
        font-size: 11px; color: #58a6ff; background-color: rgba(56, 139, 253, 0.1);
        padding: 3px 10px; border-radius: 12px; margin-right: 8px; border: 1px solid rgba(56, 139, 253, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FONCTION D'EMBEDDING SÉCURISÉE ---
def get_embeddings(text):
    api_url = "https://api-inference.huggingface.co/models/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=20)
        data = response.json()
        
        if response.status_code == 200:
            # Si c'est une liste de listes, on prend la première
            if isinstance(data, list) and isinstance(data[0], list):
                return data[0]
            return data
        elif response.status_code == 503:
            return "LOADING"
        else:
            return None
    except:
        return None

# --- 3. INITIALISATION ---
@st.cache_resource
def init_clients():
    q = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    g = Groq(api_key=st.secrets["G_API"])
    return q, g

client_q, client_g = init_clients()

if "all_chats" not in st.session_state: st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Assistant Expert")
    if st.button("＋ Nouvelle discussion", use_container_width=True):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("---")
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        if st.button(f"• {data['title'][:20]}", key=f"s_{cid}", use_container_width=True):
            st.session_state.current_chat_id = cid
            st.rerun()

# --- 5. ZONE DE CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

# Affichage des messages
for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# Nouvelle question
if prompt := st.chat_input("Posez votre question..."):
    # Affichage immédiat de la question
    chat_data["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Titre auto
    if chat_data["title"] == "Nouvelle discussion":
        chat_data["title"] = prompt[:30]

    # Traitement
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        
        # 1. Obtenir l'embedding
        status_placeholder.status("🔍 Analyse de la question (Hugging Face)...")
        vector = None
        for i in range(10): # Max 10 tentatives
            res = get_embeddings(prompt)
            if res == "LOADING":
                status_placeholder.warning(f"⏳ Le moteur d'analyse démarre... (Tentative {i+1}/10)")
                time.sleep(8)
            elif res is not None:
                vector = res
                break
            else:
                time.sleep(2)
        
        if vector:
            try:
                status_placeholder.status("📂 Recherche dans les documents (Qdrant)...")
                search = client_q.query_points(collection_name="ma_base_expert", query=vector, limit=3).points
                
                context = "\n".join([f"- {r.payload['text']}" for r in search])
                sources = list(set([r.payload['source'] for r in search]))

                status_placeholder.status("✍️ Rédaction de la réponse (Groq Llama 3.3)...")
                msgs = [{"role": "system", "content": f"Réponds avec ce contexte : {context}"}]
                for m in chat_data["messages"][-5:]:
                    msgs.append({"role": m["role"], "content": m["content"]})

                completion = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
                answer = completion.choices[0].message.content
                
                # Affichage final
                status_placeholder.empty() # On enlève les messages de chargement
                st.markdown(answer)
                src_html = " ".join([f'<span class="source-badge">{s}</span>' for s in sources])
                st.markdown(src_html, unsafe_allow_html=True)
                
                # Sauvegarde
                chat_data["messages"].append({"role": "assistant", "content": answer})
            except Exception as e:
                status_placeholder.error(f"Erreur technique : {e}")
        else:
            status_placeholder.error("❌ Le moteur d'analyse n'a pas pu répondre. Réessayez dans 30 secondes.")

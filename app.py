import streamlit as st
from qdrant_client import QdrantClient
from groq import Groq
import uuid

# --- 1. DESIGN CONFORTABLE (DARK GREY) ---
st.set_page_config(page_title="IA Expert - Instant", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stChatMessage { background-color: #0d1117 !important; border-bottom: 1px solid #21262d !important; }
    div.stButton > button:first-child {
        background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d;
        border-radius: 20px; width: 100%; transition: 0.2s;
    }
    .rtl-text { direction: rtl; text-align: right; font-size: 1.1rem; }
    .source-badge {
        font-size: 10px; color: #58a6ff; background-color: rgba(56, 139, 253, 0.1);
        padding: 2px 8px; border-radius: 10px; border: 1px solid rgba(56, 139, 253, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALISATION INSTANTANÉE (FastEmbed) ---
@st.cache_resource
def init_all():
    # On utilise FastEmbed directement intégré à Qdrant
    # Le modèle BGE-M3 est téléchargé UNE SEULE FOIS au démarrage de l'app
    client_q = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    client_q.set_model("BAAI/bge-m3") # Version optimisée et légère
    
    client_g = Groq(api_key=st.secrets["G_API"])
    return client_q, client_g

client_q, client_g = init_all()

# --- 3. GESTION DES SESSIONS ---
if "all_chats" not in st.session_state: st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='font-size: 18px;'>⚡ Assistant Instantané</h2>", unsafe_allow_html=True)
    if st.button("＋ Nouvelle discussion"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("<p style='margin-top: 20px; font-size: 11px; color: #8b949e;'>HISTORIQUE</p>", unsafe_allow_html=True)
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        active = "color: #58a6ff;" if cid == st.session_state.current_chat_id else ""
        if st.button(f"• {data['title'][:20]}", key=f"chat_{cid}"):
            st.session_state.current_chat_id = cid
            st.rerun()

# --- 5. ZONE DE CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else: st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion": chat_data["title"] = prompt[:25]
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("Réponse immédiate..."):
        try:
            # ÉTAPE CLÉ : client_q.query(...) fait l'embedding ET la recherche en une seule fois !
            # C'est local, c'est compressé, ça ne crash pas la RAM.
            search = client_q.query(
                collection_name="ma_base_expert",
                query_text=prompt,
                limit=3
            )

            context = "\n".join([f"- {r.metadata['text']}" for r in search])
            sources = list(set([r.metadata['source'] for r in search]))

            # Groq (Llama 3.3 70B) pour la rapidité de génération
            msgs = [{"role": "system", "content": f"Tu es un expert. Réponds avec ce contexte : {context}"}]
            for m in chat_data["messages"][-5:]: msgs.append({"role": m["role"], "content": m["content"]})

            res = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
            ans = res.choices[0].message.content
            
            with st.chat_message("assistant"):
                st.markdown(ans)
                src_html = "".join([f'<span class="source-badge">{s}</span>' for s in sources])
                st.markdown(src_html, unsafe_allow_html=True)
            
            chat_data["messages"].append({"role": "assistant", "content": ans})
            st.rerun()
            
        except Exception as e:
            st.error(f"Détail : {e}")

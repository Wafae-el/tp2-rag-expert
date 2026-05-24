import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import uuid

# --- 1. DESIGN SOFT DARK ---
st.set_page_config(page_title="IA Expert Pro", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #8b949e !important; }
    div.stButton > button:first-child {
        background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d;
        border-radius: 20px; width: 100%; transition: 0.2s;
    }
    .stChatMessage { background-color: #0d1117 !important; border-bottom: 1px solid #21262d !important; }
    .rtl-text { direction: rtl; text-align: right; color: #c9d1d9; font-size: 1.1rem; }
    .source-badge {
        font-size: 11px; color: #58a6ff; background-color: rgba(56, 139, 253, 0.1);
        padding: 3px 10px; border-radius: 12px; margin-right: 8px; border: 1px solid rgba(56, 139, 253, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALISATION (MODÈLE LÉGER 384 DIM) ---
@st.cache_resource
def load_resources():
    # Ce modèle est multilingue (Fr, Ar, En) et tient dans la RAM de Streamlit
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    q_client = QdrantClient(url=st.secrets["Q_URL"], api_key=st.secrets["Q_API"])
    g_client = Groq(api_key=st.secrets["G_API"])
    return model, q_client, g_client

model_emb, client_q, client_g = load_resources()

# --- 3. GESTION DES SESSIONS ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}
if "current_chat_id" not in st.session_state:
    nid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
    st.session_state.current_chat_id = nid

# --- 4. SIDEBAR : HISTORIQUE ET SUPPRESSION ---
with st.sidebar:
    st.markdown("<h2 style='font-size: 20px;'>Assistant Expert</h2>", unsafe_allow_html=True)
    if st.button("＋ Nouvelle discussion"):
        nid = str(uuid.uuid4())[:8]
        st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
        st.session_state.current_chat_id = nid
        st.rerun()
    
    st.markdown("<p style='margin-top: 30px; font-size: 12px;'>HISTORIQUE</p>", unsafe_allow_html=True)
    
    ids_to_delete = []
    for cid, data in reversed(list(st.session_state.all_chats.items())):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            active = "color: #58a6ff; font-weight: bold;" if cid == st.session_state.current_chat_id else ""
            if st.button(f"• {data['title'][:20]}", key=f"s_{cid}", use_container_width=True):
                st.session_state.current_chat_id = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"d_{cid}"):
                ids_to_delete.append(cid)

    for i in ids_to_delete:
        del st.session_state.all_chats[i]
        if i == st.session_state.current_chat_id:
            if st.session_state.all_chats:
                st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[0]
            else:
                nid = str(uuid.uuid4())[:8]
                st.session_state.all_chats[nid] = {"title": "Nouvelle discussion", "messages": []}
                st.session_state.current_chat_id = nid
        st.rerun()

# --- 5. ZONE DE CHAT ---
cur_id = st.session_state.current_chat_id
chat_data = st.session_state.all_chats[cur_id]

st.markdown(f"<h4 style='color: #8b949e;'>{chat_data['title']}</h4>", unsafe_allow_html=True)

for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        if any("\u0600" <= c <= "\u06FF" for c in msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else: st.markdown(msg["content"])

if prompt := st.chat_input("Écrivez votre message..."):
    chat_data["messages"].append({"role": "user", "content": prompt})
    if chat_data["title"] == "Nouvelle discussion":
        chat_data["title"] = prompt[:30]
    
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner("Analyse immédiate..."):
        try:
            # 1. Embedding local (384 dimensions)
            vector = model_emb.encode(prompt).tolist()

            # 2. Recherche Qdrant
            search = client_q.query_points(collection_name="ma_base_expert", query=vector, limit=3).points
            context = "\n".join([f"- {r.payload['text']}" for r in search])
            sources = list(set([r.payload['source'] for r in search]))

            # 3. Groq LLM
            msgs = [{"role": "system", "content": f"Tu es un expert. Réponds avec ce contexte : {context}"}]
            for m in chat_data["messages"][-5:]: msgs.append({"role": m["role"], "content": m["content"]})

            res = client_g.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
            ans = res.choices[0].message.content
            
            with st.chat_message("assistant"):
                st.markdown(ans)
                src_html = " ".join([f'<span class="source-badge">{s}</span>' for s in sources])
                st.markdown(src_html, unsafe_allow_html=True)
            
            chat_data["messages"].append({"role": "assistant", "content": ans})
            st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")

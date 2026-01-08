import streamlit as st
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import utils 
import streamlit.components.v1 as components

st.set_page_config(page_title="Mushroom Identification", page_icon="🍄", layout="wide")

st.markdown("""
    <style>
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .stApp {
        background-color: #1b261d;
        color: #e0e7e1;
    }
    
    div[data-testid="stChatMessageContainer"] {
        scroll-behavior: smooth;
    }

    .header-style {
        text-align: center;
        color: #94ad9a !important;
        font-family: 'Segoe UI', Tahoma, sans-serif;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 20px;
        animation: fadeIn 1.5s ease-in;
    }

    [data-testid="stImage"] {
        animation: fadeIn 1.2s ease-out;
    }

    .user-bubble {
        background-color: #2e5a39; 
        color: #ffffff; 
        padding: 15px; 
        border-radius: 18px 18px 2px 18px; 
        margin-bottom: 15px; 
        border: 1px solid #3d7a4c;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        animation: slideUp 0.4s ease-out;
    }

    .assistant-bubble {
        background-color: #38453a; 
        color: #e8ede9; 
        padding: 15px; 
        border-radius: 18px 18px 18px 2px; 
        margin-bottom: 15px; 
        border: 1px solid #4a5c4d;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        animation: slideUp 0.4s ease-out;
    }

    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
    }

    .stChatInputContainer {
        background-color: #2d3e30 !important;
        border-radius: 10px !important;
    }
    section[data-testid="stFileUploadDropzone"] {
        background-color: #2d3e30 !important;
        border: 1px dashed #4e6b54 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_cnn_model():
    return load_model("./mushroomCNNclasifier.h5")

model_loaded = get_cnn_model()
file_paths = ["Knowledge_base/mushroom_guide.json", 
              "Knowledge_base/mushroom_world.json", 
              "Knowledge_base/wikipedia.json"
              ]

knowledge_base = utils.prepare_knowledge_base(file_paths)

embeddings = np.load("Knowledge_base/embeddings.npy")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "active_file" not in st.session_state:
    st.session_state["active_file"] = None

upload_ui, chat_ui = st.columns([1, 2])


with upload_ui:
    st.markdown('<h2 class="header-style">🍄 YOUR MUSHROOM 🍄</h2>', unsafe_allow_html=True)
    file = st.file_uploader("*Upload a mushroom photo:*", type=["png", "jpg", "jpeg"])

    if file is not None:
        if st.session_state["active_file"] != file.name:
            st.session_state["active_file"] = file.name
            st.session_state["messages"] = []
            st.session_state["agent"] = utils.MushroomRAGAgent(
                api_key=utils.get_api_key(),
                knowledge_base=knowledge_base,
                model_name="gemini-2.5-flash-lite",
                embedding_model="all-MiniLM-L6-v2",
                embeddings=embeddings
            )
            with open("temp_img.jpg", "wb") as f:
                f.write(file.getbuffer())
            
            preds = utils.get_predictions("temp_img.jpg", top_k=3)
            with st.spinner("Analyzing your mushroom..."):
                initial_info = st.session_state["agent"].initialize_from_predictions(preds)
                st.session_state["messages"].append({"role": "assistant", "content": initial_info})
            st.rerun()

        st.image(Image.open(file), use_container_width=True)
    else:
        st.session_state["active_file"] = None
        st.session_state["agent"] = None
        st.session_state["messages"] = []


with chat_ui:
    st.markdown('<h2 class="header-style">🍄 MUSHROOM IDENTIFICATOR 🍄</h2>', unsafe_allow_html=True)
    chat_container = st.container(height=650)

    if st.session_state["agent"]:
        with chat_container:
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    c1, c2 = st.columns([1, 4])
                    with c2:
                        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            

        if prompt := st.chat_input("Ask a follow-up question..."):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.rerun()

        if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
            with chat_container:
                c1, c2 = st.columns([4, 1])
                with c1:
                    with st.spinner("Searching for information..."):
                        last_prompt = st.session_state["messages"][-1]["content"]
                        response = st.session_state["agent"].send_message(last_prompt)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                        st.rerun()
    else:
        st.info("🌲 Please upload a mushroom photo to start the analysis.")
"""Main Streamlit application for Mushroom Identification."""

import streamlit as st
from PIL import Image
import numpy as np

from config import (
    API_KEY, KNOWLEDGE_BASE_FILES, EMBEDDINGS_PATH,
    GEMINI_MODEL_NAME, EMBEDDING_MODEL_NAME, TEMP_IMAGE_PATH
)
from styles import CUSTOM_CSS
from knowledge_base import prepare_knowledge_base
from predictor import MushroomPredictor
from RAG_Agent import MushroomRAGAgent

# Page configuration
st.set_page_config(
    page_title="Mushroom Identification",
    page_icon="🍄",
    layout="wide"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the CNN predictor model."""
    return MushroomPredictor()


@st.cache_resource
def load_knowledge_base():
    """Load knowledge base and embeddings."""
    kb = prepare_knowledge_base(KNOWLEDGE_BASE_FILES)
    emb = np.load(EMBEDDINGS_PATH)
    return kb, emb


# Initialize resources
predictor = load_predictor()
knowledge_base, embeddings = load_knowledge_base()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "active_file" not in st.session_state:
    st.session_state["active_file"] = None

# Create UI layout
upload_ui, chat_ui = st.columns([1, 2])

# Upload section
with upload_ui:
    st.markdown(
        '<h2 class="header-style">🍄 YOUR MUSHROOM 🍄</h2>',
        unsafe_allow_html=True
    )
    file = st.file_uploader(
        "*Upload a mushroom photo:*",
        type=["png", "jpg", "jpeg"]
    )

    if file is not None:
        # Check if new file uploaded
        if st.session_state["active_file"] != file.name:
            st.session_state["active_file"] = file.name
            st.session_state["messages"] = []

            # Initialize RAG agent
            st.session_state["agent"] = MushroomRAGAgent(
                api_key=API_KEY,
                knowledge_base=knowledge_base,
                model_name=GEMINI_MODEL_NAME,
                embedding_model=EMBEDDING_MODEL_NAME,
                embeddings=embeddings
            )

            # Save temporary image
            with open(TEMP_IMAGE_PATH, "wb") as f:
                f.write(file.getbuffer())

            # Get predictions
            predictions = predictor.predict(TEMP_IMAGE_PATH, top_k=3)

            # Generate initial identification
            with st.spinner("Analyzing your mushroom..."):
                initial_info = st.session_state["agent"].initialize_from_predictions(
                    predictions
                )
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": initial_info
                })
            st.rerun()

        # Display image
        st.image(Image.open(file), width='stretch')
    else:
        # Reset state when no file
        st.session_state["active_file"] = None
        st.session_state["agent"] = None
        st.session_state["messages"] = []

# Chat section
with chat_ui:
    st.markdown(
        '<h2 class="header-style">🍄 MUSHROOM IDENTIFICATOR 🍄</h2>',
        unsafe_allow_html=True
    )
    chat_container = st.container(height=650)

    if st.session_state["agent"]:
        # Display messages
        with chat_container:
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    c1, c2 = st.columns([1, 4])
                    with c2:
                        st.markdown(
                            f'<div class="user-bubble">{msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(
                            f'<div class="assistant-bubble">{msg["content"]}</div>',
                            unsafe_allow_html=True
                        )

        # Chat input
        if prompt := st.chat_input("Ask a follow-up question..."):
            st.session_state["messages"].append({
                "role": "user",
                "content": prompt
            })
            st.rerun()

        # Process last user message
        if (st.session_state["messages"] and
                st.session_state["messages"][-1]["role"] == "user"):
            with chat_container:
                c1, c2 = st.columns([4, 1])
                with c1:
                    with st.spinner("Searching for information..."):
                        last_prompt = st.session_state["messages"][-1]["content"]
                        response = st.session_state["agent"].send_message(last_prompt)
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": response
                        })
                        st.rerun()
    else:
        st.info("🌲 Please upload a mushroom photo to start the analysis.")
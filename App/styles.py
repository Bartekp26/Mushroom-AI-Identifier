"""CSS styles for the Mushroom Identification Streamlit app."""

CUSTOM_CSS = """
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
"""
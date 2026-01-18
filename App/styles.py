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
        background-color: #190b28;
        color: #edfdee;
    }

    div[data-testid="stChatMessageContainer"] {
        scroll-behavior: smooth;
    }

    .header-style {
        text-align: center;
        color: #edfdee !important;
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
        background-color: #23967f; 
        color: #edfdee; 
        padding: 15px; 
        border-radius: 18px 18px 2px 18px; 
        margin-bottom: 15px; 
        border: 2px solid #1a6e5d;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        animation: slideUp 0.4s ease-out;
    }

    .assistant-bubble {
        background-color: #bd4089; 
        color: #edfdee; 
        padding: 15px; 
        border-radius: 18px 18px 18px 2px; 
        margin-bottom: 15px; 
        border: 2px solid #8a2f64; /* Ciemniejszy odcień Fuchsia Plum */
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        animation: slideUp 0.4s ease-out;
    }

    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
    }

    </style>
"""
import streamlit as st
import google.generativeai as genai
from bot_logic import process_file, build_prompt

# ---- Gemini API Config ----
api_key = st.secrets["GEMINI"]["AIzaSyBhkBva0qymAKkPvx8LA6LL50rS2nSuxz4"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini 2.5 Pro")

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
        body {
            background-color: white;
        }
        .top-bar {
            background-color: #007BFF;
            color: white;
            padding: 1rem 2rem;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            border-radius: 0 0 8px 8px;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Blue Top Bar ----
st.markdown('<div class="top-bar">Retrieval-Based Chatbot with Local Semantic Search</div>', unsafe_allow_html=True)

# ---- Session State ----
if "history" not in st.session_state:
    st.session_state.history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.vectorizer = None
    st.session_state.chunk_vectors = None

# ---- Sidebar File Upload UI ----
st.sidebar.header("📄 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
if uploaded_file:
    file_type = uploaded_file.type.split("/")[-1]
    chunks, vectorizer, chunk_vectors = process_file(uploaded_file, file_type)
    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.chunk_vectors = chunk_vectors
    st.sidebar.success("✅ File processed!")

# ---- Chat Input UI ----
st.subheader("💬 Ask Queries")
user_query = st.text_input("You:", placeholder="Ask about courses, hostel, fees, etc...")

if st.button("Ask") and user_query and st.session_state.vectorizer:
    prompt = build_prompt(
        user_query,
        st.session_state.history,
        st.session_state.vectorizer,
        st.session_state.chunk_vectors,
        st.session_state.chunks
    )
    try:
        response = model.generate_content(prompt)
        bot_reply = response.text.strip()
        st.session_state.history.append((user_query, bot_reply))
        st.success("✅ Response generated")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---- Display Chat History ----
if st.session_state.history:
    st.subheader("🗨️ Conversation")
    for user, bot in reversed(st.session_state.history):
        st.markdown(f"**You:** {user}")
        st.markdown(f"**Bot:** {bot}")
        st.markdown("---")

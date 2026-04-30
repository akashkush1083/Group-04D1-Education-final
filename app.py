"""
EduFusion AI - Smart Learning Assistant
Modern UI with Image Generation
"""

import os
import base64
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agentic_system import create_explanation
import requests

load_dotenv()

# Page config
st.set_page_config(page_title="LearnWithMe", page_icon="🚀", layout="wide")

# Custom CSS - Modern Dark Theme
st.markdown("""
<style>
.stApp { 
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    min-height: 100vh;
}

.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(139, 92, 246, 0.2);
    padding: 1rem 2rem;
    margin-bottom: 2rem;
}

.app-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.app-subtitle {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 0.25rem;
    font-weight: 400;
}

.card {
    background: rgba(30, 27, 75, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.card:hover {
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.1);
}

.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    padding: 0.875rem 2rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

.stTextInput > div > input,
.stTextArea > div > textarea {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(139, 92, 246, 0.3);
    color: #f1f5f9;
    padding: 1rem;
    border-radius: 12px;
    font-size: 1rem;
    transition: all 0.3s ease;
}

.stTextInput > div > input:focus,
.stTextArea > div > textarea:focus {
    border-color: #8b5cf6;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
}

.output-container {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
    .app-title { font-size: 1.5rem; }
    .card { padding: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# Initialize LLM
@st.cache_resource
def load_qa_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1000
    )

# Session state
if "image_result" not in st.session_state:
    st.session_state.image_result = None
if "unified_answer" not in st.session_state:
    st.session_state.unified_answer = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Smart input detection
def detect_input_type(user_input):
    if not user_input or not user_input.strip():
        return "empty"
    input_lower = user_input.lower()
    image_keywords = ['image', 'diagram', 'visual', 'picture', 'illustration', 'draw', 'create image', 'generate image']
    if any(word in input_lower for word in image_keywords):
        return "image"
    if len(user_input.split()) > 10 or any(word in input_lower for word in ['explain', 'describe', 'create', 'write']):
        return "prompt"
    return "topic"

# Sidebar with Search History
with st.sidebar:
    st.markdown('<div style="padding: 1rem 0; border-bottom: 1px solid rgba(139, 92, 246, 0.2); margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin: 0;">📜 Search History</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.conversation_history:
        for i, (question, answer) in enumerate(reversed(st.session_state.conversation_history), 1):
            with st.expander(f"Q{i}: {question[:50]}...", expanded=False):
                st.markdown(f'<div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">Question:</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="color: #f1f5f9; font-size: 0.9rem; margin-bottom: 1rem;">{question}</div>', unsafe_allow_html=True)
                if st.button(f"👁️ View Full Answer", key=f"view_answer_{i}", use_container_width=True):
                    st.session_state.unified_answer = answer
                    st.session_state.image_result = None
                    st.rerun()
                st.markdown(f'<div style="color: #94a3b8; font-size: 0.85rem; margin: 0.5rem 0;">Preview:</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="color: #64748b; font-size: 0.85rem;">{answer[:300]}{"..." if len(answer) > 300 else ""}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #64748b; font-size: 0.9rem; padding: 1rem; text-align: center;">No search history yet</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 1px; background: rgba(139, 92, 246, 0.2); margin: 1rem 0;"></div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear History", use_container_width=True, key="sidebar_clear"):
        st.session_state.conversation_history = []
        st.session_state.unified_answer = None
        st.session_state.image_result = None
        st.rerun()

# Navbar
st.markdown("""
<div class="navbar">
    <div>
        <h1 class="app-title">🚀 EduFusion AI — Smart Learning Assistant</h1>
        <p class="app-subtitle">Your unified AI for learning and image generation</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== MAIN SECTION ====================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧠 AI Learning Assistant</div>', unsafe_allow_html=True)

# Show conversation memory indicator
if st.session_state.conversation_history:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div style="padding: 0.75rem; background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; margin-bottom: 1rem;">
            <span style="color: #8b5cf6; font-weight: 600;">💬 Memory:</span> 
            <span style="color: #f1f5f9;">{len(st.session_state.conversation_history)} exchanges remembered</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_memory"):
            st.session_state.conversation_history = []
            st.rerun()

# Input
user_prompt = st.text_area(
    "Ask anything or request an explanation...",
    placeholder="e.g., 'Explain photosynthesis', 'What is machine learning?', 'Create a diagram of DNA'",
    height=100,
    key="unified_prompt_input",
    label_visibility="collapsed"
)

generate_btn = st.button("🚀 Generate", use_container_width=True, type="primary", key="unified_generate")

# Output
if st.session_state.get('unified_answer') or st.session_state.image_result:
    st.markdown('<div class="output-container">', unsafe_allow_html=True)

    if st.session_state.get('unified_answer'):
        st.markdown('<div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">📝 Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #f1f5f9; line-height: 1.75;">{st.session_state.unified_answer}</div>', unsafe_allow_html=True)

    if st.session_state.image_result:
        st.markdown('<div style="color: #94a3b8; font-size: 0.9rem; margin: 1.5rem 0 1rem 0;">🎨 Generated Image</div>', unsafe_allow_html=True)
        try:
            image_data = base64.b64decode(st.session_state.image_result)
            st.image(image_data, caption="AI Generated Image", use_column_width=True)
            st.download_button(
                label="📥 Download Image",
                data=image_data,
                file_name=f"edufusion_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== PROCESSING LOGIC ====================
if generate_btn and user_prompt and user_prompt.strip():
    qa_llm = load_qa_llm()
    input_type = detect_input_type(user_prompt.strip())

    with st.spinner("🧠 Processing..."):
        try:
            st.session_state.unified_answer = None
            st.session_state.image_result = None

            # Build conversation context
            conversation_context = ""
            if st.session_state.conversation_history:
                conversation_context = "\n\nPrevious conversation:\n"
                for prev_q, prev_a in st.session_state.conversation_history[-3:]:
                    conversation_context += f"\nUser: {prev_q}\nAssistant: {prev_a[:500]}...\n"

            # Generate text answer
            if input_type in ["topic", "prompt"]:
                enhanced_prompt = user_prompt
                if conversation_context:
                    enhanced_prompt = f"""
                    {user_prompt}

                    Context to consider:
                    {conversation_context}

                    Instructions:
                    - Reference previous conversation when relevant
                    - Provide a comprehensive, well-structured answer
                    """
                try:
                    explanation = create_explanation(enhanced_prompt)
                    st.session_state.unified_answer = str(explanation)
                except Exception as e:
                    print(f"Agentic error: {e}")
                    prompt = f"Answer this question: {user_prompt}\n{conversation_context}\nProvide a comprehensive, well-structured answer in Markdown format."
                    response = qa_llm.invoke(prompt)
                    st.session_state.unified_answer = response.content

            # Generate image if requested
            if input_type == "image" or any(word in user_prompt.lower() for word in ['image', 'diagram', 'visual', 'picture', 'illustration', 'draw']):
                encoded_prompt = requests.utils.quote(user_prompt.strip())
                image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
                response = requests.get(image_url, timeout=30)
                if response.status_code == 200:
                    st.session_state.image_result = base64.b64encode(response.content).decode('utf-8')
                    st.success("✅ Image generated!")
                else:
                    st.error("❌ Image generation failed")

            # Save to conversation history
            if st.session_state.unified_answer:
                st.session_state.conversation_history.append((user_prompt, st.session_state.unified_answer))
                if len(st.session_state.conversation_history) > 20:
                    st.session_state.conversation_history = st.session_state.conversation_history[-20:]

            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(139,92,246,0.2), transparent); margin:3rem 0;"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;font-size:0.85rem;color:#64748b;padding-bottom:1rem;letter-spacing:0.05em;">🚀 EduFusion AI &nbsp;·&nbsp; Smart Learning Assistant &nbsp;·&nbsp; AI-Powered Education</p>', unsafe_allow_html=True)
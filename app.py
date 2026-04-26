"""
EduFusion AI - Smart Learning Assistant
Modern UI with Image Generation + PDF Assistant
"""

import os
import json
import re
import base64
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from vector_db import vector_db
from agentic_system import create_study_guide, create_explanation

load_dotenv()

# Page config
st.set_page_config(page_title="EduFusion AI", page_icon="🚀", layout="wide")

# Custom CSS - Modern Dark Theme
st.markdown("""
<style>
.stApp { 
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    min-height: 100vh;
}

/* Navbar Styles */
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

/* Card Styles */
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

/* Button Styles */
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

/* Input Styles */
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

/* File Upload Styles */
.stFileUploader {
    background: rgba(15, 23, 42, 0.8);
    border: 2px dashed rgba(139, 92, 246, 0.4);
    border-radius: 12px;
    padding: 2rem;
}

/* Output Container */
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

/* Loading Animation */
.loading-spinner {
    border: 3px solid rgba(139, 92, 246, 0.2);
    border-top: 3px solid #8b5cf6;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 1rem auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive */
@media (max-width: 768px) {
    .app-title { font-size: 1.5rem; }
    .card { padding: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# Initialize system
@st.cache_resource
def load_qa_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1000  # Reduced from 1500 to stay under rate limit
    )

# Session state
if "show_pdf_modal" not in st.session_state:
    st.session_state.show_pdf_modal = False
if "image_result" not in st.session_state:
    st.session_state.image_result = None
if "image_explanation" not in st.session_state:
    st.session_state.image_explanation = None
if "pdf_answer" not in st.session_state:
    st.session_state.pdf_answer = None
if "unified_answer" not in st.session_state:
    st.session_state.unified_answer = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Modern Navbar
st.markdown("""
<div class="navbar">
    <div>
        <h1 class="app-title">🚀 EduFusion AI — Smart Learning Assistant</h1>
        <p class="app-subtitle">Your unified AI for learning, PDFs, and image generation</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Smart input detection
def detect_input_type(user_input):
    if not user_input or not user_input.strip():
        return "empty"
    
    input_lower = user_input.lower()
    
    # Check image requests
    image_keywords = ['image', 'diagram', 'visual', 'picture', 'illustration', 'draw', 'create image', 'generate image']
    if any(word in input_lower for word in image_keywords):
        return "image"
    
    # Check detailed prompts
    if len(user_input.split()) > 10 or any(word in input_lower for word in ['explain', 'describe', 'create', 'write']):
        return "prompt"
    
    return "topic"

# ==================== UNIFIED SECTION: AI Assistant with PDF Support ====================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧠 AI Learning Assistant</div>', unsafe_allow_html=True)

# PDF Upload
with st.expander("📤 Upload PDF Document (Optional)", expanded=False):
    uploaded_file = st.file_uploader("Drag and drop your PDF here", type=['pdf'], key="pdf_upload")
    
    if uploaded_file:
        st.success(f"✅ Selected: {uploaded_file.name}")
        
        if st.button("📤 Upload PDF", type="primary", key="upload_pdf_btn"):
            with st.spinner("Processing PDF..."):
                try:
                    result = vector_db.add_pdf(uploaded_file, uploaded_file.name)
                    if result['status'] == 'success':
                        st.success(f"✅ PDF processed! {result['chunks_added']} chunks added.")
                        st.session_state.pdf_uploaded = True
                        st.session_state.current_pdf = uploaded_file.name
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result['message']}")
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")

# Show PDF status
if st.session_state.get('pdf_uploaded', False):
    st.markdown(f"""
    <div style="padding: 1rem; background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 8px; margin-bottom: 1rem;">
        <span style="color: #22c55e; font-weight: 600;">✅ PDF Ready:</span> 
        <span style="color: #f1f5f9;">{st.session_state.get('current_pdf', 'Unknown')}</span>
        <span style="color: #94a3b8; margin-left: 0.5rem;">· Answers will combine PDF content + AI knowledge</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="padding: 1rem; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; margin-bottom: 1rem;">
        <span style="color: #3b82f6; font-weight: 600;">💡 AI Mode:</span> 
        <span style="color: #f1f5f9;">Using AI knowledge base (upload PDF for enhanced answers)</span>
    </div>
    """, unsafe_allow_html=True)

# Show conversation history and clear button
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

# Unified Input
user_prompt = st.text_area(
    "Ask anything or request an explanation...",
    placeholder="e.g., 'Explain photosynthesis', 'What is the first topic in my PDF?', 'Create a diagram of DNA'",
    height=100,
    key="unified_prompt_input",
    label_visibility="collapsed"
)

generate_btn = st.button("🚀 Generate", use_container_width=True, type="primary", key="unified_generate")

# Output container
if st.session_state.get('unified_answer') or st.session_state.image_result or st.session_state.image_explanation:
    st.markdown('<div class="output-container">', unsafe_allow_html=True)
    
    # Show explanation/answer
    if st.session_state.get('unified_answer'):
        st.markdown('<div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">📝 Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #f1f5f9; line-height: 1.75;">{st.session_state.unified_answer}</div>', unsafe_allow_html=True)
    
    # Show image if generated
    if st.session_state.image_result:
        st.markdown('<div style="color: #94a3b8; font-size: 0.9rem; margin: 1.5rem 0 1rem 0;">🎨 Generated Image</div>', unsafe_allow_html=True)
        try:
            image_data = base64.b64decode(st.session_state.image_result)
            st.image(image_data, caption="AI Generated Image", use_column_width=True)
            
            if st.session_state.get('image_url'):
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

# Process Unified Section
if generate_btn and user_prompt and user_prompt.strip():
    qa_llm = load_qa_llm()
    input_type = detect_input_type(user_prompt.strip())
    
    with st.spinner("🧠 Processing..."):
        try:
            # Clear previous outputs
            st.session_state.unified_answer = None
            st.session_state.image_explanation = None
            st.session_state.image_result = None
            
            # Check if PDF is uploaded and search it
            pdf_context = ""
            pdf_found = False
            
            if st.session_state.get('pdf_uploaded', False):
                search_results = vector_db.search(user_prompt.strip(), k=3)  # Reduced from 5 to 3
                # Check if search_results is a list (not an error dict)
                if isinstance(search_results, list) and search_results and len(search_results) > 0:
                    # Filter for PDF-type results only (not explanations)
                    pdf_results = [r for r in search_results if r.get('metadata', {}).get('type') != 'explanation']
                    if pdf_results:
                        pdf_context = "\n\n".join([result['content'][:800] for result in pdf_results])  # Truncate each chunk
                        pdf_found = True
            
            # Always search vector DB for previous explanations (even without PDF)
            explanation_context = ""
            explanation_found = False
            search_results = vector_db.search(user_prompt.strip(), k=2)  # Reduced from 3 to 2
            # Check if search_results is a list (not an error dict)
            if isinstance(search_results, list) and search_results and len(search_results) > 0:
                # Filter for explanation-type results only
                explanation_results = [r for r in search_results if r.get('metadata', {}).get('type') == 'explanation']
                if explanation_results:
                    explanation_context = "\n\n".join([result['content'][:600] for result in explanation_results])  # Truncate
                    explanation_found = True
            
            # Build conversation history context
            conversation_context = ""
            if st.session_state.conversation_history:
                conversation_context = "\n\nPrevious conversation:\n"
                for i, (prev_q, prev_a) in enumerate(st.session_state.conversation_history[-3:], 1):  # Keep last 3 exchanges to reduce tokens
                    conversation_context += f"\nUser: {prev_q}\nAssistant: {prev_a[:500]}...\n"  # Truncate answers
            
            # Generate explanation/answer using agentic system
            if input_type in ["topic", "prompt"]:
                # Build enhanced prompt with context for agentic system
                enhanced_prompt = user_prompt
                
                if pdf_found or explanation_found or conversation_context:
                    enhanced_prompt = f"""
                    {user_prompt}
                    
                    Context to consider:
                    {conversation_context if conversation_context else ""}
                    
                    {f"PDF Context:\n{pdf_context}" if pdf_found else ""}
                    
                    {f"Previous Explanations:\n{explanation_context}" if explanation_found else ""}
                    
                    Instructions:
                    - Use the provided context (conversation, PDF, previous explanations) to enhance your answer
                    - Reference information from previous conversations when relevant
                    - If PDF content is available, use it as a primary source
                    - Provide a comprehensive, well-structured answer
                    """
                
                try:
                    # Use agentic system (research + writing agents)
                    explanation = create_explanation(enhanced_prompt)
                    st.session_state.unified_answer = str(explanation)
                except Exception as e:
                    # Fallback to direct LLM if agentic fails
                    print(f"Agentic error: {e}")
                    prompt = f"""
                    Answer this question: {user_prompt}
                    
                    {conversation_context if conversation_context else ""}
                    
                    {f"PDF Context:\n{pdf_context}" if pdf_found else ""}
                    
                    {f"Previous Explanations:\n{explanation_context}" if explanation_found else ""}
                    
                    Provide a comprehensive, well-structured answer in Markdown format.
                    """
                    response = qa_llm.invoke(prompt)
                    st.session_state.unified_answer = response.content
            
            # Generate image if requested
            if input_type == "image" or any(word in user_prompt.lower() for word in ['image', 'diagram', 'visual', 'picture', 'illustration', 'draw']):
                import requests
                
                encoded_prompt = requests.utils.quote(user_prompt.strip())
                image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
                
                response = requests.get(image_url, timeout=30)
                
                if response.status_code == 200:
                    base64_image = base64.b64encode(response.content).decode('utf-8')
                    st.session_state.image_result = base64_image
                    st.session_state.image_url = image_url
                    st.success("✅ Image generated!")
                else:
                    st.error("❌ Image generation failed")
            
            # Save to conversation history
            if st.session_state.unified_answer:
                st.session_state.conversation_history.append((user_prompt, st.session_state.unified_answer))
                # Keep only last 20 exchanges to prevent memory issues
                if len(st.session_state.conversation_history) > 20:
                    st.session_state.conversation_history = st.session_state.conversation_history[-20:]
                
                # Save explanation to vector DB for future reference (only for topic/prompt types, not images)
                if input_type in ["topic", "prompt"]:
                    try:
                        result = vector_db.add_explanation(user_prompt, st.session_state.unified_answer)
                        if result.get('status') != 'success':
                            print(f"Error saving to vector DB: {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        print(f"Error saving to vector DB: {e}")
            
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(139,92,246,0.2), transparent); margin:3rem 0;"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;font-size:0.85rem;color:#64748b;padding-bottom:1rem;letter-spacing:0.05em;">🚀 EduFusion AI &nbsp;·&nbsp; Smart Learning Assistant &nbsp;·&nbsp; AI-Powered Education</p>', unsafe_allow_html=True)

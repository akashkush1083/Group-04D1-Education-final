# EduVisual AI - Unified Education System

A smart multimodal education system that understands topics, prompts, images, and documents.

## Features

- 📚 **Topic Study Guides** - Generate comprehensive study materials
- ✨ **Custom Prompts** - Get detailed explanations on any topic
- 🎨 **AI Image Generation** - Create educational visuals
- 📄 **PDF Q&A** - Upload and query documents

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open in browser:** `http://localhost:8501`

## How to Use

### Main Search Section
- **Topics**: "Photosynthesis", "Machine Learning"
- **Prompts**: "Explain quantum computing in detail"
- **Images**: "Create a diagram of photosynthesis"

### PDF Section
1. Click 📄 button to upload PDF
2. Use "Search PDFs..." input to ask questions
3. Get answers from your documents

## File Structure

```
d:\Combine\
├── app.py                 # Main application
├── vector_db.py          # PDF storage and search
├── requirements.txt      # Python dependencies
├── .env                  # API keys
├── .streamlit/          # Streamlit config
└── outputs/             # Generated content
```

## Configuration

Update `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_huggingface_key_here
```

## Dependencies

- streamlit - Web interface
- langchain-groq - LLM integration
- sentence-transformers - Text embeddings
- faiss-cpu - Vector search
- PyPDF2 - PDF processing
- requests - API calls
- python-dotenv - Environment variables

## PDF Workflow

1. Upload PDF via 📄 button
2. PDF is processed and stored in vector database
3. Ask questions in PDF section
4. Get answers with sources

## Support

For issues or questions, check the vector database status in the PDF section.

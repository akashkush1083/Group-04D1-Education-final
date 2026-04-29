"""
Vector Database for PDF Document Storage and Retrieval
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses FAISS and HuggingFace Inference API for semantic search
(No local torch/sentence-transformers — keeps Docker image small)
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np
import faiss
import requests
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_EMBEDDING_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


class VectorDatabase:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
        self.db_path = "outputs/vector_db"

        os.makedirs(self.db_path, exist_ok=True)
        self._load_database()

    # ──────────────────────────────────────────────
    # Embedding via HuggingFace Inference API
    # ──────────────────────────────────────────────

    def _get_embeddings(self, texts: list[str], retries: int = 3) -> np.ndarray:
        """
        Call HF Inference API to get embeddings.
        Retries automatically if the model is loading (503).
        """
        if not HF_API_KEY:
            raise ValueError(
                "HF_API_KEY is not set. Add it to your .env file.\n"
                "Get a free key at https://huggingface.co/settings/tokens"
            )

        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        for attempt in range(retries):
            response = requests.post(HF_EMBEDDING_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                embeddings = response.json()
                # HF returns list[list[float]] for batch inputs
                return np.array(embeddings, dtype="float32")

            elif response.status_code == 503:
                # Model is loading on HF side — wait and retry
                wait_time = (attempt + 1) * 10
                print(f"HF model loading, retrying in {wait_time}s...")
                time.sleep(wait_time)

            else:
                raise Exception(
                    f"HF API error {response.status_code}: {response.text}"
                )

        raise Exception("HF embedding API failed after max retries.")

    # ──────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────

    def _load_database(self):
        """Load existing vector database from disk."""
        try:
            if os.path.exists(f"{self.db_path}/index.faiss"):
                self.index = faiss.read_index(f"{self.db_path}/index.faiss")
            if os.path.exists(f"{self.db_path}/documents.pkl"):
                with open(f"{self.db_path}/documents.pkl", "rb") as f:
                    self.documents = pickle.load(f)
            if os.path.exists(f"{self.db_path}/metadata.pkl"):
                with open(f"{self.db_path}/metadata.pkl", "rb") as f:
                    self.metadata = pickle.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            self.metadata = []

    def _save_database(self):
        """Save vector database to disk."""
        try:
            faiss.write_index(self.index, f"{self.db_path}/index.faiss")
            with open(f"{self.db_path}/documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            with open(f"{self.db_path}/metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving database: {e}")

    # ──────────────────────────────────────────────
    # PDF Handling
    # ──────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text content from PDF file."""
        try:
            reader = PdfReader(pdf_file)
            return "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {e}")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks for better embedding coverage."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk) > 50:  # Skip tiny chunks
                chunks.append(chunk)
        return chunks

    def add_pdf(self, pdf_file, filename=None) -> dict:
        """Add PDF to vector database."""
        try:
            text = self.extract_text_from_pdf(pdf_file)
            chunks = self.chunk_text(text)

            if not chunks:
                raise Exception("No text content found in PDF")

            embeddings = self._get_embeddings(chunks)
            self.index.add(embeddings)

            fname = filename or pdf_file.name
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.metadata.append({
                    "filename": fname,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

            self._save_database()
            return {"status": "success", "chunks_added": len(chunks), "filename": fname}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ──────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict] | dict:
        """Search for relevant document chunks by semantic similarity."""
        try:
            if self.index.ntotal == 0:
                return []

            query_embedding = self._get_embeddings([query])  # shape (1, 384)
            distances, indices = self.index.search(query_embedding, k)

            results = []
            for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if 0 <= idx < len(self.documents):
                    results.append({
                        "rank": rank + 1,
                        "content": self.documents[idx],
                        "metadata": self.metadata[idx],
                        "similarity": float(1 / (1 + dist)),
                    })
            return results

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ──────────────────────────────────────────────
    # Document Management
    # ──────────────────────────────────────────────

    def get_document_list(self) -> list[dict]:
        """Return a list of all unique documents in the database."""
        docs: dict[str, dict] = {}
        for meta in self.metadata:
            fname = meta["filename"]
            if fname not in docs:
                docs[fname] = {"filename": fname, "chunks": 0, "added_date": meta["added_date"]}
            docs[fname]["chunks"] += 1
        return list(docs.values())

    def delete_document(self, filename: str) -> dict:
        """Remove all chunks belonging to a specific document and rebuild the index."""
        try:
            to_delete = {i for i, m in enumerate(self.metadata) if m["filename"] == filename}

            if not to_delete:
                return {"status": "error", "message": "Document not found"}

            kept_docs = [d for i, d in enumerate(self.documents) if i not in to_delete]
            kept_meta = [m for i, m in enumerate(self.metadata) if i not in to_delete]

            # Re-embed only the kept chunks (batched for efficiency)
            self.index = faiss.IndexFlatL2(self.dimension)
            if kept_docs:
                embeddings = self._get_embeddings(kept_docs)
                self.index.add(embeddings)

            self.documents = kept_docs
            self.metadata = kept_meta
            self._save_database()

            return {"status": "success", "deleted_chunks": len(to_delete)}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_stats(self) -> dict:
        """Return high-level database statistics."""
        return {
            "total_documents": len({m["filename"] for m in self.metadata}),
            "total_chunks": len(self.documents),
            "index_size": self.index.ntotal,
        }

    def add_explanation(self, topic: str, explanation: str) -> dict:
        """Persist a generated explanation so it can be retrieved later."""
        try:
            text = f"Topic: {topic}\n\nExplanation: {explanation}"
            chunks = self.chunk_text(text)

            if not chunks:
                raise Exception("No content to add")

            embeddings = self._get_embeddings(chunks)
            self.index.add(embeddings)

            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.metadata.append({
                    "filename": f"explanation_{topic[:50]}",
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "explanation",
                    "topic": topic,
                })

            self._save_database()
            return {"status": "success", "chunks_added": len(chunks), "topic": topic}

        except Exception as e:
            return {"status": "error", "message": str(e)}


# Global instance
vector_db = VectorDatabase()

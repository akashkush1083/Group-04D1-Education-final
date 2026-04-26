"""
Vector Database for PDF Document Storage and Retrieval
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses FAISS and Sentence Transformers for semantic search
"""

import os
import json
import pickle
from datetime import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

class VectorDatabase:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
        self.db_path = "outputs/vector_db"
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Load existing database if available
        self._load_database()
    
    def _load_database(self):
        """Load existing vector database from disk"""
        try:
            if os.path.exists(f"{self.db_path}/index.faiss"):
                self.index = faiss.read_index(f"{self.db_path}/index.faiss")
            
            if os.path.exists(f"{self.db_path}/documents.pkl"):
                with open(f"{self.db_path}/documents.pkl", 'rb') as f:
                    self.documents = pickle.load(f)
            
            if os.path.exists(f"{self.db_path}/metadata.pkl"):
                with open(f"{self.db_path}/metadata.pkl", 'rb') as f:
                    self.metadata = pickle.load(f)
                    
        except Exception as e:
            print(f"Error loading database: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            self.metadata = []
    
    def _save_database(self):
        """Save vector database to disk"""
        try:
            faiss.write_index(self.index, f"{self.db_path}/index.faiss")
            with open(f"{self.db_path}/documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            with open(f"{self.db_path}/metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from PDF file"""
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {e}")
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into chunks for better embedding"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 50:  # Only keep meaningful chunks
                chunks.append(chunk)
        
        return chunks
    
    def add_pdf(self, pdf_file, filename=None):
        """Add PDF to vector database"""
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            
            # Create chunks
            chunks = self.chunk_text(text)
            
            if not chunks:
                raise Exception("No text content found in PDF")
            
            # Generate embeddings
            embeddings = self.model.encode(chunks)
            
            # Add to FAISS index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store documents and metadata
            start_idx = len(self.documents)
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.metadata.append({
                    'filename': filename or pdf_file.name,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'added_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Save database
            self._save_database()
            
            return {
                'status': 'success',
                'chunks_added': len(chunks),
                'filename': filename or pdf_file.name
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def search(self, query, k=5):
        """Search for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Search in FAISS
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), k
            )
            
            # Retrieve results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    results.append({
                        'rank': i + 1,
                        'content': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'similarity': 1 / (1 + dist)  # Convert distance to similarity
                    })
            
            return results
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_document_list(self):
        """Get list of all documents in database"""
        documents = {}
        for meta in self.metadata:
            filename = meta['filename']
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'chunks': 0,
                    'added_date': meta['added_date']
                }
            documents[filename]['chunks'] += 1
        
        return list(documents.values())
    
    def delete_document(self, filename):
        """Delete all chunks of a specific document"""
        try:
            # Find indices to delete
            indices_to_delete = []
            for i, meta in enumerate(self.metadata):
                if meta['filename'] == filename:
                    indices_to_delete.append(i)
            
            if not indices_to_delete:
                return {'status': 'error', 'message': 'Document not found'}
            
            # Rebuild database without deleted document
            new_documents = []
            new_metadata = []
            new_embeddings = []
            
            for i in range(len(self.documents)):
                if i not in indices_to_delete:
                    new_documents.append(self.documents[i])
                    new_metadata.append(self.metadata[i])
                    # Re-encode embedding for this document
                    embedding = self.model.encode([self.documents[i]])[0]
                    new_embeddings.append(embedding)
            
            # Update database
            self.documents = new_documents
            self.metadata = new_metadata
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            if new_embeddings:
                self.index.add(np.array(new_embeddings).astype('float32'))
            
            # Save updated database
            self._save_database()
            
            return {'status': 'success', 'deleted_chunks': len(indices_to_delete)}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_stats(self):
        """Get database statistics"""
        return {
            'total_documents': len(set(meta['filename'] for meta in self.metadata)),
            'total_chunks': len(self.documents),
            'index_size': self.index.ntotal
        }
    
    def add_explanation(self, topic, explanation):
        """Add an explanation to the vector database for future reference"""
        try:
            # Create a document from the explanation
            text = f"Topic: {topic}\n\nExplanation: {explanation}"
            
            # Chunk the explanation
            chunks = self.chunk_text(text)
            
            if not chunks:
                raise Exception("No content to add")
            
            # Generate embeddings
            embeddings = self.model.encode(chunks)
            
            # Add to FAISS index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store documents and metadata
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.metadata.append({
                    'filename': f"explanation_{topic[:50]}",
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'added_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'type': 'explanation',
                    'topic': topic
                })
            
            # Save database
            self._save_database()
            
            return {
                'status': 'success',
                'chunks_added': len(chunks),
                'topic': topic
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# Global instance
vector_db = VectorDatabase()

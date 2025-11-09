
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """
    SINGLE RESPONSIBILITY: Convert text to vectors
    Used for both query embeddings and specialist embeddings
    """
    
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
    
    def create_embedding(self, text):
        """
        Convert text to 384-dimensional vector
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: 384-dimensional vector as list
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def create_embeddings_batch(self, texts):
        """
        Convert multiple texts to vectors (more efficient)
        
        Args:
            texts (list): List of strings
            
        Returns:
            list: List of 384-dimensional vectors
        """
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]

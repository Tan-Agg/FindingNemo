
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import SIMILARITY_THRESHOLD

class MemoryBank:
    """
    SINGLE RESPONSIBILITY: Store specialists and search for matches
    Does NOT create embeddings (uses EmbeddingService for that)
    """
    
    def __init__(self, bank_file='data/memory_bank.json'):
        self.bank_file = bank_file
        self.specialists = []
        self.load()
    
    def load(self):
        """Load specialists from disk"""
        try:
            with open(self.bank_file, 'r') as f:
                data = json.load(f)
                self.specialists = data.get('specialists', [])
            print(f"✅ Loaded {len(self.specialists)} specialists")
        except FileNotFoundError:
            print(f"⚠️  Memory bank not found, creating new")
            self.specialists = []
            self.save()
    
    def save(self):
        """Save specialists to disk"""
        data = {"specialists": self.specialists}
        with open(self.bank_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def search(self, query_embedding):
        """
        Find matching specialist using semantic similarity
        
        Args:
            query_embedding (list): 384-dim vector from EmbeddingService
            
        Returns:
            dict or None: {specialist: dict, similarity: float} if match found
        """
        if not self.specialists:
            return None
        
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        best_match = None
        best_similarity = 0
        
        for specialist in self.specialists:
            spec_vec = np.array(specialist['embedding']).reshape(1, -1)
            similarity = cosine_similarity(query_vec, spec_vec)[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = specialist
        
        if best_similarity >= SIMILARITY_THRESHOLD:
            return {
                "specialist": best_match,
                "similarity": float(best_similarity)
            }
        
        return None
    
    def add_specialist(self, intent_label, description, endpoint, embedding, metadata=None):
        """
        Add specialist with PRE-COMPUTED embedding
        
        Args:
            intent_label (str): Intent label
            description (str): Description
            endpoint (str): API endpoint
            embedding (list): Pre-computed 384-dim vector
            metadata (dict): Optional metadata
        """
        # Check duplicate
        for spec in self.specialists:
            if spec['intent_label'] == intent_label:
                print(f"⚠️  Specialist '{intent_label}' already exists")
                return False
        
        specialist = {
            "intent_label": intent_label,
            "description": description,
            "endpoint": endpoint,
            "embedding": embedding,
            "metadata": metadata or {}
        }
        
        self.specialists.append(specialist)
        self.save()
        print(f"✅ Added specialist: {intent_label}")
        return True
    
    def get_all_specialists(self):
        """Return list of all specialists"""
        return self.specialists

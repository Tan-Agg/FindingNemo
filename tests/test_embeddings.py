import sys
sys.path.append('..')

from core.embeddings import EmbeddingService

def test_embeddings():
    service = EmbeddingService()
    
    text = "Generate SQL queries for database operations"
    
    print("\n" + "="*60)
    print("TESTING EMBEDDINGS")
    print("="*60 + "\n")
    
    print(f"Text: {text}")
    embedding = service.create_embedding(text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Vector type: {type(embedding)}")

if __name__ == "__main__":
    test_embeddings()
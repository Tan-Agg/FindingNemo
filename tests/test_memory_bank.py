import sys
sys.path.append('..')

from core.embeddings import EmbeddingService
from core.memory_bank import MemoryBank

def test_memory_bank():
    # Initialize services
    embedding_service = EmbeddingService()
    memory_bank = MemoryBank()
    
    print("\n" + "="*60)
    print("TESTING MEMORY BANK")
    print("="*60 + "\n")
    
    # Add a specialist (if not exists)
    if len(memory_bank.specialists) == 0:
        print("Adding demo specialist...")
        description = "Generate SQL queries for database operations and data retrieval"
        embedding = embedding_service.create_embedding(description)
        
        memory_bank.add_specialist(
            intent_label="sql_generation",
            description=description,
            endpoint="https://api.nvidia.com/v1/models/sql-specialist",
            embedding=embedding,
            metadata={"accuracy": 0.95}
        )
    
    # Test search
    print("\nTesting search...")
    query_text = "Write SQL query for top customers"
    query_embedding = embedding_service.create_embedding(query_text)
    
    result = memory_bank.search(query_embedding)
    
    if result:
        print(f"✅ Found: {result['specialist']['intent_label']}")
        print(f"   Similarity: {result['similarity']:.3f}")
    else:
        print("❌ No match found")

if __name__ == "__main__":
    test_memory_bank()
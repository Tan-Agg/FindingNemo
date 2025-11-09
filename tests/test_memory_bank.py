# import sys
# sys.path.append('..')

# from core.embeddings import EmbeddingService
# from core.memory_bank import MemoryBank
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def test_memory_bank():
#     # Initialize services
#     embedding_service = EmbeddingService()
#     memory_bank = MemoryBank()
    
#     print("\n" + "="*70)
#     print("TESTING MEMORY BANK")
#     print("="*70 + "\n")
    
#     # Show current state
#     print("📊 Current Memory Bank State:")
#     specialists = memory_bank.get_all_specialists()
#     if specialists:
#         for i, spec in enumerate(specialists, 1):
#             print(f"  {i}. {spec['intent_label']}: {spec['description'][:50]}...")
#     else:
#         print("  (empty)")
#     print()
    
#     # Add a specialist (if not exists)
#     if len(memory_bank.specialists) == 0:
#         print("➕ Adding demo specialist...")
#         description = "Generate SQL queries for database operations and data retrieval"
#         embedding = embedding_service.create_embedding(description)
        
#         memory_bank.add_specialist(
#             intent_label="sql_generation",
#             description=description,
#             endpoint="https://api.nvidia.com/v1/models/sql-specialist",
#             embedding=embedding,
#             metadata={"accuracy": 0.95}
#         )
#         print()
    
#     # Test queries with different similarity levels
#     test_cases = [
#         ("Write SQL query for top customers", "Should match SQL specialist"),
#         ("Generate SQL for database", "Should match SQL specialist (high similarity)"),
#         ("Create database query for analytics", "Should match SQL specialist"),
#         ("Review Python code for bugs", "Should NOT match (different topic)"),
#         ("Translate this to Spanish", "Should NOT match (different topic)")
#     ]
    
#     print("="*70)
#     print("TESTING SEMANTIC SEARCH")
#     print("="*70 + "\n")
    
#     # Get specialist embedding for manual comparison
#     specialist = memory_bank.specialists[0]
#     specialist_emb = np.array(specialist['embedding'])
    
#     for query_text, expected in test_cases:
#         print(f"Query: '{query_text}'")
#         print(f"Expected: {expected}")
        
#         # Create query embedding
#         query_embedding = embedding_service.create_embedding(query_text)
#         query_emb = np.array(query_embedding)
        
#         # Calculate similarity manually (for display)
#         manual_similarity = cosine_similarity(
#             query_emb.reshape(1, -1), 
#             specialist_emb.reshape(1, -1)
#         )[0][0]
        
#         print(f"Calculated Similarity: {manual_similarity:.3f}")
#         print(f"Threshold: {memory_bank.SIMILARITY_THRESHOLD if hasattr(memory_bank, 'SIMILARITY_THRESHOLD') else 'from config'}")
        
#         # Search using memory bank
#         result = memory_bank.search(query_embedding)
        
#         if result:
#             print(f"✅ MATCH FOUND!")
#             print(f"   Specialist: {result['specialist']['intent_label']}")
#             print(f"   Similarity: {result['similarity']:.3f}")
#         else:
#             print(f"❌ NO MATCH (below threshold)")
        
#         print("-" * 70 + "\n")
    
#     # Summary
#     print("="*70)
#     print("TEST SUMMARY")
#     print("="*70)
#     print(f"Total specialists in memory bank: {len(memory_bank.specialists)}")
#     print(f"Total test queries: {len(test_cases)}")
#     print("\nKey Observations:")
#     print("  - SQL-related queries should have similarity > 0.70")
#     print("  - Non-SQL queries should have similarity < 0.50")
#     print("  - Threshold determines what gets matched")

# if __name__ == "__main__":
#     test_memory_bank()


from core.embeddings import EmbeddingService
from core.memory_bank import MemoryBank
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import SIMILARITY_THRESHOLD

def test_japan_specialist():
    # Initialize services
    embedding_service = EmbeddingService()
    memory_bank = MemoryBank()
    
    print("\n" + "="*70)
    print("TESTING JAPAN SPECIALIST")
    print("="*70 + "\n")
    
    # Show current state
    print("📊 Current Memory Bank State:")
    specialists = memory_bank.get_all_specialists()
    if specialists:
        for i, spec in enumerate(specialists, 1):
            print(f"  {i}. {spec['intent_label']}: {spec['description'][:50]}...")
    else:
        print("  (empty)")
        return
    print()
    
    # Find Japan specialist
    japan_specialist = None
    for spec in specialists:
        if spec['intent_label'] == 'japan_travel':
            japan_specialist = spec
            break
    
    if not japan_specialist:
        print("❌ No Japan specialist found with intent_label='japan_travel'!")
        return
    
    print(f"🎯 Testing against: {japan_specialist['intent_label']}")
    print(f"   Description: {japan_specialist['description']}")
    print(f"   Threshold: {SIMILARITY_THRESHOLD}\n")
    
    # Test queries
    test_cases = [
        # Should MATCH (Japan-related)
        ("Plan a trip to Tokyo", "Should match - Japan travel"),
        ("Tell me about Japanese culture", "Should match - Japan culture"),
        ("Best sushi restaurants in Kyoto", "Should match - Japan food"),
        ("What to do in Osaka", "Should match - Japan travel"),
        ("Japanese language tips", "Should match - Japan language"),
        ("Visiting Mount Fuji recommendations", "Should match - Japan landmarks"),
        
        # Should NOT MATCH (other topics)
        ("Book a flight to Paris", "Should NOT match - France travel"),
        ("Italian restaurants near me", "Should NOT match - Italy food"),
        ("How to learn Spanish", "Should NOT match - Spanish language"),
        ("Generate SQL query", "Should NOT match - completely different"),
        ("Python code review", "Should NOT match - programming"),
    ]
    
    print("="*70)
    print("TESTING SEMANTIC SEARCH")
    print("="*70 + "\n")
    
    # Get specialist embedding
    specialist_emb = np.array(japan_specialist['embedding'])
    
    matches = 0
    non_matches = 0
    
    for query_text, expected in test_cases:
        print(f"Query: '{query_text}'")
        print(f"Expected: {expected}")
        
        # Create query embedding
        query_embedding = embedding_service.create_embedding(query_text)
        query_emb = np.array(query_embedding)
        
        # Calculate similarity
        manual_similarity = cosine_similarity(
            query_emb.reshape(1, -1), 
            specialist_emb.reshape(1, -1)
        )[0][0]
        
        print(f"Similarity Score: {manual_similarity:.4f}")
        print(f"Threshold: {SIMILARITY_THRESHOLD}")
        
        # Search using memory bank
        result = memory_bank.search(query_embedding)
        
        if result and result['specialist']['intent_label'] == 'japan_travel':
            print(f"✅ MATCH FOUND!")
            print(f"   Specialist: {result['specialist']['intent_label']}")
            print(f"   Similarity: {result['similarity']:.4f}")
            matches += 1
        else:
            if result:
                print(f"⚠️  MATCHED DIFFERENT SPECIALIST: {result['specialist']['intent_label']}")
            else:
                print(f"❌ NO MATCH (below threshold)")
            non_matches += 1
        
        print("-" * 70 + "\n")
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total specialists in memory bank: {len(memory_bank.specialists)}")
    print(f"Testing specialist: {japan_specialist['intent_label']}")
    print(f"Total test queries: {len(test_cases)}")
    print(f"Matches found: {matches}")
    print(f"Non-matches: {non_matches}")
    print(f"\nSimilarity Threshold: {SIMILARITY_THRESHOLD}")
    print("\n💡 Analysis:")
    print("  - Japan-related queries should match japan_travel specialist")
    print("  - Non-Japan queries should not match")
    print(f"  - Success rate: {matches}/{len([t for t in test_cases if 'Should match' in t[1]])} Japan queries matched")

if __name__ == "__main__":
    test_japan_specialist()
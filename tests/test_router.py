import sys
sys.path.append('..')

from core.router import IntentRouter

def test_router():
    router = IntentRouter()
    
    test_queries = [
        "Write SQL to find top customers",
        "Review this Python code",
        "Summarize this document"
    ]
    
    print("\n" + "="*60)
    print("TESTING ROUTER")
    print("="*60 + "\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        intent = router.generate_intent(query)
        print(f"Label: {intent['intent_label']}")
        print(f"Description: {intent['description']}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    test_router()
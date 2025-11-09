import sys
sys.path.append('..')

from coreOLD.router import IntentRouter

def test_router():
    router = IntentRouter()
    
    test_queries = [
        "Write SQL to find top customers",
        "Review this Python code",
        "Summarize this document",
        "I want to know about the Japan culture",
        "What would Anne Frank do if she came to country Japan today?",
        "how to recognise flying airplane from ground"
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
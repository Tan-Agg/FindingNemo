import sys
sys.path.append('..')

from main import NemotronMetaAgent

def test_integration():
    """
    Test the complete end-to-end pipeline
    """
    # Initialize
    agent = NemotronMetaAgent()
    
    # Show initial status
    agent.print_status()
    
    # Test single query
    print("\n" + "="*60)
    print("TESTING SINGLE QUERY")
    print("="*60)
    
    result = agent.process_query("Write SQL to find top customers")
    
    print("\n📊 RESULT:")
    print(f"Answer received: {len(result['answer'])} characters")
    print(f"Routed to: {result['metadata']['routed_to']}")
    print(f"Latency: {result['metadata']['latency']}s")
    print(f"Intent: {result['metadata']['intent_label']}")
    
    # Show updated status
    print("\n" + "="*60)
    print("UPDATED STATUS")
    print("="*60)
    agent.print_status()

if __name__ == "__main__":
    test_integration()
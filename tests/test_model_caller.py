import sys
sys.path.append('..')

from core.model_caller import ModelCaller

def test_model_caller():
    caller = ModelCaller()
    
    print("\n" + "="*60)
    print("TESTING MODEL CALLER")
    print("="*60 + "\n")
    
    # Test generalist
    print("Testing Generalist (253B)...")
    query = "What is 2+2?"
    result = caller.call_generalist(query)
    
    if result['error']:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Answer: {result['answer']}")
        print(f"   Tokens: {result['tokens_used']}")
        print(f"   Model: {result['model']}")
    
    print("\n" + "-"*60 + "\n")
    
    # Test specialist (will fail if endpoint doesn't exist, that's ok)
    print("Testing Specialist endpoint (may fail if not deployed)...")
    fake_endpoint = "nvidia/llama-3.1-nemotron-8b-instruct"  # Use base 8B as test
    result = caller.call_specialist(fake_endpoint, "Say hello")
    
    if result['error']:
        print(f"⚠️  Expected - specialist not deployed yet")
        print(f"   Error: {result['error']}")
    else:
        print(f"✅ Answer: {result['answer']}")
        print(f"   Model: {result['model']}")

if __name__ == "__main__":
    test_model_caller()
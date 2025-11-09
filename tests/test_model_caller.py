"""
Test suite for ModelCaller
Tests both generalist and specialist API calls
"""

import sys
import time
from core.model_caller import ModelCaller

def print_separator(title):
    """Print a nice separator"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_generalist_simple():
    """Test 1: Simple generalist call"""
    print_separator("TEST 1: Simple Generalist Call")
    
    caller = ModelCaller()
    
    test_queries = [
        "What is 2+2?",
        "Write a haiku about AI",
        "Explain quantum computing in one sentence"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test 1.{i}] Query: {query}")
        print("-" * 70)
        
        result = caller.call_generalist(query, max_tokens=100)
        
        print(f"\n✅ Result:")
        print(f"  Answer: {result['answer'][:200] if result['answer'] else 'None'}...")
        print(f"  Tokens: {result['tokens_used']}")
        print(f"  Error: {result['error']}")
        
        assert result['answer'] is not None, "Answer should not be None"
        assert isinstance(result['answer'], str), "Answer should be a string"
        assert result['tokens_used'] > 0, "Should use some tokens"
        
        print(f"\n✅ Test 1.{i} PASSED")
        time.sleep(1)  # Be nice to API

def test_generalist_sql():
    """Test 2: SQL generation (typical use case)"""
    print_separator("TEST 2: SQL Query Generation")
    
    caller = ModelCaller()
    
    query = "Write SQL to find top 10 customers by revenue"
    print(f"Query: {query}")
    print("-" * 70)
    
    result = caller.call_generalist(query, max_tokens=500)
    
    print(f"\n✅ Result:")
    print(f"  Answer: {result['answer']}")
    print(f"  Tokens: {result['tokens_used']}")
    print(f"  Model: {result['model']}")
    print(f"  Error: {result['error']}")
    
    assert result['answer'] is not None, "Should get SQL answer"
    assert "SELECT" in result['answer'].upper() or "sql" in result['answer'].lower(), "Should contain SQL"
    
    print(f"\n✅ TEST 2 PASSED")

def test_generalist_long_response():
    """Test 3: Longer response"""
    print_separator("TEST 3: Long Response Test")
    
    caller = ModelCaller()
    
    query = "Explain the concept of neural networks in detail with examples"
    print(f"Query: {query}")
    print("-" * 70)
    
    result = caller.call_generalist(query, max_tokens=1000)
    
    print(f"\n✅ Result:")
    print(f"  Answer length: {len(result['answer']) if result['answer'] else 0} chars")
    print(f"  First 200 chars: {result['answer'][:200] if result['answer'] else 'None'}...")
    print(f"  Tokens: {result['tokens_used']}")
    print(f"  Error: {result['error']}")
    
    assert result['answer'] is not None, "Should get answer"
    assert len(result['answer']) > 100, "Should be a substantial response"
    
    print(f"\n✅ TEST 3 PASSED")

def test_specialist_call():
    """Test 4: Specialist call (if you have a specialist endpoint)"""
    print_separator("TEST 4: Specialist Call")
    
    caller = ModelCaller()
    
    # Use a different model as "specialist" for testing
    specialist_endpoint = "meta/llama-3.1-8b-instruct"  # Smaller, faster model
    query = "What's the capital of France?"
    
    print(f"Specialist: {specialist_endpoint}")
    print(f"Query: {query}")
    print("-" * 70)
    
    result = caller.call_specialist(specialist_endpoint, query, max_tokens=100)
    
    print(f"\n✅ Result:")
    print(f"  Answer: {result['answer']}")
    print(f"  Tokens: {result['tokens_used']}")
    print(f"  Model: {result['model']}")
    print(f"  Error: {result['error']}")
    
    if result['error']:
        print(f"\n⚠️  Specialist call failed (this is OK if model not available)")
    else:
        assert result['answer'] is not None, "Should get answer"
        print(f"\n✅ TEST 4 PASSED")

def test_error_handling():
    """Test 5: Error handling with invalid inputs"""
    print_separator("TEST 5: Error Handling")
    
    caller = ModelCaller()
    
    # Test with empty query
    print("[Test 5.1] Empty query")
    result = caller.call_generalist("", max_tokens=100)
    print(f"  Result: Answer={'None' if not result['answer'] else 'Got answer'}, Error={result['error']}")
    
    # Test with very long query (edge case)
    print("\n[Test 5.2] Very long query")
    long_query = "Write SQL " * 1000  # 2000+ words
    result = caller.call_generalist(long_query[:5000], max_tokens=100)
    print(f"  Result: Answer={'None' if not result['answer'] else 'Got answer'}, Error={result['error']}")
    
    # Test with invalid specialist endpoint
    print("\n[Test 5.3] Invalid specialist endpoint")
    result = caller.call_specialist("invalid/model/path", "test query", max_tokens=100)
    print(f"  Result: Answer={result['answer'][:100] if result['answer'] else 'None'}")
    print(f"  Error: {result['error']}")
    assert result['error'] is not None, "Should have error for invalid model"
    
    print(f"\n✅ TEST 5 PASSED (Error handling works)")

def test_response_structure():
    """Test 6: Validate response structure"""
    print_separator("TEST 6: Response Structure Validation")
    
    caller = ModelCaller()
    
    result = caller.call_generalist("Hello", max_tokens=50)
    
    print("Checking response structure...")
    
    # Check all required fields exist
    assert 'answer' in result, "Response should have 'answer' field"
    assert 'model' in result, "Response should have 'model' field"
    assert 'tokens_used' in result, "Response should have 'tokens_used' field"
    assert 'error' in result, "Response should have 'error' field"
    
    # Check field types
    assert isinstance(result['answer'], (str, type(None))), "answer should be str or None"
    assert isinstance(result['model'], str), "model should be str"
    assert isinstance(result['tokens_used'], int), "tokens_used should be int"
    assert isinstance(result['error'], (str, type(None))), "error should be str or None"
    
    print("✅ All fields present and correct types")
    print(f"\nResponse structure:")
    for key, value in result.items():
        value_preview = str(value)[:50] if value else "None"
        print(f"  {key}: {type(value).__name__} = {value_preview}")
    
    print(f"\n✅ TEST 6 PASSED")

def test_token_limits():
    """Test 7: Test different token limits"""
    print_separator("TEST 7: Token Limit Tests")
    
    caller = ModelCaller()
    query = "Count from 1 to 100"
    
    token_limits = [50, 100, 200, 500]
    
    for limit in token_limits:
        print(f"\n[Test 7.{token_limits.index(limit)+1}] Max tokens: {limit}")
        result = caller.call_generalist(query, max_tokens=limit)
        
        print(f"  Tokens used: {result['tokens_used']}")
        print(f"  Answer length: {len(result['answer']) if result['answer'] else 0} chars")
        
        if result['answer']:
            assert result['tokens_used'] <= limit * 1.2, f"Should respect token limit (with buffer)"
        
        time.sleep(0.5)
    
    print(f"\n✅ TEST 7 PASSED")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("  MODEL CALLER TEST SUITE")
    print("="*70)
    
    tests = [
        ("Simple Generalist Calls", test_generalist_simple),
        ("SQL Generation", test_generalist_sql),
        ("Long Response", test_generalist_long_response),
        ("Specialist Call", test_specialist_call),
        ("Error Handling", test_error_handling),
        ("Response Structure", test_response_structure),
        ("Token Limits", test_token_limits),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n\n{'='*70}")
            print(f"Running: {test_name}")
            print(f"{'='*70}")
            
            test_func()
            results.append((test_name, "✅ PASSED", None))
            
        except AssertionError as e:
            results.append((test_name, "❌ FAILED", str(e)))
            print(f"\n❌ TEST FAILED: {e}")
            
        except Exception as e:
            results.append((test_name, "⚠️  ERROR", str(e)))
            print(f"\n⚠️  TEST ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70 + "\n")
    
    for test_name, status, error in results:
        print(f"{status}  {test_name}")
        if error:
            print(f"     Error: {error}")
    
    passed = sum(1 for _, status, _ in results if status == "✅ PASSED")
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"  {passed}/{total} tests passed")
    print(f"{'='*70}\n")
    
    return passed == total


if __name__ == "__main__":
    # Run individual test or all tests
    if len(sys.argv) > 1:
        test_num = sys.argv[1]
        
        test_map = {
            "1": test_generalist_simple,
            "2": test_generalist_sql,
            "3": test_generalist_long_response,
            "4": test_specialist_call,
            "5": test_error_handling,
            "6": test_response_structure,
            "7": test_token_limits,
        }
        
        if test_num in test_map:
            print(f"\nRunning Test {test_num}...")
            test_map[test_num]()
        else:
            print(f"Invalid test number. Choose from: {', '.join(test_map.keys())}")
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
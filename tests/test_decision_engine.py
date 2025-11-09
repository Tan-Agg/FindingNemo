import sys
sys.path.append('..')

from core.decision_engine import DecisionEngine
def test_decision_engine():
    engine = DecisionEngine()
    
    print("\n" + "="*60)
    print("TESTING DECISION ENGINE")
    print("="*60 + "\n")
    
    # Test Case 1: Below threshold
    print("Test Case 1: Below threshold (3 queries)")
    print("-" * 60)
    result = engine.make_decision("sql_generation", count=3)
    print(f"Decision: {result['decision']}")
    for reason in result['reasons']:
        print(reason)
    
    print("\n" + "="*60 + "\n")
    
    # Test Case 2: Meets threshold, high-value intent
    print("Test Case 2: Meets threshold (5 queries, high-value)")
    print("-" * 60)
    result = engine.make_decision("sql_generation", count=5)
    print(f"Decision: {result['decision']}")
    for reason in result['reasons']:
        print(reason)
    
    if result['decision'] == "TRAIN":
        print("\n" + "-" * 60)
        print("Training Plan:")
        print("-" * 60)
        plan = engine.get_training_plan(
            "sql_generation",
            "Generate SQL queries for database operations"
        )
        for step in plan['steps']:
            print(f"\nStep {step['step']}: {step['name']}")
            print(f"  Tool: {step['tool']}")
            print(f"  Details: {step['details']}")
            print(f"  Time: {step['estimated_time']}")
            print(f"  Cost: {step['estimated_cost']}")
        
        print(f"\nTotal Cost: {plan['total_cost']}")
        print(f"Total Time: {plan['total_time']}")
    
    print("\n" + "="*60 + "\n")
    
    # Test Case 3: High count but low-value intent
    print("Test Case 3: High count (10 queries) but low-value intent")
    print("-" * 60)
    result = engine.make_decision("random_task", count=10)
    print(f"Decision: {result['decision']}")
    for reason in result['reasons']:
        print(reason)
    
    print("\n" + "="*60 + "\n")
    
    # Test Case 4: Edge case - exactly threshold
    print("Test Case 4: Exactly at threshold (5 queries)")
    print("-" * 60)
    result = engine.make_decision("code_review", count=5)
    print(f"Decision: {result['decision']}")
    for reason in result['reasons']:
        print(reason)

if __name__ == "__main__":
    test_decision_engine()
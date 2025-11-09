import sys
sys.path.append('..')

from core.query_logger import QueryLogger

def test_query_logger():
    logger = QueryLogger()
    
    print("\n" + "="*60)
    print("TESTING QUERY LOGGER")
    print("="*60 + "\n")
    
    # Log some queries
    print("Logging queries...")
    logger.log_query("sql_generation", "Generate SQL queries", "Write SQL for top customers")
    logger.log_query("sql_generation", "Generate SQL queries", "Create query for revenue")
    logger.log_query("sql_generation", "Generate SQL queries", "SQL for analytics")
    logger.log_query("code_review", "Review code", "Check my Python code")
    
    print("\n" + "-"*60 + "\n")
    
    # Check counts
    print("Checking counts...")
    sql_count = logger.get_count("sql_generation")
    code_count = logger.get_count("code_review")
    print(f"sql_generation: {sql_count} queries")
    print(f"code_review: {code_count} queries")
    
    print("\n" + "-"*60 + "\n")
    
    # Check bottlenecks
    print("Checking bottlenecks (threshold=3)...")
    bottlenecks = logger.get_bottlenecks(threshold=3)
    
    if bottlenecks:
        print(f"✅ Found {len(bottlenecks)} bottlenecks:")
        for b in bottlenecks:
            print(f"   - {b['intent_label']}: {b['count']} queries")
    else:
        print("❌ No bottlenecks found")

if __name__ == "__main__":
    test_query_logger()
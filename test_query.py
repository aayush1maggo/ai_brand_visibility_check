#!/usr/bin/env python3
"""
Test the query_model_with_search function specifically
"""

import time

def test_single_query():
    print("Testing single model query...")
    try:
        from app import query_model_with_search
        print("Imported query_model_with_search")
        
        start = time.time()
        result = query_model_with_search("GPT-4o-mini", "test prompt about web development", "Test Brand")
        elapsed = time.time() - start
        
        print(f"Query completed in {elapsed:.2f}s")
        print(f"Response length: {len(result.get('response', ''))}")
        print(f"Brand mentioned: {result.get('brand_mentioned', False)}")
        print(f"First 100 chars: {result.get('response', '')[:100]}...")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Query Test ===")
    success = test_single_query()
    if success:
        print("Single query works fine")
    else:
        print("Single query is the problem")
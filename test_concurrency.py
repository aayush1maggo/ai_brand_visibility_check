#!/usr/bin/env python3
"""
Test the concurrent execution to find the threading/concurrency issue
"""

import os
import time
import concurrent.futures
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_model_with_search_test(model_name, prompt, brand_name):
    """Same function as in app.py but with timing"""
    try:
        start_time = time.time()
        print(f"[START] {model_name} - {prompt[:50]}...")
        
        # Get the actual model ID
        MODELS = {
            "GPT-4o": "gpt-4o",
            "GPT-4o-mini": "gpt-4o-mini",
            "GPT-3.5-turbo": "gpt-3.5-turbo"
        }
        model_id = MODELS.get(model_name, model_name)
        
        # Use OpenAI API with search functionality
        resp = openai.responses.create(
            model=model_id,
            tools=[{
                "type": "web_search_preview",
                "user_location": {
                    "type": "approximate",
                    "country": "AU"
                }
            }],
            input=prompt
        )
        text = resp.output_text
        mentioned = brand_name.lower() in text.lower()
        
        elapsed = time.time() - start_time
        print(f"[DONE] {model_name} completed in {elapsed:.2f}s - mentioned: {mentioned}")
        
        return {"model": model_name, "prompt": prompt, "response": text, "brand_mentioned": mentioned}
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] {model_name} failed after {elapsed:.2f}s: {e}")
        return {"model": model_name, "prompt": prompt, "response": "", "brand_mentioned": False}

def test_sequential_calls():
    """Test making calls one by one (should work)"""
    print("=== Testing Sequential Calls ===")
    
    prompts = [
        "best web development companies in Australia 2025",
        "how to find reliable web development services",
        "top web development providers near me"
    ]
    models = ["GPT-4o", "GPT-4o-mini", "GPT-3.5-turbo"]
    brand_name = "Test Company"
    
    results = []
    total_start = time.time()
    
    for prompt in prompts:
        for model in models:
            result = query_model_with_search_test(model, prompt, brand_name)
            results.append(result)
            time.sleep(1)  # Small delay between calls
    
    total_elapsed = time.time() - total_start
    successful = sum(1 for r in results if r['response'])
    
    print(f"\nSequential Results: {successful}/{len(results)} successful in {total_elapsed:.2f}s")
    return len(results) == successful

def test_concurrent_calls():
    """Test the concurrent execution (this should hang/fail)"""
    print("\n=== Testing Concurrent Calls ===")
    
    prompts = [
        "best web development companies in Australia 2025",
        "how to find reliable web development services", 
        "top web development providers near me"
    ]
    models = ["GPT-4o", "GPT-4o-mini", "GPT-3.5-turbo"]
    brand_name = "Test Company"
    
    results = []
    total_start = time.time()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            futures = {
                executor.submit(query_model_with_search_test, m, p, brand_name): (p, m) 
                for p in prompts for m in models
            }
            
            print(f"Submitted {len(futures)} concurrent tasks...")
            
            # Wait for completion with timeout
            completed_count = 0
            for fut in concurrent.futures.as_completed(futures, timeout=60):
                p, m = futures[fut]
                result = fut.result()
                results.append(result)
                completed_count += 1
                
                elapsed = time.time() - total_start
                print(f"[PROGRESS] {completed_count}/{len(futures)} completed in {elapsed:.2f}s")
    
    except concurrent.futures.TimeoutError:
        elapsed = time.time() - total_start
        print(f"[TIMEOUT] Concurrent execution timed out after {elapsed:.2f}s")
        print(f"Completed: {len(results)}/{len(futures)}")
        return False
        
    except Exception as e:
        elapsed = time.time() - total_start
        print(f"[ERROR] Concurrent execution failed after {elapsed:.2f}s: {e}")
        return False
    
    total_elapsed = time.time() - total_start
    successful = sum(1 for r in results if r['response'])
    
    print(f"\nConcurrent Results: {successful}/{len(results)} successful in {total_elapsed:.2f}s")
    return len(results) == successful

def test_reduced_concurrency():
    """Test with fewer concurrent workers"""
    print("\n=== Testing Reduced Concurrency (2 workers) ===")
    
    prompts = ["best web development companies in Australia 2025", "top web development providers"]
    models = ["GPT-4o-mini", "GPT-3.5-turbo"]  # Use fewer models
    brand_name = "Test Company"
    
    results = []
    total_start = time.time()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers
            futures = {
                executor.submit(query_model_with_search_test, m, p, brand_name): (p, m) 
                for p in prompts for m in models
            }
            
            print(f"Submitted {len(futures)} tasks with 2 workers...")
            
            for fut in concurrent.futures.as_completed(futures, timeout=45):
                p, m = futures[fut]
                result = fut.result()
                results.append(result)
    
    except Exception as e:
        elapsed = time.time() - total_start
        print(f"[ERROR] Reduced concurrency failed after {elapsed:.2f}s: {e}")
        return False
    
    total_elapsed = time.time() - total_start
    successful = sum(1 for r in results if r['response'])
    
    print(f"\nReduced Concurrency Results: {successful}/{len(results)} successful in {total_elapsed:.2f}s")
    return len(results) == successful

if __name__ == "__main__":
    print("=== Testing Concurrency Issues ===\n")
    
    # Test 1: Sequential (should work)
    sequential_works = test_sequential_calls()
    
    # Test 2: Full concurrency (might fail)
    concurrent_works = test_concurrent_calls()
    
    # Test 3: Reduced concurrency (might work)
    reduced_works = test_reduced_concurrency()
    
    print("\n=== RESULTS ===")
    print(f"Sequential execution: {'WORKS' if sequential_works else 'FAILS'}")
    print(f"Full concurrency (8 workers): {'WORKS' if concurrent_works else 'FAILS'}")
    print(f"Reduced concurrency (2 workers): {'WORKS' if reduced_works else 'FAILS'}")
    
    if sequential_works and not concurrent_works:
        print("\n[CONCLUSION] The issue is definitely in the concurrent execution!")
        print("Possible solutions:")
        print("1. Reduce max_workers from 8 to 2-3")
        print("2. Add delays between API calls")
        print("3. Implement retry logic with exponential backoff")
        if reduced_works:
            print("4. The reduced concurrency worked - use fewer workers!")
    elif not sequential_works:
        print("\n[CONCLUSION] The issue is in the individual API calls, not concurrency")
    else:
        print("\n[CONCLUSION] Concurrency works fine - issue might be elsewhere")
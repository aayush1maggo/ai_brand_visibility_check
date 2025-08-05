#!/usr/bin/env python3
"""
Quick timing test to identify the slowest parts of your analysis.
This runs a minimal version to isolate bottlenecks.
"""

import time
import os
from dotenv import load_dotenv
import openai

# Load environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def time_openai_calls():
    """Test the speed of different OpenAI API calls"""
    print("Testing OpenAI API call speeds:")
    print("-" * 40)
    
    # Test different models
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "o4-mini"]
    
    for model in models:
        try:
            start = time.time()
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10
            )
            duration = time.time() - start
            print(f"{model}: {duration:.2f}s")
        except Exception as e:
            print(f"{model}: ERROR - {e}")
    
    print("\nTesting web search API:")
    try:
        start = time.time()
        response = openai.responses.create(
            model="o4-mini",
            tools=[{
                "type": "web_search_preview",
                "user_location": {
                    "type": "approximate",
                    "country": "AU"
                }
            }],
            input="Best video production companies Australia"
        )
        duration = time.time() - start
        print(f"Web search: {duration:.2f}s")
    except Exception as e:
        print(f"Web search: ERROR - {e}")

def test_concurrent_calls():
    """Test how parallel calls perform vs sequential"""
    import concurrent.futures
    
    print("\nTesting concurrent vs sequential calls:")
    print("-" * 40)
    
    def single_call():
        return openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
    
    # Sequential calls
    start = time.time()
    for i in range(3):
        single_call()
    sequential_time = time.time() - start
    print(f"3 sequential calls: {sequential_time:.2f}s")
    
    # Parallel calls with MAX_PARALLEL=3
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(single_call) for _ in range(3)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    parallel_time = time.time() - start
    print(f"3 parallel calls (max_workers=3): {parallel_time:.2f}s")
    
    print(f"Speedup: {sequential_time/parallel_time:.1f}x faster")

if __name__ == "__main__":
    print("Quick Timing Test")
    print("=" * 30)
    
    time_openai_calls()
    test_concurrent_calls()
    
    print("\n" + "=" * 30)
    print("Test complete!")
#!/usr/bin/env python3
"""
Test the full analyze flow step by step to find where it hangs
"""

import os
import sys
import traceback
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_step_by_step():
    """Test each step of the analysis to find where it hangs"""
    
    print("Testing full analysis flow step by step...")
    
    try:
        from app import generate_prompts_for_keyword, query_model_with_search, run_full_analysis
        print("[OK] Imported main functions")
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        traceback.print_exc()
        return
    
    keyword = "web development"
    brand_name = "Test Company"
    
    # Step 1: Test prompt generation
    print("\n--- Step 1: Testing prompt generation ---")
    try:
        start_time = time.time()
        prompts = generate_prompts_for_keyword(keyword)
        elapsed = time.time() - start_time
        print(f"[OK] Generated {len(prompts)} prompts in {elapsed:.2f}s")
        if prompts:
            print(f"Sample prompt: {prompts[0]}")
    except Exception as e:
        print(f"[ERROR] Prompt generation failed: {e}")
        traceback.print_exc()
        return
    
    # Step 2: Test single model query
    print("\n--- Step 2: Testing single model query ---")
    try:
        start_time = time.time()
        if prompts:
            result = query_model_with_search("GPT-4o-mini", prompts[0], brand_name)
            elapsed = time.time() - start_time
            print(f"[OK] Single query completed in {elapsed:.2f}s")
            print(f"Response length: {len(result.get('response', ''))}")
            print(f"Brand mentioned: {result.get('brand_mentioned', False)}")
        else:
            print("[SKIP] No prompts to test with")
    except Exception as e:
        print(f"[ERROR] Single query failed: {e}")
        traceback.print_exc()
        return
    
    # Step 3: Test competitor analysis specifically
    print("\n--- Step 3: Testing competitor analysis ---")
    try:
        from robust_competitor_analysis import create_robust_competitor_analyzer
        import openai
        
        # Create sample results
        sample_results = [
            {"model": "GPT-4o", "prompt": prompts[0] if prompts else "test", "response": "Sample response with Microsoft, Google, and Amazon mentioned", "brand_mentioned": False},
            {"model": "GPT-4o-mini", "prompt": prompts[0] if prompts else "test", "response": f"{brand_name} is a great company along with Apple and IBM", "brand_mentioned": True}
        ]
        
        start_time = time.time()
        competitor_analyzer = create_robust_competitor_analyzer(openai)
        competitor_analysis = competitor_analyzer.analyze_competitors(sample_results, brand_name, skip_ai_validation=True)
        elapsed = time.time() - start_time
        
        print(f"[OK] Competitor analysis completed in {elapsed:.2f}s")
        print(f"Found competitors: {competitor_analysis['competitors']}")
        
    except Exception as e:
        print(f"[ERROR] Competitor analysis failed: {e}")
        traceback.print_exc()
        return
    
    # Step 4: Test full analysis with timeout
    print("\n--- Step 4: Testing full analysis (with 30s timeout) ---")
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler():
        raise TimeoutError("Analysis timed out after 30 seconds")
    
    try:
        start_time = time.time()
        
        # Set a simple progress callback
        def progress_callback(step, message, percentage):
            elapsed = time.time() - start_time
            print(f"  Progress: Step {step} - {message} ({percentage}%) - {elapsed:.1f}s elapsed")
            if elapsed > 30:  # 30 second timeout
                raise TimeoutError("Analysis timed out after 30 seconds")
        
        result = run_full_analysis(keyword, brand_name, progress_callback, performance_mode=True)
        elapsed = time.time() - start_time
        
        print(f"[OK] Full analysis completed in {elapsed:.2f}s")
        print(f"Score: {result.get('score', 'N/A')}")
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print(f"[TIMEOUT] Analysis timed out after {elapsed:.2f}s")
        print("This is likely where the 500 error occurs!")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] Full analysis failed after {elapsed:.2f}s: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Testing Full Analysis Flow ===\n")
    test_step_by_step()
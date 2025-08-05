#!/usr/bin/env python3
"""
Test to isolate the competitor analyzer bug causing 500 error
"""

import os
import sys
import traceback
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def test_competitor_analyzer_bug():
    """Test the specific bug in competitor analyzer"""
    
    print("Testing competitor analyzer bug...")
    
    # Import the function
    try:
        from robust_competitor_analysis import create_robust_competitor_analyzer
        print("[OK] Imported create_robust_competitor_analyzer")
    except Exception as e:
        print(f"[ERROR] Failed to import: {e}")
        return False
    
    # Create sample results data (same format as real app)
    sample_results = [
        {"model": "GPT-4o", "prompt": "test prompt", "response": "Sample response mentioning Company A and Company B", "brand_mentioned": True},
        {"model": "GPT-4o-mini", "prompt": "test prompt 2", "response": "Another response with Company C", "brand_mentioned": False}
    ]
    
    print(f"[OK] Created sample results: {len(sample_results)} items")
    
    # Test 1: Try the current way (passing openai module - this should fail)
    print("\n--- Test 1: Current buggy way ---")
    try:
        competitor_analyzer = create_robust_competitor_analyzer(openai)
        print("[TRYING] Created analyzer with openai module...")
        
        # This should fail or hang
        competitor_analysis = competitor_analyzer.analyze_competitors(sample_results, "Test Brand", skip_ai_validation=True)
        print("[UNEXPECTED] Analysis completed - this shouldn't work!")
        
    except Exception as e:
        print(f"[EXPECTED ERROR] Failed as expected: {e}")
        print("This confirms the bug!")
    
    # Test 2: Try the correct way (passing openai client)
    print("\n--- Test 2: Fixed way ---")
    try:
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        competitor_analyzer = create_robust_competitor_analyzer(openai_client)
        print("[OK] Created analyzer with proper OpenAI client")
        
        # This should work
        competitor_analysis = competitor_analyzer.analyze_competitors(sample_results, "Test Brand", skip_ai_validation=True)
        print(f"[SUCCESS] Analysis completed! Found {len(competitor_analysis['competitors'])} competitors")
        print(f"Competitors: {competitor_analysis['competitors']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Even the fix failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Competitor Analyzer Bug ===\n")
    
    success = test_competitor_analyzer_bug()
    
    if success:
        print("\n[CONCLUSION] Bug confirmed and fix verified!")
        print("The issue is in app.py line 1179:")
        print("  BUGGY: create_robust_competitor_analyzer(openai)")
        print("  FIXED: create_robust_competitor_analyzer(openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')))")
    else:
        print("\n[CONCLUSION] The bug might be elsewhere or more complex.")
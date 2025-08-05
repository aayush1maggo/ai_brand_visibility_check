#!/usr/bin/env python3
"""
Simple test to find exactly where it hangs
"""

import time
import sys

def test_prompt_generation():
    print("Testing prompt generation...")
    try:
        from app import generate_prompts_for_keyword
        print("Imported generate_prompts_for_keyword")
        
        start = time.time()
        prompts = generate_prompts_for_keyword("test keyword")
        elapsed = time.time() - start
        
        print(f"Generated prompts in {elapsed:.2f}s: {prompts}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Simple Test ===")
    success = test_prompt_generation()
    if success:
        print("Prompt generation works fine")
    else:
        print("Prompt generation is the problem")
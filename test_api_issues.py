#!/usr/bin/env python3
"""
Test for specific API issues that could cause 500 error
"""

import os
import openai
from dotenv import load_dotenv

load_dotenv()

def test_web_search_api():
    """Test the exact web search API call"""
    print("Testing web search API call...")
    
    try:
        # Test the exact same call as in the app
        resp = openai.responses.create(
            model="gpt-4o-mini",
            tools=[{
                "type": "web_search_preview",
                "user_location": {
                    "type": "approximate",
                    "country": "AU"
                }
            }],
            input="best web development companies"
        )
        
        text = resp.output_text
        print(f"[SUCCESS] Web search API call worked!")
        print(f"Response length: {len(text)}")
        print(f"First 200 chars: {text[:200]}...")
        return True
        
    except Exception as e:
        print(f"[ERROR] Web search API failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def test_model_access():
    """Test if the models are accessible"""
    print("\nTesting model access...")
    
    models_to_test = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    for model in models_to_test:
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print(f"[OK] {model} is accessible")
        except Exception as e:
            print(f"[ERROR] {model} failed: {e}")

if __name__ == "__main__":
    print("=== Testing API Issues ===\n")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] No OPENAI_API_KEY found")
        exit(1)
    else:
        print(f"[OK] API key found: {api_key[:10]}...")
    
    # Set the API key
    openai.api_key = api_key
    
    # Test web search
    web_search_works = test_web_search_api()
    
    # Test model access
    test_model_access()
    
    if not web_search_works:
        print("\n[CONCLUSION] The web search API is the problem.")
        print("Possible causes:")
        print("- API key doesn't have web search preview access")
        print("- Web search preview not available in your region")
        print("- Rate limiting or quota exceeded")
    else:
        print("\n[CONCLUSION] Web search API works fine individually.")
        print("The issue might be in the concurrent execution or threading.")
#!/usr/bin/env python3
"""
Test script to simulate form submission and identify the 500 error
"""

import requests
import json

def test_analyze_endpoint():
    """Test the /analyze endpoint with form data"""
    
    url = "http://127.0.0.1:8002/analyze"
    
    # Test data - same as what would come from the form
    form_data = {
        "keyword": "web development",
        "brand_name": "Test Company",
        "performance_mode": False,  # This is the key - demo_mode will default to False
        "demo_mode": False
    }
    
    print("Testing /analyze endpoint...")
    print(f"URL: {url}")
    print(f"Form data: {form_data}")
    
    try:
        # Send POST request
        response = requests.post(url, data=form_data)
        
        print(f"\nResponse status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("[SUCCESS] Request completed successfully!")
            result = response.json()
            print(f"Score: {result.get('score', 'N/A')}")
            print(f"Summary: {result.get('summary', 'N/A')[:100]}...")
        else:
            print(f"[ERROR] Request failed with status {response.status_code}")
            print(f"Response text: {response.text}")
            
            # Try to parse as JSON for detailed error
            try:
                error_data = response.json()
                print(f"Error detail: {error_data.get('detail', 'No detail provided')}")
            except:
                print("Response is not valid JSON")
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to server. Is it running on http://127.0.0.1:8002?")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    test_analyze_endpoint()
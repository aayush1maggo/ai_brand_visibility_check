#!/usr/bin/env python3
"""
Debug script to test the analyze functionality and identify the 500 error
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Load environment variables
load_dotenv()

def test_imports():
    """Test all imports to identify missing dependencies"""
    print("Testing imports...")
    try:
        import fastapi
        print("[OK] FastAPI imported")
        
        import uvicorn
        print("[OK] Uvicorn imported")
        
        import openai
        print(f"[OK] OpenAI imported (version: {openai.__version__})")
        
        import pandas
        print("[OK] Pandas imported")
        
        import matplotlib
        print("[OK] Matplotlib imported")
        
        import plotly
        print("[OK] Plotly imported")
        
        import seaborn
        print("[OK] Seaborn imported")
        
        import textblob
        print("[OK] TextBlob imported")
        
        import reportlab
        print("[OK] ReportLab imported")
        
        from robust_competitor_analysis import create_robust_competitor_analyzer
        print("[OK] RobustCompetitorAnalyzer imported")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Import error: {e}")
        traceback.print_exc()
        return False

def test_openai_config():
    """Test OpenAI configuration"""
    print("\nTesting OpenAI configuration...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found in environment variables")
        return False
    
    if api_key.startswith("sk-"):
        print("[OK] OPENAI_API_KEY format looks correct")
    else:
        print("[ERROR] OPENAI_API_KEY format looks incorrect (should start with 'sk-')")
        return False
    
    # Test a simple API call
    try:
        import openai
        openai.api_key = api_key
        
        # Try a simple completion
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("[OK] OpenAI API call successful")
        return True
        
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return False

def test_analyze_function():
    """Test the main analyze function with sample data"""
    print("\nTesting analyze function...")
    
    try:
        from app import run_full_analysis
        
        # Test with minimal parameters
        result = run_full_analysis(
            keyword="test service",
            brand_name="Test Brand",
            performance_mode=True  # Use performance mode to reduce API calls
        )
        
        print("[OK] run_full_analysis completed successfully")
        print(f"  Score: {result.get('score', 'N/A')}")
        print(f"  Summary: {result.get('summary', 'N/A')[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] run_full_analysis failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Debug Analysis for 500 Error ===\n")
    
    imports_ok = test_imports()
    if not imports_ok:
        print("\n[FAIL] Import test failed - this might be the cause of the 500 error")
        sys.exit(1)
    
    openai_ok = test_openai_config()
    if not openai_ok:
        print("\n[FAIL] OpenAI configuration test failed - this might be the cause of the 500 error")
        print("Make sure you have OPENAI_API_KEY set in your .env file")
        sys.exit(1)
    
    analyze_ok = test_analyze_function()
    if not analyze_ok:
        print("\n[FAIL] Analyze function test failed - this is likely the cause of the 500 error")
        sys.exit(1)
    
    print("\n[SUCCESS] All tests passed! The 500 error might be intermittent or caused by specific inputs.")
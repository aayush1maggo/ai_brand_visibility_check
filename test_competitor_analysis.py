#!/usr/bin/env python3
"""
Test script to demonstrate the robust competitor analysis system.
"""

import os
from dotenv import load_dotenv
import openai
from robust_competitor_analysis import create_robust_competitor_analyzer

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_competitor_analysis():
    """
    Test the robust competitor analysis with sample data.
    """
    
    # Sample search results (simulating AI responses)
    sample_results = [
        {
            "model": "GPT-4o",
            "prompt": "Find the best hardware stores in Melbourne",
            "response": """
            Here are the top hardware stores in Melbourne:
            
            ## Bunnings Warehouse
            The largest hardware chain in Australia with multiple locations.
            
            ## Mitre 10
            Another major hardware retailer with good selection.
            
            ## Home Hardware
            Local hardware store with personalized service.
            
            ## Masters Home Improvement
            Competitive pricing and wide selection.
            
            ## True Value Hardware
            Independent hardware store with expert advice.
            """,
            "brand_mentioned": True
        },
        {
            "model": "GPT-4o-mini", 
            "prompt": "Best DIY stores Melbourne",
            "response": """
            Top DIY stores in Melbourne:
            
            • **Bunnings** - Largest selection
            • **Mitre 10** - Competitive prices  
            • **Home Hardware** - Local expertise
            • **Masters** - Good value
            • **True Value** - Expert advice
            """,
            "brand_mentioned": True
        }
    ]
    
    # Test the robust analyzer
    analyzer = create_robust_competitor_analyzer(openai)
    
    print("=== Testing Robust Competitor Analysis ===")
    print(f"User brand: Bunnings")
    print(f"Sample results: {len(sample_results)} responses")
    print()
    
    try:
        analysis = analyzer.analyze_competitors(sample_results, "Bunnings")
        
        print("✅ Analysis Results:")
        print(f"Competitors found: {len(analysis['competitors'])}")
        print(f"Competitors: {analysis['competitors']}")
        print()
        print("Mention counts:")
        for comp, count in analysis['mention_counts'].items():
            print(f"  {comp}: {count}")
        print()
        print("Market shares:")
        for comp, share in analysis['market_shares'].items():
            print(f"  {comp}: {share:.1f}%")
        print()
        print(f"Total mentions: {analysis['total_mentions']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def compare_methods():
    """
    Compare the old vs new method.
    """
    print("\n=== Method Comparison ===")
    
    # Old method (fragile regex)
    print("Old Method Issues:")
    print("- Relies on specific markdown patterns")
    print("- Misses competitors in different formats")
    print("- No AI validation")
    print("- No fuzzy matching")
    print("- Prone to false positives")
    
    print("\nNew Method Improvements:")
    print("- Multiple extraction methods")
    print("- AI-powered validation")
    print("- Fuzzy matching for name variations")
    print("- Better filtering of false positives")
    print("- Handles different text formats")
    print("- More accurate counting")

if __name__ == "__main__":
    test_competitor_analysis()
    compare_methods() 
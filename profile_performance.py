#!/usr/bin/env python3
"""
Performance profiler for the brand visibility analysis.
Run this to identify bottlenecks in your code.
"""

import cProfile
import pstats
import io
import time
from app import run_full_analysis

def profile_analysis(keyword="video production", brand_name="Test Brand"):
    """Profile a complete analysis run"""
    print(f"Profiling analysis for: {brand_name} - {keyword}")
    
    # Create a profiler
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    try:
        # Run the analysis
        result = run_full_analysis(keyword, brand_name, performance_mode=True)
        print(f"Analysis completed successfully. Score: {result.get('score', 'N/A')}")
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    # Stop profiling
    profiler.disable()
    
    # Create a string buffer to capture the stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Sort by cumulative time and show top 20 functions
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    
    # Print the results
    print("\n" + "="*80)
    print("TOP 20 SLOWEST FUNCTIONS (by cumulative time)")
    print("="*80)
    print(s.getvalue())
    
    # Also sort by total time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('tottime')
    ps.print_stats(20)
    
    print("\n" + "="*80)
    print("TOP 20 SLOWEST FUNCTIONS (by total time)")
    print("="*80)
    print(s.getvalue())

def time_individual_functions():
    """Time individual functions separately"""
    from app import generate_prompts_for_keyword, query_model_with_search, batch_sentiment_analysis
    
    keyword = "video production"
    brand_name = "Test Brand"
    
    print("Timing individual functions:")
    print("-" * 40)
    
    # Time prompt generation
    start = time.time()
    prompts = generate_prompts_for_keyword(keyword)
    print(f"generate_prompts_for_keyword: {time.time() - start:.2f}s")
    
    if prompts:
        # Time a single search query
        start = time.time()
        result = query_model_with_search("GPT-4o-mini", prompts[0], brand_name)
        print(f"query_model_with_search (single): {time.time() - start:.2f}s")
        
        # Time sentiment analysis
        if result['response']:
            start = time.time()
            sentiment = batch_sentiment_analysis((result['response'],), brand_name)
            print(f"batch_sentiment_analysis (single): {time.time() - start:.2f}s")

if __name__ == "__main__":
    print("Brand Visibility Analysis - Performance Profiler")
    print("=" * 50)
    
    # First, time individual functions
    time_individual_functions()
    
    print("\n" + "=" * 50)
    print("Running full profiler...")
    
    # Then run the full profiler
    profile_analysis()
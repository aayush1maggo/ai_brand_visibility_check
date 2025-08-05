import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import traceback

# Add the current directory to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions we want to test
from app import (
    query_model_with_search,
    batch_sentiment_analysis,
    extract_sentiment_drivers,
    generate_recommendations,
    create_seaborn_pie,
    create_plotly_enhanced_pie,
    generate_demo_data,
    run_full_analysis
)

class TestAppFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.test_brand_name = "TestBrand"
        self.test_keyword = "test product"
        self.test_texts = ("This is a great product", "Amazing quality", "Excellent service")
        self.test_positives = ("Great quality products", "Excellent customer service", "Amazing experience")
    
    def test_batch_sentiment_analysis(self):
        """Test the batch sentiment analysis function"""
        try:
            print("\n[TEST] Testing batch_sentiment_analysis...")
            result = batch_sentiment_analysis(self.test_texts, self.test_brand_name)
            print(f"[TEST] batch_sentiment_analysis result: {result}")
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), len(self.test_texts))
            print("[TEST] ✓ batch_sentiment_analysis passed")
        except Exception as e:
            print(f"[TEST] ✗ batch_sentiment_analysis failed: {e}")
            traceback.print_exc()
            raise
    
    def test_extract_sentiment_drivers(self):
        """Test the sentiment drivers extraction function"""
        try:
            print("\n[TEST] Testing extract_sentiment_drivers...")
            result = extract_sentiment_drivers(self.test_positives, self.test_brand_name)
            print(f"[TEST] extract_sentiment_drivers result: {result}")
            self.assertIsInstance(result, dict)
            self.assertIn("drivers", result)
            print("[TEST] ✓ extract_sentiment_drivers passed")
        except Exception as e:
            print(f"[TEST] ✗ extract_sentiment_drivers failed: {e}")
            traceback.print_exc()
            raise
    
    def test_generate_recommendations(self):
        """Test the recommendations generation function"""
        try:
            print("\n[TEST] Testing generate_recommendations...")
            result = generate_recommendations(self.test_brand_name, self.test_keyword)
            print(f"[TEST] generate_recommendations result length: {len(result)}")
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            print("[TEST] ✓ generate_recommendations passed")
        except Exception as e:
            print(f"[TEST] ✗ generate_recommendations failed: {e}")
            traceback.print_exc()
            raise
    
    def test_create_seaborn_pie(self):
        """Test the seaborn pie chart creation"""
        try:
            print("\n[TEST] Testing create_seaborn_pie...")
            labels = ["Brand A", "Brand B", "Brand C"]
            values = [30, 25, 45]
            result = create_seaborn_pie(labels, values, self.test_brand_name)
            print(f"[TEST] create_seaborn_pie result type: {type(result)}")
            self.assertIsInstance(result, object)  # matplotlib figure object
            print("[TEST] ✓ create_seaborn_pie passed")
        except Exception as e:
            print(f"[TEST] ✗ create_seaborn_pie failed: {e}")
            traceback.print_exc()
            raise
    
    def test_create_plotly_enhanced_pie(self):
        """Test the plotly pie chart creation"""
        try:
            print("\n[TEST] Testing create_plotly_enhanced_pie...")
            labels = ["Brand A", "Brand B", "Brand C"]
            values = [30, 25, 45]
            result = create_plotly_enhanced_pie(labels, values, self.test_brand_name)
            print(f"[TEST] create_plotly_enhanced_pie result type: {type(result)}")
            self.assertIsInstance(result, object)  # plotly figure object
            print("[TEST] ✓ create_plotly_enhanced_pie passed")
        except Exception as e:
            print(f"[TEST] ✗ create_plotly_enhanced_pie failed: {e}")
            traceback.print_exc()
            raise
    
    @patch('app.openai')
    def test_query_model_with_search(self, mock_openai):
        """Test the OpenAI search query function"""
        try:
            print("\n[TEST] Testing query_model_with_search...")
            # Mock the OpenAI response
            mock_response = MagicMock()
            mock_response.output_text = "This is a test response mentioning TestBrand"
            mock_openai.responses.create.return_value = mock_response
            
            result = query_model_with_search("GPT-4o", "test prompt", self.test_brand_name)
            print(f"[TEST] query_model_with_search result: {result}")
            self.assertIsInstance(result, dict)
            self.assertIn("model", result)
            self.assertIn("response", result)
            self.assertIn("brand_mentioned", result)
            print("[TEST] ✓ query_model_with_search passed")
        except Exception as e:
            print(f"[TEST] ✗ query_model_with_search failed: {e}")
            traceback.print_exc()
            raise
    
    def test_generate_demo_data(self):
        """Test the demo data generation function"""
        try:
            print("\n[TEST] Testing generate_demo_data...")
            result = generate_demo_data(self.test_keyword, self.test_brand_name)
            print(f"[TEST] generate_demo_data keys: {list(result.keys())}")
            self.assertIsInstance(result, dict)
            required_keys = ["score", "summary", "sentiment", "perception", "sentiment_drivers", "summary_df", "raw_results", "competitor_plot", "competitor_insights", "recommendations"]
            for key in required_keys:
                self.assertIn(key, result)
            print("[TEST] ✓ generate_demo_data passed")
        except Exception as e:
            print(f"[TEST] ✗ generate_demo_data failed: {e}")
            traceback.print_exc()
            raise
    
    @patch('app.query_model_with_search')
    @patch('app.batch_sentiment_analysis')
    @patch('app.extract_sentiment_drivers')
    @patch('app.generate_recommendations')
    def test_run_full_analysis(self, mock_recommendations, mock_sentiment_drivers, mock_sentiment, mock_query):
        """Test the full analysis function with mocked dependencies"""
        try:
            print("\n[TEST] Testing run_full_analysis...")
            
            # Mock all the dependencies
            mock_query.return_value = {
                "model": "GPT-4o",
                "response": "Test response",
                "brand_mentioned": True
            }
            mock_sentiment.return_value = ["Positive", "Neutral", "Positive"]
            mock_sentiment_drivers.return_value = {"drivers": [{"driver": "Quality", "strength": 8}]}
            mock_recommendations.return_value = "Test recommendations"
            
            def mock_progress_callback(step, message, percentage):
                print(f"[PROGRESS] Step {step}: {message} ({percentage}%)")
            
            result = run_full_analysis(self.test_keyword, self.test_brand_name, mock_progress_callback, False)
            print(f"[TEST] run_full_analysis keys: {list(result.keys())}")
            self.assertIsInstance(result, dict)
            required_keys = ["score", "summary", "sentiment", "perception", "sentiment_drivers", "summary_df", "raw_results", "competitor_plot", "competitor_insights", "recommendations"]
            for key in required_keys:
                self.assertIn(key, result)
            print("[TEST] ✓ run_full_analysis passed")
        except Exception as e:
            print(f"[TEST] ✗ run_full_analysis failed: {e}")
            traceback.print_exc()
            raise

def run_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("STARTING UNIT TESTS FOR APP FUNCTIONS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAppFunctions)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Test script for NeuroVest API
Tests all endpoints with real requests
"""

import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8000"  # Change to Railway URL after deployment
# API_URL = "https://neurovest-api.up.railway.app"  # Production URL

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(name):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}TEST: {name}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")

def print_info(msg):
    print(f"{YELLOW}ℹ {msg}{RESET}")

def test_root():
    """Test root endpoint"""
    print_test("GET / (API Information)")
    try:
        r = requests.get(f"{API_URL}/")
        print(f"Status Code: {r.status_code}")
        print(f"Response: {json.dumps(r.json(), indent=2)}")

        if r.status_code == 200:
            print_success("Root endpoint working")
            return True
        else:
            print_error(f"Expected 200, got {r.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_health():
    """Test health check endpoint"""
    print_test("GET /health (Health Check)")
    try:
        r = requests.get(f"{API_URL}/health")
        print(f"Status Code: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            assert data["status"] == "healthy", "Status should be 'healthy'"
            assert data["database"] == "connected", "Database should be connected"
            assert "assets_count" in data, "Should include assets_count"

            print_success("Health check passed")
            print_info(f"Assets in DB: {data['assets_count']}")
            print_info(f"Last prediction: {data.get('last_prediction', 'N/A')}")
            return True
        else:
            print_error(f"Expected 200, got {r.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_register():
    """Test user registration"""
    print_test("POST /api/auth/register (Create User)")
    try:
        username = f"test_user_{int(requests.get(f'{API_URL}/health').json()['timestamp'].split('T')[1].replace(':', '').replace('.', '')[:6])}"
        r = requests.post(f"{API_URL}/api/auth/register?username={username}")
        print(f"Status Code: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            assert "api_key" in data, "Should return API key"
            assert "user_id" in data, "Should return user ID"

            api_key = data["api_key"]
            print_success(f"User created: {username}")
            print_info(f"User ID: {data['user_id']}")
            print_info(f"API Key: {api_key}")

            return api_key
        else:
            print_error(f"Expected 200, got {r.status_code}")
            return None
    except Exception as e:
        print_error(f"Request failed: {e}")
        return None

def test_predictions_no_auth():
    """Test predictions without authentication (should fail)"""
    print_test("GET /api/predictions (Without Auth - Should Fail)")
    try:
        r = requests.get(f"{API_URL}/api/predictions")
        print(f"Status Code: {r.status_code}")

        if r.status_code == 401:
            print_success("Correctly rejected unauthorized request")
            return True
        else:
            print_error(f"Expected 401, got {r.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_predictions_with_auth(api_key):
    """Test predictions with authentication"""
    print_test("GET /api/predictions (With Auth)")
    try:
        headers = {"X-API-Key": api_key}
        r = requests.get(f"{API_URL}/api/predictions", headers=headers)
        print(f"Status Code: {r.status_code}")

        if r.status_code == 200:
            predictions = r.json()
            print(f"Number of predictions: {len(predictions)}")

            if len(predictions) > 0:
                # Show first prediction
                print(f"\nSample prediction:")
                print(json.dumps(predictions[0], indent=2))

                # Validate structure
                first = predictions[0]
                assert "ticker" in first, "Should have ticker"
                assert "prediction_label" in first, "Should have prediction_label"
                assert "prob_crash" in first, "Should have prob_crash"
                assert "prob_normal" in first, "Should have prob_normal"
                assert "prob_spike" in first, "Should have prob_spike"
                assert "confidence" in first, "Should have confidence"

                print_success(f"Retrieved {len(predictions)} predictions")
                return True
            else:
                print_info("No predictions in database yet")
                return True
        else:
            print_error(f"Expected 200, got {r.status_code}")
            print(f"Response: {r.text}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_specific_asset(api_key, ticker="SPY"):
    """Test specific asset prediction"""
    print_test(f"GET /api/predictions/{ticker} (Specific Asset)")
    try:
        headers = {"X-API-Key": api_key}
        r = requests.get(f"{API_URL}/api/predictions/{ticker}", headers=headers)
        print(f"Status Code: {r.status_code}")

        if r.status_code == 200:
            prediction = r.json()
            print(f"Response: {json.dumps(prediction, indent=2)}")

            assert prediction["ticker"] == ticker, f"Ticker should be {ticker}"
            assert prediction["prediction_label"] in ["CRASH", "NORMAL", "SPIKE"], "Invalid label"

            print_success(f"Retrieved prediction for {ticker}")
            print_info(f"Prediction: {prediction['prediction_label']}")
            print_info(f"Confidence: {prediction['confidence']}")
            return True
        elif r.status_code == 404:
            print_info(f"No prediction found for {ticker}")
            return True
        else:
            print_error(f"Expected 200 or 404, got {r.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_assets(api_key):
    """Test assets list endpoint"""
    print_test("GET /api/assets (Available Assets)")
    try:
        headers = {"X-API-Key": api_key}
        r = requests.get(f"{API_URL}/api/assets", headers=headers)
        print(f"Status Code: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            print(f"Total assets: {data['total']}")
            print(f"Assets: {', '.join(data['assets'][:10])}...")  # Show first 10

            print_success(f"Retrieved {data['total']} assets")
            return True
        else:
            print_error(f"Expected 200, got {r.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_invalid_ticker(api_key):
    """Test invalid ticker (should return 404)"""
    print_test("GET /api/predictions/INVALID (Invalid Ticker)")
    try:
        headers = {"X-API-Key": api_key}
        r = requests.get(f"{API_URL}/api/predictions/INVALID_TICKER_999", headers=headers)
        print(f"Status Code: {r.status_code}")

        if r.status_code == 404:
            print_success("Correctly returned 404 for invalid ticker")
            return True
        else:
            print_error(f"Expected 404, got {r.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def main():
    """Run all tests"""
    print(f"\n{BLUE}{'='*70}")
    print("NeuroVest API Test Suite")
    print(f"{'='*70}{RESET}\n")
    print(f"Testing API at: {YELLOW}{API_URL}{RESET}\n")

    results = []

    # Test 1: Root endpoint
    results.append(("Root Endpoint", test_root()))

    # Test 2: Health check
    results.append(("Health Check", test_health()))

    # Test 3: Register user
    api_key = test_register()
    if api_key:
        results.append(("User Registration", True))

        # Test 4: No auth (should fail)
        results.append(("Auth Required", test_predictions_no_auth()))

        # Test 5: All predictions
        results.append(("All Predictions", test_predictions_with_auth(api_key)))

        # Test 6: Specific asset
        results.append(("Specific Asset (SPY)", test_specific_asset(api_key, "SPY")))

        # Test 7: Assets list
        results.append(("Assets List", test_assets(api_key)))

        # Test 8: Invalid ticker
        results.append(("Invalid Ticker", test_invalid_ticker(api_key)))
    else:
        results.append(("User Registration", False))
        print_error("Cannot continue tests without API key")

    # Summary
    print(f"\n{BLUE}{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}{RESET}\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{name:.<50} {status}")

    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"Results: {GREEN}{passed}/{total} tests passed{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    if passed == total:
        print(f"{GREEN}✓ All tests passed!{RESET}\n")
        sys.exit(0)
    else:
        print(f"{RED}✗ Some tests failed{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()

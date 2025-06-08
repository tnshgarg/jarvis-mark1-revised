#!/usr/bin/env python3
"""
Quick API Test Script

Test the Mark-1 API endpoints to ensure they're working correctly
"""

import requests
import json
import time
import sys

def test_api(base_url="http://127.0.0.1:8000"):
    """Test the API endpoints"""
    print(f"🧪 Testing Mark-1 API at {base_url}")
    print("="*50)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Health Check
    try:
        print("1️⃣ Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data.get('status', 'unknown')}")
            tests_passed += 1
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        tests_failed += 1
    
    # Test 2: System Status
    try:
        print("2️⃣ Testing system status...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ System: {data.get('service', 'unknown')}")
            tests_passed += 1
        else:
            print(f"   ❌ System status failed: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ System status error: {e}")
        tests_failed += 1
    
    # Test 3: List Agents
    try:
        print("3️⃣ Testing agents endpoint...")
        response = requests.get(f"{base_url}/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Agents: {data.get('total', 0)} found")
            tests_passed += 1
        else:
            print(f"   ❌ Agents endpoint failed: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Agents endpoint error: {e}")
        tests_failed += 1
    
    # Test 4: Create Agent (should validate schema)
    try:
        print("4️⃣ Testing agent creation (invalid data)...")
        agent_data = {
            "name": "test_agent",
            "framework": "invalid_framework"  # This should trigger validation error
        }
        response = requests.post(f"{base_url}/agents", json=agent_data, timeout=5)
        if response.status_code == 422:  # Validation error expected
            print("   ✅ Schema validation working (422 for invalid data)")
            tests_passed += 1
        else:
            print(f"   ❌ Schema validation not working: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Agent creation test error: {e}")
        tests_failed += 1
    
    # Test 5: Metrics
    try:
        print("5️⃣ Testing metrics endpoint...")
        response = requests.get(f"{base_url}/metrics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Metrics: {len(data)} metric categories")
            tests_passed += 1
        else:
            print(f"   ❌ Metrics endpoint failed: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Metrics endpoint error: {e}")
        tests_failed += 1
    
    # Test 6: OpenAPI Documentation
    try:
        print("6️⃣ Testing OpenAPI docs...")
        response = requests.get(f"{base_url}/openapi.json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            paths = data.get('paths', {})
            print(f"   ✅ OpenAPI: {len(paths)} endpoints documented")
            tests_passed += 1
        else:
            print(f"   ❌ OpenAPI docs failed: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ OpenAPI docs error: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*50)
    print(f"📊 Test Results: {tests_passed} passed, {tests_failed} failed")
    
    if tests_failed == 0:
        print("🎉 All tests passed! Your API is working perfectly!")
        print(f"\n🌐 Access your API at: {base_url}")
        print(f"📚 View docs at: {base_url}/docs")
        return True
    else:
        print("⚠️ Some tests failed. Check the server logs for details.")
        return False

def find_api_port():
    """Try to find which port the API is running on"""
    for port in [8000, 8001, 8002, 8003, 8004, 8005]:
        try:
            url = f"http://127.0.0.1:{port}"
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return url
        except:
            continue
    return None

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        # Try to auto-detect API
        print("🔍 Looking for running API server...")
        base_url = find_api_port()
        if not base_url:
            print("❌ No API server found. Please start the server first with:")
            print("   python start_api_smart.py")
            return 1
        else:
            print(f"✅ Found API server at {base_url}")
    
    success = test_api(base_url)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 
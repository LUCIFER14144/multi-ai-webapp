#!/usr/bin/env python3
"""
Test script for the Multi-AI Web App
"""
import asyncio
import json
from app.main import app, TaskRequest, APIKeys, AIProvider

async def test_validation():
    """Test input validation"""
    print("Testing input validation...")
    
    # Test valid request
    try:
        request = TaskRequest(
            prompt="Test prompt that is long enough",
            api_keys=APIKeys(openai="sk-test123456789"),
            provider=AIProvider.OPENAI
        )
        print("✅ Valid request passed validation")
    except Exception as e:
        print(f"❌ Valid request failed: {e}")
    
    # Test invalid prompt (too short)
    try:
        request = TaskRequest(
            prompt="Hi",
            api_keys=APIKeys(openai="sk-test123456789"),
            provider=AIProvider.OPENAI
        )
        print("❌ Short prompt should have failed validation")
    except Exception as e:
        print(f"✅ Short prompt correctly rejected: {e}")
    
    # Test invalid API key (too short)
    try:
        request = TaskRequest(
            prompt="Test prompt that is long enough",
            api_keys=APIKeys(openai="short"),
            provider=AIProvider.OPENAI
        )
        print("❌ Short API key should have failed validation")
    except Exception as e:
        print(f"✅ Short API key correctly rejected: {e}")

def test_provider_configs():
    """Test provider configurations"""
    print("\nTesting provider configurations...")
    
    from app.main import PROVIDER_CONFIGS, AIProvider
    
    for provider in AIProvider:
        config = PROVIDER_CONFIGS[provider]
        print(f"✅ {provider.value}:")
        print(f"   Base URL: {config['base_url']}")
        print(f"   Default Model: {config['default_model']}")
        print(f"   Available Models: {config['models']}")

def test_api_structure():
    """Test API endpoints are properly defined"""
    print("\nTesting API structure...")
    
    # Check if endpoints exist
    routes = [route.path for route in app.routes]
    expected_routes = ["/", "/health", "/api/generate", "/api/providers"]
    
    for route in expected_routes:
        if route in routes:
            print(f"✅ Route {route} exists")
        else:
            print(f"❌ Route {route} missing")

if __name__ == "__main__":
    print("🧪 Multi-AI Web App - Test Suite\n")
    
    # Run sync tests
    test_provider_configs()
    test_api_structure()
    
    # Run async tests
    asyncio.run(test_validation())
    
    print("\n✅ Test suite completed!")
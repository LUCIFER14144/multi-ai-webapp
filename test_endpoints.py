#!/usr/bin/env python3
"""
Test endpoints of the Multi-AI Web App
"""
import httpx
import asyncio
import json

async def test_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing API Endpoints\n")
        
        # Test 1: Health endpoint
        print("1. Testing /health endpoint...")
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ✅ Response: {response.json()}")
            else:
                print(f"   ❌ Status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 2: Providers endpoint
        print("\n2. Testing /api/providers endpoint...")
        try:
            response = await client.get(f"{base_url}/api/providers")
            if response.status_code == 200:
                print(f"   ✅ Status: {response.status_code}")
                data = response.json()
                print(f"   ✅ Available providers: {list(data['providers'].keys())}")
                for provider, config in data['providers'].items():
                    print(f"      - {provider}: {config['models']}")
            else:
                print(f"   ❌ Status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 3: Root endpoint (serves frontend)
        print("\n3. Testing / (root) endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            if response.status_code == 200:
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ✅ Content-Type: {response.headers.get('content-type')}")
                if 'html' in response.headers.get('content-type', '').lower():
                    print(f"   ✅ Serving HTML frontend")
            else:
                print(f"   ❌ Status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 4: API validation (invalid request)
        print("\n4. Testing /api/generate validation (should fail)...")
        try:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "prompt": "Hi",  # Too short
                    "api_keys": {"openai": "short"},  # Invalid key
                    "provider": "openai"
                }
            )
            if response.status_code == 422:  # Validation error
                print(f"   ✅ Status: {response.status_code} (Validation working)")
                error_data = response.json()
                print(f"   ✅ Validation errors detected: {len(error_data.get('detail', []))} errors")
            else:
                print(f"   ⚠️  Status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("\n✅ Endpoint tests completed!")

if __name__ == "__main__":
    print("Starting endpoint tests...")
    print("Make sure the server is running on http://localhost:8000\n")
    asyncio.run(test_endpoints())

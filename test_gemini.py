import httpx
import sys

# Test Gemini API with your key
api_key = input("Enter your Gemini API key: ")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}
payload = {
    "contents": [{
        "parts": [{"text": "Say hello in one word"}]
    }]
}

print(f"\nTesting URL: {url[:80]}...")
print(f"Payload: {payload}\n")

try:
    response = httpx.post(url, headers=headers, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}\n")
    
    if response.status_code == 200:
        print("✅ SUCCESS! Your API key works!")
        data = response.json()
        if "candidates" in data:
            print(f"Model response: {data['candidates'][0]['content']['parts'][0]['text']}")
    elif response.status_code == 400:
        print("❌ ERROR 400: Invalid API key or request format")
        print("Get a new key from: https://aistudio.google.com/app/apikey")
    elif response.status_code == 404:
        print("❌ ERROR 404: API key not found or model doesn't exist")
        print("Get a new key from: https://aistudio.google.com/app/apikey")
    else:
        print(f"❌ ERROR {response.status_code}: Unexpected error")
        
except Exception as e:
    print(f"❌ Exception: {e}")

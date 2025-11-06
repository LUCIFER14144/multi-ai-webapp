#!/usr/bin/env python3
"""
Comprehensive deployment readiness test
"""
import subprocess
import time
import sys
import os

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_imports():
    """Test if all required modules can be imported"""
    print("📦 Testing Python Module Imports...")
    modules = [
        'fastapi',
        'uvicorn',
        'httpx',
        'tenacity',
        'pydantic',
        'starlette'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            all_ok = False
    
    return all_ok

def test_app_import():
    """Test if the app can be imported without errors"""
    print("\n🔧 Testing App Import...")
    try:
        from app.main import app, PROVIDER_CONFIGS, AIProvider
        print("   ✅ App module imported successfully")
        print(f"   ✅ Found {len(PROVIDER_CONFIGS)} providers configured")
        return True
    except Exception as e:
        print(f"   ❌ Failed to import app: {e}")
        return False

def test_file_structure():
    """Test if all necessary files exist"""
    print("\n📁 Testing File Structure...")
    
    files = [
        'app/main.py',
        'app/frontend/index.html',
        'app/frontend/app.js',
        'requirements.txt',
        'Dockerfile',
        'README.md'
    ]
    
    all_ok = True
    for file_path in files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size} bytes)")
        else:
            print(f"   ❌ {file_path} NOT FOUND")
            all_ok = False
    
    return all_ok

def test_syntax():
    """Test Python syntax"""
    print("\n🔍 Testing Python Syntax...")
    
    success, stdout, stderr = run_command("python -m py_compile app/main.py")
    
    if success:
        print("   ✅ app/main.py syntax is valid")
        return True
    else:
        print(f"   ❌ Syntax error: {stderr}")
        return False

def test_server_start():
    """Test if server can start"""
    print("\n🚀 Testing Server Startup...")
    print("   Starting uvicorn server (will timeout after 5 seconds)...")
    
    try:
        # Start server in background
        proc = subprocess.Popen(
            ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if process is still running
        if proc.poll() is None:
            print("   ✅ Server started successfully")
            proc.terminate()
            proc.wait(timeout=2)
            return True
        else:
            stdout, stderr = proc.communicate()
            print(f"   ❌ Server failed to start")
            print(f"   Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error starting server: {e}")
        try:
            proc.terminate()
        except:
            pass
        return False

def test_provider_configs():
    """Test provider configurations"""
    print("\n🤖 Testing AI Provider Configurations...")
    
    try:
        from app.main import PROVIDER_CONFIGS, AIProvider
        
        for provider in AIProvider:
            config = PROVIDER_CONFIGS[provider]
            print(f"   ✅ {provider.value}:")
            print(f"      Models: {', '.join(config['models'])}")
            print(f"      Default: {config['default_model']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 MULTI-AI WEB APP - DEPLOYMENT READINESS TEST")
    print("=" * 60)
    print()
    
    results = {
        "File Structure": test_file_structure(),
        "Module Imports": test_imports(),
        "App Import": test_app_import(),
        "Python Syntax": test_syntax(),
        "Provider Configs": test_provider_configs(),
        "Server Startup": test_server_start(),
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        print("=" * 60)
        print("\n📝 Next Steps:")
        print("   1. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        print("   2. Open: http://localhost:8000")
        print("   3. Enter your API keys in the frontend")
        print("   4. Test with a sample prompt")
        print("\n🐳 Docker Deployment:")
        print("   docker build -t multi-ai-app .")
        print("   docker run -p 8000:8000 multi-ai-app")
        return 0
    else:
        print("❌ SOME TESTS FAILED - NOT READY FOR DEPLOYMENT")
        print("=" * 60)
        print("\n⚠️  Please fix the failed tests before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

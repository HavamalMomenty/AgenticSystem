"""
Simple test script to verify the Property Analysis System setup
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        from main import PropertyAnalysisSystem
        print("✅ Main module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration"""
    print("🔧 Testing configuration...")
    
    try:
        import config
        print("✅ Configuration loaded successfully")
        
        # Check if paths exist
        if config.DATABASE_DIR.exists():
            print("✅ Database directory found")
        else:
            print("⚠️ Database directory not found")
            
        if config.UNSTRUCTURED_DATA_DIR.exists():
            print("✅ Unstructured data directory found")
        else:
            print("⚠️ Unstructured data directory not found")
            
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_api_key():
    """Test API key setup"""
    print("🔑 Testing API key...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    elif api_key == "your_openai_api_key_here":
        print("⚠️ API key not set (still using placeholder)")
        return False
    else:
        print("✅ API key found")
        return True

def main():
    """Run all tests"""
    print("🚀 Property Analysis System - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("API Key", test_api_key)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("Next steps:")
        print("1. Make sure your OpenAI API key is set in .env file")
        print("2. Run: python main.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        
        if not results.get("API Key", False):
            print("\n💡 To fix API key issue:")
            print("1. Edit .env file")
            print("2. Replace 'your_openai_api_key_here' with your actual OpenAI API key")
            
        if not results.get("Imports", False):
            print("\n💡 To fix import issues:")
            print("1. Run: pip install -r requirements.txt")
            print("2. Or run: python setup.py")

if __name__ == "__main__":
    main()

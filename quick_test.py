"""
Quick test to verify the system is working
"""

import os
from dotenv import load_dotenv
from pathlib import Path

def test_setup():
    """Test the basic setup"""
    print("🧪 Testing System Setup")
    print("=" * 30)
    
    # Test environment
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key and api_key != "your_openai_api_key_here":
        print("✅ API Key loaded")
    else:
        print("❌ API Key not found")
        return False
    
    # Test paths
    base_dir = Path(__file__).parent
    database_dir = base_dir / "database"
    
    if database_dir.exists():
        print("✅ Database directory found")
    else:
        print("❌ Database directory not found")
        return False
    
    # Test OpenAI import
    try:
        import openai
        print("✅ OpenAI package available")
    except ImportError:
        print("❌ OpenAI package not available")
        return False
    
    print("\n🎉 All checks passed!")
    return True

def quick_analysis():
    """Run a quick analysis"""
    print("\n🔍 Running Quick Analysis...")
    
    try:
        from simple_analysis import SimplePropertyAnalysis
        system = SimplePropertyAnalysis()
        
        # Load just a few documents for testing
        system.load_documents()
        print(f"✅ Loaded {len(system.documents)} documents")
        
        if len(system.documents) > 0:
            print("📄 Sample document:")
            sample = system.documents[0]
            print(f"   Name: {sample['name']}")
            print(f"   Content preview: {sample['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if test_setup():
        quick_analysis()
    else:
        print("\n⚠️ Setup issues found. Please fix them before running the full analysis.")

"""
Minimal working example to test LlamaIndex setup
"""

def test_basic_import():
    """Test basic imports step by step"""
    print("Testing basic imports...")
    
    try:
        print("1. Testing llama_index.core...")
        import llama_index.core
        print("   ✅ llama_index.core imported")
        
        print("2. Testing VectorStoreIndex...")
        from llama_index.core import VectorStoreIndex
        print("   ✅ VectorStoreIndex imported")
        
        print("3. Testing SimpleDirectoryReader...")
        from llama_index.core import SimpleDirectoryReader
        print("   ✅ SimpleDirectoryReader imported")
        
        print("4. Testing Settings...")
        from llama_index.core import Settings
        print("   ✅ Settings imported")
        
        print("5. Testing OpenAI LLM...")
        from llama_index.llms.openai import OpenAI
        print("   ✅ OpenAI LLM imported")
        
        print("6. Testing OpenAI Embedding...")
        from llama_index.embeddings.openai import OpenAIEmbedding
        print("   ✅ OpenAI Embedding imported")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_minimal_setup():
    """Test minimal setup without full system"""
    print("\nTesting minimal setup...")
    
    try:
        from llama_index.core import Settings
        from llama_index.llms.openai import OpenAI
        
        # Test basic configuration
        print("1. Configuring OpenAI LLM...")
        llm = OpenAI(model="gpt-4o-mini", api_key="test_key")
        print("   ✅ OpenAI LLM configured")
        
        print("2. Setting up Settings...")
        Settings.llm = llm
        print("   ✅ Settings configured")
        
        print("\n🎉 Minimal setup successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 LlamaIndex Minimal Test")
    print("=" * 40)
    
    success1 = test_basic_import()
    success2 = test_minimal_setup() if success1 else False
    
    if success1 and success2:
        print("\n✅ All tests passed! LlamaIndex is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

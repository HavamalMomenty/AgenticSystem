#!/usr/bin/env python3
"""
Test script to check LlamaIndex imports
"""

print("Testing LlamaIndex imports...")

try:
    # Test basic imports
    from llama_index.core import VectorStoreIndex
    print("✅ VectorStoreIndex imported successfully")
except ImportError as e:
    print(f"❌ VectorStoreIndex import failed: {e}")

try:
    from llama_index.core import SimpleDirectoryReader
    print("✅ SimpleDirectoryReader imported successfully")
except ImportError as e:
    print(f"❌ SimpleDirectoryReader import failed: {e}")

try:
    from llama_index.core import Settings
    print("✅ Settings imported successfully")
except ImportError as e:
    print(f"❌ Settings import failed: {e}")

try:
    from llama_index.core import Document
    print("✅ Document imported successfully")
except ImportError as e:
    print(f"❌ Document import failed: {e}")

try:
    from llama_index.core import StorageContext
    print("✅ StorageContext imported successfully")
except ImportError as e:
    print(f"❌ StorageContext import failed: {e}")

try:
    from llama_index.core import load_index_from_storage
    print("✅ load_index_from_storage imported successfully")
except ImportError as e:
    print(f"❌ load_index_from_storage import failed: {e}")

# Test node parser
try:
    from llama_index.core.node_parser import SimpleNodeParser
    print("✅ SimpleNodeParser imported successfully")
except ImportError as e:
    print(f"❌ SimpleNodeParser import failed: {e}")

# Test OpenAI integrations
try:
    from llama_index.llms.openai import OpenAI
    print("✅ OpenAI LLM imported successfully")
except ImportError as e:
    print(f"❌ OpenAI LLM import failed: {e}")

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    print("✅ OpenAIEmbedding imported successfully")
except ImportError as e:
    print(f"❌ OpenAIEmbedding import failed: {e}")

# Test query engine
try:
    from llama_index.core.query_engine import RetrieverQueryEngine
    print("✅ RetrieverQueryEngine imported successfully")
except ImportError as e:
    print(f"❌ RetrieverQueryEngine import failed: {e}")

try:
    from llama_index.core.retrievers import VectorIndexRetriever
    print("✅ VectorIndexRetriever imported successfully")
except ImportError as e:
    print(f"❌ VectorIndexRetriever import failed: {e}")

try:
    from llama_index.core.postprocessor import SimilarityPostprocessor
    print("✅ SimilarityPostprocessor imported successfully")
except ImportError as e:
    print(f"❌ SimilarityPostprocessor import failed: {e}")

print("\nImport testing completed!")

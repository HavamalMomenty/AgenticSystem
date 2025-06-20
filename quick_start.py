"""
Quick Start Example for Property Analysis System
Run this after setting up your OpenAI API key in .env file
"""

import os
from pathlib import Path
from main import PropertyAnalysisSystem

def quick_demo():
    """
    Quick demonstration of the system
    """
    print("🏠 Property Analysis System - Quick Demo")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("❌ Please set your OpenAI API key in the .env file first!")
        print("   Edit .env and replace 'your_openai_api_key_here' with your actual API key")
        return
    
    try:
        # Initialize system
        print("\n🔧 Initializing system...")
        system = PropertyAnalysisSystem()
        
        # Build index
        print("📚 Building knowledge base from your documents...")
        system.build_index(force_rebuild=False)
        
        # Setup query engine
        print("🔍 Setting up query engine...")
        system.setup_query_engine()
        
        # Test with a simple query
        print("\n💬 Testing with a sample query...")
        sample_query = "What types of properties are available in the database?"
        response = system.query_database(sample_query)
        print(f"Query: {sample_query}")
        print(f"Response: {response[:200]}...")
        
        # Generate full report
        print("\n📄 Generating comprehensive property analysis report...")
        report = system.generate_report()
        
        # Find the most recent report
        reports_dir = Path("reports")
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.txt"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                print(f"✅ Report generated successfully!")
                print(f"📁 Report saved to: {latest_report}")
                print(f"📏 Report length: {len(report)} characters")
                
                # Show preview
                print("\n📖 Report Preview (first 500 characters):")
                print("-" * 50)
                print(report[:500] + "..." if len(report) > 500 else report)
            else:
                print("❌ Report file not found")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your OpenAI API key is correct")
        print("2. Check that you have internet connection")
        print("3. Verify that your database files exist")
        print("4. Check the logs directory for detailed error information")


def interactive_query_demo():
    """
    Interactive query demonstration
    """
    print("\n🔍 Interactive Query Mode")
    print("Type 'exit' to quit, or ask any question about your property data")
    print("-" * 50)
    
    try:
        system = PropertyAnalysisSystem()
        system.build_index()
        system.setup_query_engine()
        
        while True:
            query = input("\n💬 Your question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            try:
                response = system.query_database(query)
                print(f"\n🤖 Response:\n{response}\n")
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                
    except Exception as e:
        print(f"❌ Error initializing interactive mode: {e}")


if __name__ == "__main__":
    quick_demo()
    
    # Uncomment the line below for interactive mode
    # interactive_query_demo()

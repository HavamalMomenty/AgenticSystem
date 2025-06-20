"""
Advanced usage examples and alternative methods for the Property Analysis System
"""

from main import PropertyAnalysisSystem
from config import *
import json
from pathlib import Path
from typing import Dict, Any, List


class AdvancedPropertyAnalysis(PropertyAnalysisSystem):
    """
    Extended version with advanced features and alternative methods
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.conversation_history = []
    
    def multi_step_analysis(self, input_file: Path = INPUT_FILE) -> Dict[str, Any]:
        """
        Perform multi-step analysis with different aspects
        
        Returns:
            Dictionary containing different analysis aspects
        """
        analysis_steps = {
            "market_overview": "Provide a comprehensive market overview for this property type and location",
            "financial_analysis": "Perform detailed financial analysis including cash flow projections, IRR, and NPV calculations",
            "risk_assessment": "Identify and analyze all potential risks associated with this investment",
            "comparable_analysis": "Find and analyze comparable properties from the database",
            "recommendation": "Provide final investment recommendation with supporting rationale"
        }
        
        results = {}
        input_content = self.load_input_document(input_file)
        
        for step_name, step_prompt in analysis_steps.items():
            full_prompt = f"""
            {step_prompt}
            
            Property to analyze: {input_file.name}
            Property details: {input_content[:1000]}...
            
            Previous analysis context: {json.dumps(results, indent=2) if results else "None"}
            
            Provide detailed analysis for this specific aspect:
            """
            
            response = self.query_database(full_prompt)
            results[step_name] = response
            
            # Add to conversation history
            self.conversation_history.append({
                "step": step_name,
                "prompt": step_prompt,
                "response": response
            })
        
        return results
    
    def interactive_analysis(self) -> None:
        """
        Interactive analysis session with the user
        """
        print("🏠 Interactive Property Analysis System")
        print("=" * 50)
        print("Available commands:")
        print("- analyze: Perform full analysis")
        print("- query <your question>: Ask specific questions")
        print("- compare: Compare with similar properties")
        print("- risks: Focus on risk analysis")
        print("- financials: Deep dive into financial analysis")
        print("- exit: Exit the session")
        print()
        
        while True:
            try:
                user_input = input("🔍 Enter command: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'analyze':
                    self._interactive_full_analysis()
                elif user_input.startswith('query '):
                    question = user_input[6:]
                    response = self.query_database(question)
                    print(f"\n📊 Response:\n{response}\n")
                elif user_input.lower() == 'compare':
                    self._interactive_comparison()
                elif user_input.lower() == 'risks':
                    self._interactive_risk_analysis()
                elif user_input.lower() == 'financials':
                    self._interactive_financial_analysis()
                else:
                    print("❌ Unknown command. Type 'exit' to quit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Session ended by user.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _interactive_full_analysis(self):
        """Perform interactive full analysis"""
        print("\n🔄 Performing comprehensive analysis...")
        results = self.multi_step_analysis()
        
        for step, result in results.items():
            print(f"\n📋 {step.replace('_', ' ').title()}:")
            print("-" * 40)
            print(result[:500] + "..." if len(result) > 500 else result)
    
    def _interactive_comparison(self):
        """Interactive comparison analysis"""
        prompt = """
        Find similar properties in the database and provide a detailed comparison analysis.
        Include:
        1. Similar property types and locations
        2. Comparable investment metrics
        3. Market positioning
        4. Key differences and similarities
        """
        response = self.query_database(prompt)
        print(f"\n🏘️ Comparable Properties Analysis:\n{response}\n")
    
    def _interactive_risk_analysis(self):
        """Interactive risk analysis"""
        prompt = """
        Provide comprehensive risk analysis including:
        1. Market risks
        2. Financial risks
        3. Operational risks
        4. Regulatory risks
        5. Risk mitigation strategies
        """
        response = self.query_database(prompt)
        print(f"\n⚠️ Risk Analysis:\n{response}\n")
    
    def _interactive_financial_analysis(self):
        """Interactive financial analysis"""
        prompt = """
        Provide detailed financial analysis including:
        1. Cash flow projections
        2. Return calculations (IRR, NPV, Cash-on-Cash)
        3. Sensitivity analysis
        4. Financing considerations
        5. Exit strategy financial implications
        """
        response = self.query_database(prompt)
        print(f"\n💰 Financial Analysis:\n{response}\n")


def alternative_openai_configurations():
    """
    Suggest alternative OpenAI model configurations and parameters
    """
    
    configurations = {
        "Conservative Analysis": {
            "temperature": 0.1,
            "top_p": 0.8,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
            "max_tokens": 3000,
            "description": "More factual, less creative responses for financial analysis"
        },
        
        "Creative Insights": {
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.3,
            "max_tokens": 4000,
            "description": "More creative responses for identifying opportunities and innovative strategies"
        },
        
        "Detailed Technical": {
            "temperature": 0.3,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.0,
            "max_tokens": 6000,
            "description": "Balanced approach for comprehensive technical analysis"
        },
        
        "Quick Summary": {
            "temperature": 0.2,
            "top_p": 0.8,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.2,
            "max_tokens": 1500,
            "description": "Concise, to-the-point analysis for executive summaries"
        }
    }
    
    return configurations


def advanced_llamaindex_features():
    """
    Demonstrate advanced LlamaIndex features for better results
    """
    
    features = {
        "Custom Node Parsers": {
            "description": "Use specialized parsers for different document types",
            "implementation": """
            from llama_index.core.node_parser import (
                SimpleNodeParser, 
                SentenceWindowNodeParser,
                SemanticSplitterNodeParser
            )
            
            # For financial documents
            financial_parser = SimpleNodeParser.from_defaults(
                chunk_size=512,
                chunk_overlap=50
            )
            
            # For text-heavy documents
            semantic_parser = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95
            )
            """
        },
        
        "Metadata Filtering": {
            "description": "Filter documents by metadata for targeted analysis",
            "implementation": """
            from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
            
            # Filter by document type
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="file_type", value=".xlsx"),
                    MetadataFilter(key="directory", value="financial_data")
                ]
            )
            
            retriever = VectorIndexRetriever(
                index=index,
                filters=filters,
                similarity_top_k=5
            )
            """
        },
        
        "Response Synthesis": {
            "description": "Advanced response synthesis modes",
            "implementation": """
            from llama_index.core import get_response_synthesizer
            from llama_index.core.response_synthesizers import ResponseMode
            
            # Tree summarize for comprehensive analysis
            synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.TREE_SUMMARIZE,
                use_async=True
            )
            
            # Compact for quick analysis
            compact_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT
            )
            """
        },
        
        "Custom Retrievers": {
            "description": "Implement custom retrieval strategies",
            "implementation": """
            from llama_index.core.retrievers import BaseRetriever
            
            class PropertyTypeRetriever(BaseRetriever):
                def _retrieve(self, query_bundle):
                    # Custom logic to retrieve property-specific documents
                    # Based on property type, location, etc.
                    pass
            """
        }
    }
    
    return features


def main_advanced_demo():
    """
    Demonstrate advanced features
    """
    print("🚀 Advanced Property Analysis System Demo")
    print("=" * 50)
    
    # Show alternative configurations
    print("\n📊 Alternative OpenAI Configurations:")
    configs = alternative_openai_configurations()
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Temperature: {config['temperature']}")
        print(f"  Max Tokens: {config['max_tokens']}")
    
    # Show advanced features
    print("\n🔧 Advanced LlamaIndex Features:")
    features = advanced_llamaindex_features()
    for name, feature in features.items():
        print(f"\n{name}:")
        print(f"  {feature['description']}")
    
    # Interactive demo (commented out for now)
    # try:
    #     system = AdvancedPropertyAnalysis()
    #     system.build_index()
    #     system.setup_query_engine()
    #     system.interactive_analysis()
    # except Exception as e:
    #     print(f"Demo error: {e}")


if __name__ == "__main__":
    main_advanced_demo()

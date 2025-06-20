"""
Simple Property Analysis System - Working Version
Compatible with current LlamaIndex installation
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import LlamaIndex components
try:
    logger.info("Attempting to import LlamaIndex components...")
    
    # Import what we can from llama_index
    import llama_index
    from llama_index import VectorStoreIndex, SimpleDirectoryReader
    
    logger.info("✅ Basic LlamaIndex components imported successfully")
    HAS_LLAMAINDEX = True
    
except ImportError as e:
    logger.error(f"❌ LlamaIndex import failed: {e}")
    HAS_LLAMAINDEX = False


# Configuration
BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"
UNSTRUCTURED_DATA_DIR = DATABASE_DIR / "Unstructered_data"
EXAMPLE_OUTPUT_DIR = DATABASE_DIR / "Evaluate_property_example"
INPUT_DIR = DATABASE_DIR / "Input"
EXAMPLE_OUTPUT_FILE = EXAMPLE_OUTPUT_DIR / "IC_output_carlsbergbyen.pdf"
INPUT_FILE = INPUT_DIR / "Birkehuset_IM.pdf"
STORAGE_DIR = BASE_DIR / "storage"
REPORTS_DIR = BASE_DIR / "reports"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# Analysis prompt
ANALYSIS_PROMPT = """
You are a real estate investment analysis expert. Based on the provided documents, 
generate a comprehensive investment committee (IC) report.

The report should include:
1. Executive Summary
2. Property Overview
3. Financial Analysis
4. Market Analysis
5. Risk Assessment
6. Investment Recommendation

Provide specific, data-driven insights and recommendations based on the available information.
"""


class SimplePropertyAnalysis:
    """
    Simplified property analysis system without complex LlamaIndex dependencies
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the system"""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            raise ValueError("Please set your OpenAI API key in the .env file")
        
        self.documents = []
        self.index = None
        
        logger.info("SimplePropertyAnalysis initialized")
    
    def load_documents(self, data_dir: Path = UNSTRUCTURED_DATA_DIR) -> List[str]:
        """
        Load and read documents from directory
        
        Args:
            data_dir: Directory containing documents
            
        Returns:
            List of document contents
        """
        logger.info(f"Loading documents from: {data_dir}")
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        documents = []
        supported_extensions = ['.pdf', '.txt', '.docx', '.xlsx']
        
        # Find all supported files
        for ext in supported_extensions:
            files = list(data_dir.rglob(f"*{ext}"))
            logger.info(f"Found {len(files)} {ext} files")
            
            for file_path in files:
                try:
                    content = self._read_file(file_path)
                    if content:
                        documents.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'content': content[:5000]  # Limit content size
                        })
                        logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
        
        self.documents = documents
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _read_file(self, file_path: Path) -> str:
        """Read content from a file"""
        try:
            if file_path.suffix.lower() == '.pdf':
                # Try to read PDF
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except Exception:
                    return f"PDF file: {file_path.name} (content extraction failed)"
            
            elif file_path.suffix.lower() in ['.txt']:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            
            elif file_path.suffix.lower() in ['.docx']:
                try:
                    from docx import Document
                    doc = Document(file_path)
                    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                except Exception:
                    return f"Word document: {file_path.name} (content extraction failed)"
            
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path, sheet_name=None)
                    content = ""
                    for sheet_name, sheet_df in df.items():
                        content += f"\nSheet: {sheet_name}\n"
                        content += sheet_df.to_string()[:2000] + "\n"
                    return content
                except Exception:
                    return f"Excel file: {file_path.name} (content extraction failed)"
            
            else:
                return f"Unsupported file type: {file_path.name}"
                
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return ""
    
    def build_knowledge_base(self):
        """Build a simple knowledge base from documents"""
        if not self.documents:
            self.load_documents()
        
        # Create a simple text-based knowledge base
        knowledge_text = ""
        for doc in self.documents:
            knowledge_text += f"\n--- Document: {doc['name']} ---\n"
            knowledge_text += doc['content']
            knowledge_text += "\n" + "="*50 + "\n"
        
        self.knowledge_base = knowledge_text
        logger.info("Knowledge base built successfully")
    
    def generate_report_with_openai(
        self,
        input_file: Path = INPUT_FILE,
        example_file: Path = EXAMPLE_OUTPUT_FILE,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate report using OpenAI directly
        """
        logger.info("Generating report with OpenAI...")
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Load input and example documents
        input_content = ""
        example_content = ""
        
        if input_file.exists():
            input_content = self._read_file(input_file)[:2000]
        
        if example_file.exists():
            example_content = self._read_file(example_file)[:2000]
        
        # Build context from knowledge base
        context = ""
        if hasattr(self, 'knowledge_base'):
            context = self.knowledge_base[:8000]  # Limit context size
        
        # Prepare prompt
        prompt = custom_prompt or ANALYSIS_PROMPT
        
        full_prompt = f"""
        {prompt}
        
        REFERENCE DOCUMENTS AND DATA:
        {context}
        
        EXAMPLE OUTPUT FORMAT:
        {example_content}
        
        INPUT DOCUMENT TO ANALYZE:
        File: {input_file.name}
        Content: {input_content}
        
        Generate a comprehensive investment analysis report based on the above information:
        """
        
        # Make OpenAI API call
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert real estate investment analyst."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            report_content = response.choices[0].message.content
            logger.info("Report generated successfully")
            return report_content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def save_report(self, report_content: str, output_file: Optional[Path] = None) -> Path:
        """Save report to file"""
        if output_file is None:
            REPORTS_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = REPORTS_DIR / f"property_analysis_report_{timestamp}.txt"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {output_file}")
        return output_file
    
    def analyze_property(self) -> str:
        """Complete analysis workflow"""
        logger.info("Starting property analysis...")
        
        # Build knowledge base
        self.build_knowledge_base()
        
        # Generate report
        report = self.generate_report_with_openai()
        
        # Save report
        output_file = self.save_report(report)
        
        logger.info(f"Analysis completed! Report saved to: {output_file}")
        return report


def main():
    """Main function"""
    print("🏠 Simple Property Analysis System")
    print("=" * 50)
    
    try:
        # Check if API key is set
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            print("❌ Please set your OpenAI API key in .env file!")
            print("Edit .env and replace 'your_openai_api_key_here' with your actual API key")
            return
        
        # Initialize system
        system = SimplePropertyAnalysis()
        
        # Run analysis
        report = system.analyze_property()
        
        # Show preview
        print("\n" + "="*60)
        print("📄 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nReport Preview (first 800 characters):")
        print("-" * 40)
        print(report[:800] + "..." if len(report) > 800 else report)
        print("\n✅ Full report saved in the 'reports' directory")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()

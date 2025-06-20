#!/usr/bin/env python3
"""
Property Analysis System - Complete Working Solution
This is a standalone script that uses OpenAI directly to analyze property documents.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import json

def setup_environment():
    """Setup the environment and check requirements"""
    print("🔧 Setting up environment...")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("⚠️ python-dotenv not found, using system environment")
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ OpenAI API key not found!")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    
    print("✅ OpenAI API key found")
    
    # Check OpenAI package
    try:
        import openai
        print("✅ OpenAI package available")
    except ImportError:
        print("❌ OpenAI package not found!")
        print("Install with: pip install openai")
        return False
    
    return True

def read_pdf_file(file_path: Path) -> str:
    """Read content from a PDF file"""
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        return f"PDF file: {file_path.name} (PyPDF2 not available for content extraction)"
    except Exception as e:
        return f"PDF file: {file_path.name} (Error: {e})"

def read_text_file(file_path: Path) -> str:
    """Read content from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        return f"Text file: {file_path.name} (Error: {e})"

def read_excel_file(file_path: Path) -> str:
    """Read content from an Excel file"""
    try:
        import pandas as pd
        df = pd.read_excel(file_path, sheet_name=None)
        content = ""
        for sheet_name, sheet_df in df.items():
            content += f"\n--- Sheet: {sheet_name} ---\n"
            content += sheet_df.to_string()[:1500] + "\n"
        return content
    except ImportError:
        return f"Excel file: {file_path.name} (pandas not available)"
    except Exception as e:
        return f"Excel file: {file_path.name} (Error: {e})"

def read_word_file(file_path: Path) -> str:
    """Read content from a Word file"""
    try:
        from docx import Document
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except ImportError:
        return f"Word file: {file_path.name} (python-docx not available)"
    except Exception as e:
        return f"Word file: {file_path.name} (Error: {e})"

def load_documents_from_directory(directory: Path) -> List[Dict]:
    """Load all documents from a directory"""
    print(f"📁 Loading documents from: {directory}")
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    documents = []
    
    # File type handlers
    handlers = {
        '.pdf': read_pdf_file,
        '.txt': read_text_file,
        '.xlsx': read_excel_file,
        '.xls': read_excel_file,
        '.docx': read_word_file,
        '.doc': read_word_file,
    }
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in handlers:
            print(f"   📄 Loading: {file_path.name}")
            
            handler = handlers[file_path.suffix.lower()]
            content = handler(file_path)
            
            documents.append({
                'name': file_path.name,
                'path': str(file_path),
                'type': file_path.suffix.lower(),
                'content': content[:3000]  # Limit content length
            })
    
    print(f"✅ Loaded {len(documents)} documents")
    return documents

def generate_analysis_with_openai(
    documents: List[Dict],
    input_file: Optional[Path] = None,
    example_file: Optional[Path] = None,
    custom_prompt: Optional[str] = None
) -> str:
    """Generate analysis using OpenAI"""
    print("🤖 Generating analysis with OpenAI...")
    
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception as e:
        raise Exception(f"Failed to initialize OpenAI client: {e}")
    
    # Prepare context from documents
    context = "REFERENCE DOCUMENTS:\n\n"
    for doc in documents[:10]:  # Limit to first 10 documents
        context += f"--- {doc['name']} ---\n"
        context += doc['content'][:1000] + "\n\n"
    
    # Load input document if provided
    input_content = ""
    if input_file and input_file.exists():
        if input_file.suffix.lower() == '.pdf':
            input_content = read_pdf_file(input_file)
        else:
            input_content = read_text_file(input_file)
        input_content = input_content[:2000]
    
    # Load example if provided
    example_content = ""
    if example_file and example_file.exists():
        if example_file.suffix.lower() == '.pdf':
            example_content = read_pdf_file(example_file)
        else:
            example_content = read_text_file(example_file)
        example_content = example_content[:2000]
    
    # Default prompt
    default_prompt = """
    You are an expert real estate investment analyst. Based on the provided reference documents and input property information, generate a comprehensive Investment Committee (IC) report.
    
    The report should include:
    1. Executive Summary
    2. Property Overview and Description
    3. Financial Analysis and Projections
    4. Market Analysis and Comparables
    5. Risk Assessment
    6. Investment Recommendation
    
    Make sure to provide specific, data-driven analysis and actionable recommendations.
    """
    
    prompt = custom_prompt or default_prompt
    
    # Construct the full prompt
    full_prompt = f"""
    {prompt}
    
    {context}
    
    {'EXAMPLE OUTPUT FORMAT:' + example_content if example_content else ''}
    
    {'INPUT PROPERTY TO ANALYZE:' + input_content if input_content else ''}
    
    Please generate a comprehensive investment analysis report based on the above information:
    """
    
    try:
        print("🔄 Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert real estate investment analyst with extensive experience in property evaluation and investment committee reports."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        
        report = response.choices[0].message.content
        print("✅ Analysis generated successfully")
        return report
        
    except Exception as e:
        raise Exception(f"OpenAI API error: {e}")

def save_report(report: str, output_dir: Path) -> Path:
    """Save the report to a file"""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"property_analysis_report_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"💾 Report saved to: {output_file}")
    return output_file

def main():
    """Main function"""
    print("🏠 Property Analysis System - Complete Solution")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return
    
    # Define paths (easily configurable)
    base_dir = Path(__file__).parent
    database_dir = base_dir / "database"
    unstructured_data_dir = database_dir / "Unstructered_data"
    input_file = database_dir / "Input" / "Birkehuset_IM.pdf"
    example_file = database_dir / "Evaluate_property_example" / "IC_output_carlsbergbyen.pdf"
    reports_dir = base_dir / "reports"
    
    print(f"\n📍 Configuration:")
    print(f"   Database: {database_dir}")
    print(f"   Reference data: {unstructured_data_dir}")
    print(f"   Input file: {input_file}")
    print(f"   Example file: {example_file}")
    print(f"   Reports: {reports_dir}")
    
    try:
        # Load reference documents
        documents = load_documents_from_directory(unstructured_data_dir)
        
        if not documents:
            print("⚠️ No documents found in reference directory")
            print("Make sure your database directory contains PDF, Excel, or Word files")
        
        # Generate analysis
        report = generate_analysis_with_openai(
            documents=documents,
            input_file=input_file if input_file.exists() else None,
            example_file=example_file if example_file.exists() else None
        )
        
        # Save report
        output_file = save_report(report, reports_dir)
        
        # Show results
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📄 Report saved to: {output_file}")
        print(f"📏 Report length: {len(report)} characters")
        
        print(f"\n📖 Report Preview (first 800 characters):")
        print("-" * 50)
        print(report[:800] + "..." if len(report) > 800 else report)
        
        print(f"\n✅ Full report available at: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

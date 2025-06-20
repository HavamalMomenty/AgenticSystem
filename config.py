"""
Configuration file for LlamaIndex + OpenAI O3 Report Generation System
"""
import os
from pathlib import Path

# Base paths (easily configurable)
BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"

# Directory paths
UNSTRUCTURED_DATA_DIR = DATABASE_DIR / "Unstructered_data"
EXAMPLE_OUTPUT_DIR = DATABASE_DIR / "Evaluate_property_example"
INPUT_DIR = DATABASE_DIR / "Input"

# Specific file paths
EXAMPLE_OUTPUT_FILE = EXAMPLE_OUTPUT_DIR / "IC_output_carlsbergbyen.pdf"
INPUT_FILE = INPUT_DIR / "Birkehuset_IM.pdf"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set via environment variable
OPENAI_MODEL = "gpt-4o-mini"  # Using a more widely available model, change to "o3-mini" when available

# LlamaIndex Configuration
INDEX_STORAGE_DIR = BASE_DIR / "storage"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# Custom prompt for the AI model
ANALYSIS_PROMPT = """
You are a real estate investment analysis expert. Based on the provided reference documents and the example output format, 
generate a comprehensive investment committee (IC) report for the given input property.

The report should include:
1. Executive Summary
2. Property Overview
3. Financial Analysis
4. Market Analysis
5. Risk Assessment
6. Investment Recommendation

Use the reference documents to inform your analysis and follow the structure and style of the example output.
Make sure to provide specific, data-driven insights and recommendations.

Input Property: {input_file}
Reference Data: Available in the indexed database
Example Format: Based on the provided example output

Generate a detailed, professional investment analysis report.
"""

# Advanced OpenAI Parameters
OPENAI_PARAMS = {
    "temperature": 0.3,  # Lower for more consistent, factual responses
    "max_tokens": 4000,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# File types to index
SUPPORTED_FILE_TYPES = [".pdf", ".xlsx", ".xls", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".csv"]

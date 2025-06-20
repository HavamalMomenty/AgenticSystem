"""
LlamaIndex + OpenAI Property Analysis System - Working Version
Fixed imports for LlamaIndex 0.10.57
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

# Third-party imports
import pandas as pd
from dotenv import load_dotenv

# LlamaIndex imports - Fixed for version 0.10.57
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    Settings,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Local imports
from config import *

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('report_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PropertyAnalysisSystem:
    """
    Main class for property analysis using LlamaIndex and OpenAI
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the system with OpenAI API key and configure LlamaIndex
        
        Args:
            api_key: OpenAI API key. If None, will use environment variable.
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        # Configure LlamaIndex settings
        self._configure_llama_index()
        
        # Initialize components
        self.index = None
        self.query_engine = None
        
        logger.info("PropertyAnalysisSystem initialized successfully")
    
    def _configure_llama_index(self):
        """Configure LlamaIndex with OpenAI models"""
        try:
            # Configure LLM
            Settings.llm = OpenAI(
                model=OPENAI_MODEL,
                api_key=self.api_key,
                temperature=OPENAI_PARAMS["temperature"],
                max_tokens=OPENAI_PARAMS["max_tokens"]
            )
            
            # Configure embeddings
            Settings.embed_model = OpenAIEmbedding(
                api_key=self.api_key,
                model="text-embedding-3-small"  # Using smaller model for better compatibility
            )
            
            # Configure node parser
            Settings.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            logger.info("LlamaIndex configured with OpenAI models")
            
        except Exception as e:
            logger.error(f"Error configuring LlamaIndex: {e}")
            raise
    
    def build_index(self, data_dir: Path = UNSTRUCTURED_DATA_DIR, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        Build or load the vector index from unstructured data
        
        Args:
            data_dir: Directory containing unstructured data
            force_rebuild: Whether to force rebuild the index
            
        Returns:
            VectorStoreIndex: The built or loaded index
        """
        storage_dir = INDEX_STORAGE_DIR
        
        # Check if index already exists and force_rebuild is False
        if storage_dir.exists() and not force_rebuild:
            try:
                logger.info("Loading existing index from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                self.index = load_index_from_storage(storage_context)
                logger.info("Index loaded successfully")
                return self.index
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Building new index...")
        
        # Build new index
        logger.info(f"Building new index from data directory: {data_dir}")
        
        # Load documents
        documents = self._load_documents(data_dir)
        
        if not documents:
            raise ValueError(f"No documents found in {data_dir}")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Create index
        self.index = VectorStoreIndex.from_documents(documents)
        
        # Persist index
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(storage_dir))
        
        logger.info(f"Index built and persisted to {storage_dir}")
        return self.index
    
    def _load_documents(self, data_dir: Path) -> List[Document]:
        """
        Load documents from the data directory
        
        Args:
            data_dir: Directory containing documents
            
        Returns:
            List of Document objects
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Use SimpleDirectoryReader with file type filtering
        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            required_exts=SUPPORTED_FILE_TYPES,
            recursive=True
        )
        
        documents = reader.load_data()
        
        # Add metadata to documents
        for doc in documents:
            if hasattr(doc, 'metadata') and 'file_path' in doc.metadata:
                file_path = Path(doc.metadata['file_path'])
                doc.metadata.update({
                    'file_name': file_path.name,
                    'file_type': file_path.suffix,
                    'directory': str(file_path.parent.relative_to(data_dir))
                })
        
        return documents
    
    def setup_query_engine(self, similarity_top_k: int = 10):
        """
        Setup the query engine with retrieval
        
        Args:
            similarity_top_k: Number of top similar documents to retrieve
        """
        if not self.index:
            raise ValueError("Index must be built before setting up query engine")
        
        # Create query engine - simplified version
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="compact"
        )
        
        logger.info("Query engine configured successfully")
    
    def load_example_output(self, example_file: Path = EXAMPLE_OUTPUT_FILE) -> str:
        """
        Load and extract text from the example output file
        
        Args:
            example_file: Path to the example output file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Loading example output from: {example_file}")
        
        if not example_file.exists():
            raise FileNotFoundError(f"Example output file not found: {example_file}")
        
        # Use SimpleDirectoryReader to extract text
        reader = SimpleDirectoryReader(input_files=[str(example_file)])
        documents = reader.load_data()
        
        if documents:
            return documents[0].text
        else:
            raise ValueError(f"Could not extract text from {example_file}")
    
    def load_input_document(self, input_file: Path = INPUT_FILE) -> str:
        """
        Load and extract text from the input document
        
        Args:
            input_file: Path to the input document
            
        Returns:
            Extracted text content
        """
        logger.info(f"Loading input document from: {input_file}")
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input document not found: {input_file}")
        
        # Use SimpleDirectoryReader to extract text
        reader = SimpleDirectoryReader(input_files=[str(input_file)])
        documents = reader.load_data()
        
        if documents:
            return documents[0].text
        else:
            raise ValueError(f"Could not extract text from {input_file}")
    
    def generate_report(
        self, 
        input_file: Path = INPUT_FILE,
        example_file: Path = EXAMPLE_OUTPUT_FILE,
        custom_prompt: Optional[str] = None,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generate a comprehensive property analysis report
        
        Args:
            input_file: Path to the input property document
            example_file: Path to the example output format
            custom_prompt: Custom prompt override
            output_file: Path to save the generated report
            
        Returns:
            Generated report content
        """
        if not self.query_engine:
            raise ValueError("Query engine must be set up before generating report")
        
        logger.info("Starting report generation...")
        
        # Load example output and input document
        try:
            example_content = self.load_example_output(example_file)
            input_content = self.load_input_document(input_file)
        except FileNotFoundError as e:
            logger.warning(f"Could not load reference files: {e}")
            example_content = "No example available"
            input_content = f"Analyzing file: {input_file.name}"
        
        # Prepare the comprehensive prompt
        prompt = custom_prompt or ANALYSIS_PROMPT
        
        enhanced_prompt = f"""
        {prompt}
        
        EXAMPLE OUTPUT FORMAT AND STYLE:
        {example_content[:1500]}...
        
        INPUT PROPERTY TO ANALYZE:
        {input_content[:1500]}...
        
        INSTRUCTIONS:
        1. Use the indexed reference database to gather relevant market data, comparable properties, and industry insights
        2. Follow the structure and professional tone of the example output
        3. Provide specific, data-driven analysis and recommendations
        4. Include financial projections and risk assessments based on similar properties in the database
        5. Make sure the analysis is comprehensive and actionable for investment committee decision-making
        
        Generate the complete investment analysis report now:
        """
        
        # Query the system for comprehensive analysis
        logger.info("Querying the system for comprehensive analysis...")
        response = self.query_engine.query(enhanced_prompt)
        
        report_content = str(response)
        
        # Save report if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report saved to: {output_file}")
        
        logger.info("Report generation completed successfully")
        return report_content
    
    def query_database(self, query: str) -> str:
        """
        Query the indexed database directly
        
        Args:
            query: Query string
            
        Returns:
            Query response
        """
        if not self.query_engine:
            raise ValueError("Query engine must be set up before querying")
        
        response = self.query_engine.query(query)
        return str(response)


def main():
    """
    Main function to demonstrate the system
    """
    try:
        # Initialize the system
        logger.info("Initializing Property Analysis System...")
        system = PropertyAnalysisSystem()
        
        # Build the index
        logger.info("Building index from unstructured data...")
        system.build_index(force_rebuild=False)  # Set to True to force rebuild
        
        # Setup query engine
        logger.info("Setting up query engine...")
        system.setup_query_engine()
        
        # Generate report
        logger.info("Generating property analysis report...")
        output_dir = Path("reports")
        output_file = output_dir / f"property_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report = system.generate_report(output_file=output_file)
        
        print("\n" + "="*80)
        print("PROPERTY ANALYSIS REPORT GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Report saved to: {output_file}")
        print("\nReport preview (first 1000 characters):")
        print("-"*40)
        print(report[:1000] + "..." if len(report) > 1000 else report)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your OpenAI API key is set in .env file")
        print("2. Check that your database files exist")
        print("3. Verify internet connection for OpenAI API calls")


if __name__ == "__main__":
    main()

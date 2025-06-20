# Property Analysis System with LlamaIndex + OpenAI O3

A comprehensive real estate investment analysis system that uses LlamaIndex to index unstructured property data and OpenAI's O3 model to generate professional investment committee (IC) reports.

## 🏗️ System Architecture

```
Property Analysis System
├── LlamaIndex Vector Database
│   ├── PDF Documents (Property details, Financial reports)
│   ├── Excel Files (Financial models, Market data)
│   ├── Word Documents (Investment memorandums)
│   └── PowerPoint Presentations (Market analysis)
├── OpenAI O3 Model Integration
│   ├── Advanced reasoning capabilities
│   ├── Configurable parameters
│   └── Context-aware analysis
└── Report Generation Engine
    ├── Example-based formatting
    ├── Multi-step analysis
    └── Interactive querying
```

## 🚀 Features

- **Automatic Document Indexing**: Indexes PDF, Excel, Word, and PowerPoint files
- **AI-Powered Analysis**: Uses OpenAI O3 for sophisticated property analysis
- **Example-Based Learning**: Learns from existing report formats
- **Configurable Parameters**: Easily adjustable paths, prompts, and AI settings
- **Multi-Step Analysis**: Breaks down analysis into market, financial, and risk components
- **Interactive Mode**: Query the system interactively for specific insights
- **Comprehensive Logging**: Full audit trail of analysis process

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key with O3 model access
- Windows, macOS, or Linux

## 🛠️ Installation

1. **Clone or download the project files**

2. **Run the setup script**:
   ```powershell
   python setup.py
   ```

3. **Configure your API key**:
   - Edit the `.env` file created during setup
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## 📁 Directory Structure

```
Momenty_lama/
├── main.py                 # Main application
├── config.py              # Configuration settings
├── advanced_features.py   # Advanced functionality
├── setup.py               # Setup script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from template)
├── .env.template          # Environment template
├── database/              # Your data directory
│   ├── Unstructered_data/ # Reference documents
│   ├── Input/             # Documents to analyze
│   └── Evaluate_property_example/ # Example outputs
├── storage/               # LlamaIndex storage (auto-created)
├── reports/               # Generated reports (auto-created)
└── logs/                  # Log files (auto-created)
```

## ⚙️ Configuration

### Basic Configuration (`config.py`)

```python
# Easily configurable paths
UNSTRUCTURED_DATA_DIR = DATABASE_DIR / "Unstructered_data"
EXAMPLE_OUTPUT_FILE = DATABASE_DIR / "Evaluate_property_example" / "IC_output_carlsbergbyen.pdf"
INPUT_FILE = DATABASE_DIR / "Input" / "Birkehuset_IM.pdf"

# OpenAI model configuration
OPENAI_MODEL = "o3-mini"  # or "o3" when available

# Custom analysis prompt
ANALYSIS_PROMPT = """
Your custom prompt here...
"""
```

### Advanced OpenAI Parameters

```python
OPENAI_PARAMS = {
    "temperature": 0.3,      # Creativity vs. consistency
    "max_tokens": 4000,      # Response length
    "top_p": 0.9,           # Nucleus sampling
    "frequency_penalty": 0.0, # Repetition penalty
    "presence_penalty": 0.0,  # Topic diversity
}
```

## 🎯 Usage

### Basic Usage

```python
from main import PropertyAnalysisSystem

# Initialize the system
system = PropertyAnalysisSystem()

# Build index from your data
system.build_index()

# Setup query engine
system.setup_query_engine()

# Generate comprehensive report
report = system.generate_report()
print(report)
```

### Advanced Usage

```python
from advanced_features import AdvancedPropertyAnalysis

# Initialize advanced system
system = AdvancedPropertyAnalysis()
system.build_index()
system.setup_query_engine()

# Multi-step analysis
results = system.multi_step_analysis()

# Interactive analysis session
system.interactive_analysis()
```

### Command Line Usage

```powershell
# Run basic analysis
python main.py

# Run advanced features demo
python advanced_features.py
```

## 🔧 Alternative OpenAI Configurations

The system supports multiple analysis modes optimized for different use cases:

### 1. Conservative Analysis (Financial Focus)
- **Temperature**: 0.1 (highly factual)
- **Use case**: Financial projections, risk assessment
- **Best for**: Investment committee presentations

### 2. Creative Insights (Opportunity Identification)
- **Temperature**: 0.7 (more creative)
- **Use case**: Market opportunities, innovative strategies
- **Best for**: Strategic planning, market positioning

### 3. Technical Analysis (Comprehensive)
- **Temperature**: 0.3 (balanced)
- **Use case**: Detailed technical analysis
- **Best for**: Due diligence reports

### 4. Executive Summary (Concise)
- **Max tokens**: 1500 (brief responses)
- **Use case**: Quick overviews, executive briefings
- **Best for**: High-level decision making

## 🛠️ Advanced LlamaIndex Features

### Custom Node Parsers
- **Financial documents**: Optimized chunk sizes for financial data
- **Text-heavy documents**: Semantic splitting for better context
- **Mixed content**: Adaptive parsing strategies

### Metadata Filtering
Filter analysis by:
- Document type (PDF, Excel, Word, PowerPoint)
- Directory/category
- Date ranges
- Content type

### Response Synthesis Modes
- **Tree Summarize**: Comprehensive multi-document analysis
- **Compact**: Quick focused responses
- **Accumulate**: Building analysis step-by-step

## 📊 Example Queries

```python
# Market analysis
response = system.query_database("What are the current market trends for residential properties in Copenhagen?")

# Financial analysis
response = system.query_database("Calculate expected IRR and NPV for this investment based on similar properties")

# Risk assessment
response = system.query_database("What are the key risks associated with this property type and location?")

# Comparable analysis
response = system.query_database("Find 3 comparable properties and provide detailed comparison")
```

## 🎨 Customization Options

### Custom Prompts
Modify `ANALYSIS_PROMPT` in `config.py` to customize the analysis focus:
- Add specific analysis requirements
- Include company-specific terminology
- Focus on particular metrics or KPIs

### File Type Support
Current supported formats:
- **PDF**: Property brochures, financial reports, legal documents
- **Excel**: Financial models, market data, calculations
- **Word**: Investment memorandums, due diligence reports
- **PowerPoint**: Market presentations, property overviews

### Output Customization
- **Format**: Text, JSON, structured reports
- **Length**: Executive summary to detailed analysis
- **Focus**: Financial, market, risk, or comprehensive

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run `python setup.py` to install dependencies
2. **API Key Issues**: Check `.env` file configuration
3. **File Access**: Ensure proper permissions for database directory
4. **Model Access**: Verify O3 model availability in your OpenAI account

### Performance Optimization

- **Index Rebuilding**: Set `force_rebuild=True` only when data changes significantly
- **Chunk Size**: Adjust `CHUNK_SIZE` in config for your document types
- **Similarity Threshold**: Tune `similarity_cutoff` for retrieval quality

## 📈 Scaling and Extensions

### For Larger Datasets
- Implement persistent vector storage (Pinecone, Weaviate)
- Use distributed processing for document ingestion
- Add document versioning and update mechanisms

### Integration Options
- **CRM Integration**: Connect to property management systems
- **Database Integration**: Link to SQL databases for structured data
- **API Development**: Create REST API for web applications
- **Workflow Integration**: Connect to approval and notification systems

## 🔒 Security Considerations

- Store API keys securely (environment variables, key vaults)
- Implement access controls for sensitive documents
- Consider data encryption for proprietary information
- Audit trails for analysis and decisions

## 📝 License

This project is provided as-is for educational and commercial use. Please ensure compliance with OpenAI's usage policies and any applicable data protection regulations.

## 🤝 Contributing

Feel free to extend this system with:
- Additional document parsers
- New analysis templates
- Integration with other AI models
- Enhanced visualization capabilities

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the configuration options
3. Examine the log files in the `logs/` directory
4. Test with the provided example data first

---

**Ready to transform your property analysis workflow with AI! 🏠🤖**

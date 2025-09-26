# CSV Agent - Financial Data Analyzer

A powerful CSV agent that uses Ollama LLM to understand and analyze your financial transaction data through natural language queries.

## Features

-   🤖 **Natural Language Queries**: Ask questions about your data in plain English
-   📊 **Interactive Visualizations**: Automatic charts and graphs of your financial data
-   💬 **Chat Interface**: Streamlit web app with conversational AI
-   📈 **Data Analysis**: Automatic pandas code generation and execution
-   🔍 **Data Exploration**: Filter and explore your raw data
-   💰 **Financial Insights**: Income vs expenses, spending categories, trends

## Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Start Ollama service**:
    ```bash
    ollama serve
    ```
3. **Pull a language model**:
    ```bash
    ollama pull llama3.2
    ```

### Installation

1. **Clone or download this project**
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Ensure your CSV file is named `data.csv`** in the project directory

### Usage

#### Command Line Interface

```bash
python app.py
```

#### Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## Example Questions

Ask natural language questions about your financial data:

-   "What's my total spending on office costs?"
-   "Show me all transactions in June 2022"
-   "What are my top 5 expense categories?"
-   "How much did I spend on motor vehicle costs?"
-   "What's my monthly spending trend?"
-   "Which bank account do I use most?"
-   "Show me all income transactions"

## CSV Format

The agent expects a CSV file with financial transaction data. Your CSV should have columns like:

-   **Code Name**: Transaction category/type
-   **Date**: Transaction date
-   **Bank**: Bank account used
-   **Entry #**: Transaction entry number
-   **Transaction Details**: Description
-   **Net Amount**: Amount (negative = expense, positive = income)
-   **GST**: GST amount

## How It Works

1. **Data Loading**: Automatically loads and analyzes your CSV structure
2. **Context Generation**: Creates a comprehensive description of your data for the LLM
3. **Query Processing**: Converts natural language questions into pandas operations
4. **Code Execution**: Safely executes generated pandas code
5. **Response Formatting**: Returns human-readable answers with data insights

## Available Models

The agent supports various Ollama models:

-   llama3.2 (recommended)
-   llama3.1
-   llama2
-   mistral
-   codellama

## Troubleshooting

### Common Issues

1. **"Error processing query"**: Make sure Ollama is running (`ollama serve`)
2. **"Model not found"**: Install the model (`ollama pull llama3.2`)
3. **"Error loading CSV"**: Check that `data.csv` exists and is properly formatted
4. **Date parsing issues**: The agent supports DD/MM/YYYY and DD-MM-YYYY formats

### Performance Tips

-   Use smaller models (llama3.2) for faster responses
-   Larger datasets may take longer to process
-   Complex queries might require more powerful models

## Project Structure

```
csv_agent/
├── app.py              # Command line CSV agent
├── streamlit_app.py    # Web interface
├── data.csv           # Your financial data
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Security Note

The agent executes pandas code in a controlled environment with limited access to system functions. Only safe pandas operations are allowed.

## Contributing

Feel free to enhance the agent with additional features like:

-   Support for multiple CSV files
-   More visualization types
-   Export capabilities
-   Advanced financial metrics
-   Custom model fine-tuning

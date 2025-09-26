import pandas as pd
import openai
import json
import re
import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional


class CSVAgent:
    def __init__(self, csv_file_path: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize the CSV Agent

        Args:
            csv_file_path: Path to the CSV file
            model_name: OpenAI model name to use (default: gpt-4o-mi                # Combine all code blocks for execution
                all_code = "\n\n".join(code_blocks)

                # Fix common Series comparison issues
                for col in self.df.columns:
                    # Handle specific column comparisons in various patterns
                    all_code = all_code.replace(f"df[df['{col}'] < ", f"df[df['{col}'].lt(")
                    all_code = all_code.replace(f"df[df['{col}'] > ", f"df[df['{col}'].gt(")
                    all_code = all_code.replace(f"df[df['{col}'] <= ", f"df[df['{col}'].le(")
                    all_code = all_code.replace(f"df[df['{col}'] >= ", f"df[df['{col}'].ge(")

                    # Also handle more complex comparisons
                    all_code = all_code.replace(f"df['{col}'] < ", f"df['{col}'].lt(")
                    all_code = all_code.replace(f"df['{col}'] > ", f"df['{col}'].gt(")
                    all_code = all_code.replace(f"df['{col}'] <= ", f"df['{col}'].le(")
                    all_code = all_code.replace(f"df['{col}'] >= ", f"df['{col}'].ge(")"""
        self.csv_file_path = csv_file_path
        self.model_name = model_name
        self.df = None
        self.data_summary = None

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.load_data()
        self.analyze_data()

    def load_data(self):
        """Load the CSV data into a pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(
                f"✅ Successfully loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns"
            )
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise

    def analyze_data(self):
        """Analyze the data structure and content"""
        if self.df is None:
            return

        # Convert date column if it exists
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(
                self.df["Date"], format="%d/%m/%Y", errors="coerce"
            )
            # Try alternative format
            self.df["Date"] = pd.to_datetime(
                self.df["Date"], format="%d-%m-%Y", errors="coerce"
            )

        # Convert numeric columns
        if "Net Amount" in self.df.columns:
            self.df["Net Amount"] = pd.to_numeric(
                self.df["Net Amount"], errors="coerce"
            )
        if "GST" in self.df.columns:
            self.df["GST"] = pd.to_numeric(self.df["GST"], errors="coerce")

        # Create data summary
        self.data_summary = {
            "columns": list(self.df.columns),
            "shape": self.df.shape,
            "date_range": None,
            "total_transactions": len(self.df),
            "unique_categories": None,
            "total_amount": None,
            "sample_data": self.df.head(3).to_dict("records"),
        }

        # Add specific analysis for financial data
        if "Date" in self.df.columns:
            valid_dates = self.df["Date"].dropna()
            if len(valid_dates) > 0:
                self.data_summary["date_range"] = {
                    "start": str(valid_dates.min().date()),
                    "end": str(valid_dates.max().date()),
                }

        if "Code Name" in self.df.columns:
            self.data_summary["unique_categories"] = self.df["Code Name"].nunique()

        if "Net Amount" in self.df.columns:
            self.data_summary["total_amount"] = float(self.df["Net Amount"].sum())
            self.data_summary["amount_stats"] = {
                "min": float(self.df["Net Amount"].min()),
                "max": float(self.df["Net Amount"].max()),
                "mean": float(self.df["Net Amount"].mean()),
            }

    def get_data_context(self) -> str:
        """Generate a context description of the data for the LLM"""
        context = f"""
This is a financial transactions dataset with the following characteristics:

DATASET OVERVIEW:
- Total records: {self.data_summary["total_transactions"]}
- Columns: {", ".join(self.data_summary["columns"])}
- Date range: {self.data_summary.get("date_range", "N/A")}
- Unique transaction categories: {self.data_summary.get("unique_categories", "N/A")}

FINANCIAL SUMMARY:
- Total amount: ${self.data_summary.get("total_amount", 0):,.2f}
- Amount range: ${self.data_summary.get("amount_stats", {}).get("min", 0):,.2f} to ${self.data_summary.get("amount_stats", {}).get("max", 0):,.2f}
- Average transaction: ${self.data_summary.get("amount_stats", {}).get("mean", 0):,.2f}

COLUMN DESCRIPTIONS:
- Code Name: Transaction category/type
- Date: Transaction date
- Bank: Bank account used
- Entry #: Transaction entry number
- Transaction Details: Description of the transaction
- Net Amount: Transaction amount (negative = expense, positive = income)
- GST: GST amount

SAMPLE DATA:
{json.dumps(self.data_summary["sample_data"], indent=2, default=str)}
        """
        return context

    def execute_pandas_query(self, query_code: str) -> str:
        """Execute pandas code safely and return results as string (for CLI)"""
        try:
            safe_globals = {
                "df": self.df,
                "pd": pd,
                "px": px,  # Add plotly express
                "go": go,  # Add plotly graph objects
                "plt": plt,  # Keep matplotlib for backwards compatibility
                "datetime": datetime,
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "round": round,
                "abs": abs,
            }

            safe_locals = {}
            exec(query_code, safe_globals, safe_locals)

            # Look for result variables
            result_vars = [
                "result",
                "top_expenses",
                "top_expense_categories",
                "expense_categories",
                "total_spent",
                "expense_summary",
                "top_categories",
                "monthly_data",
                "filtered_data",
                "summary",
                "answer",
                "expenses",
                "income",
                "fig",
            ]

            result = None
            result_type = None
            var_name = None
            plotly_fig = None
            matplotlib_fig = None

            # First check for matplotlib figure
            if "fig" in safe_locals:
                if hasattr(safe_locals["fig"], "__module__"):
                    if safe_locals["fig"].__module__.startswith("matplotlib"):
                        matplotlib_fig = safe_locals["fig"]
                    elif safe_locals["fig"].__module__.startswith("plotly"):
                        plotly_fig = safe_locals["fig"]

            # Then check for other result variables
            for var in result_vars:
                if (
                    var in safe_locals and var != "fig"
                ):  # Skip fig as we handled it separately
                    result = safe_locals[var]
                    result_type = type(result).__name__
                    var_name = var
                    break

            # Create a copy of all local variables for individual block execution
            all_vars = {k: v for k, v in safe_locals.items() if not k.startswith("_")}

            if result is None and matplotlib_fig is None and plotly_fig is None:
                return {
                    "success": False,
                    "error": "No result or figure was found in the executed code",
                    "type": None,
                    "variable_name": None,
                    "all_vars": all_vars,
                }

            return {
                "success": True,
                "result": result,
                "type": result_type,
                "variable_name": var_name,
                "matplotlib_fig": matplotlib_fig,
                "plotly_fig": plotly_fig,
                "all_vars": all_vars,
            }

        except Exception as e:
            return f"Error executing query: {str(e)}"

    def execute_pandas_query_for_streamlit(self, query_code: str):
        """Execute pandas code and return structured results for Streamlit display"""
        try:
            safe_globals = {
                "df": self.df,
                "pd": pd,
                "np": np,
                "plt": plt,
                "px": px,  # Add plotly express
                "go": go,  # Add plotly graph objects
                "sns": None,  # Will be properly set if seaborn is imported
                "datetime": datetime,
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "round": round,
                "abs": abs,
            }
            safe_locals = {}

            # Fix potential DataFrame boolean ambiguity issues
            enhanced_code = query_code

            # Add necessary imports - prioritize plotly
            if "import pandas" not in enhanced_code and "pd." in enhanced_code:
                enhanced_code = "import pandas as pd\n" + enhanced_code
            if "import numpy" not in enhanced_code and "np." in enhanced_code:
                enhanced_code = "import numpy as np\n" + enhanced_code
            if "import plotly" not in enhanced_code and (
                "px." in enhanced_code or "go." in enhanced_code
            ):
                enhanced_code = (
                    "import plotly.express as px\nimport plotly.graph_objects as go\n"
                    + enhanced_code
                )
            # Add plotly imports by default for plotting
            if any(
                plot_keyword in enhanced_code.lower()
                for plot_keyword in [
                    "plot",
                    "chart",
                    "graph",
                    "visualization",
                    "heatmap",
                ]
            ):
                if "import plotly" not in enhanced_code:
                    enhanced_code = (
                        "import plotly.express as px\nimport plotly.graph_objects as go\n"
                        + enhanced_code
                    )
            # Add matplotlib only if explicitly used
            if "import matplotlib" not in enhanced_code and "plt." in enhanced_code:
                enhanced_code = "import matplotlib.pyplot as plt\n" + enhanced_code

            # Fix pandas resample deprecated 'M' frequency warning
            # For resample, use 'ME' but for to_period, keep 'M'
            enhanced_code = enhanced_code.replace("df.resample('M'", "df.resample('ME'")
            # Don't change Period 'M' to 'ME'
            if "to_period" not in enhanced_code:
                enhanced_code = enhanced_code.replace(".resample('M'", ".resample('ME'")

            # Fix DataFrame.pivot() syntax error
            if "pivot(" in enhanced_code and "pivot_table" not in enhanced_code:
                # Look for common pivot patterns and replace them with proper pivot_table syntax
                # Pattern 1: Three positional arguments (index, columns, values)
                enhanced_code = enhanced_code.replace(
                    '.pivot("Code Name", "Month", "Net Amount")',
                    '.pivot_table(index="Code Name", columns="Month", values="Net Amount", aggfunc="sum")',
                )
                # Pattern 2: Two positional arguments (common mistake)
                enhanced_code = enhanced_code.replace(
                    '.pivot("Code Name", "Net Amount")',
                    '.pivot_table(index="Code Name", values="Net Amount", aggfunc="sum")',
                )

            # Initialize seaborn if needed
            if "sns." in enhanced_code:
                try:
                    import seaborn as sns

                    safe_globals["sns"] = sns
                except ImportError:
                    print(
                        "Warning: Seaborn library not found. Please install it with 'pip install seaborn'"
                    )

            # Fix pivot syntax more specifically for the heatmap example
            if (
                "result = " in enhanced_code
                and ".pivot(" in enhanced_code
                and "Code Name" in enhanced_code
            ):
                # Replace the problematic pivot with a proper solution for heatmaps
                enhanced_code = enhanced_code.replace(
                    'result = category_expenses.pivot("Code Name", "Net Amount", "Net Amount")',
                    'result = category_expenses.set_index("Code Name")["Net Amount"]',  # Create a simple series for plotting
                )
                # If this is for a heatmap, suggest using plotly instead
                if "heatmap" in enhanced_code.lower():
                    enhanced_code = enhanced_code.replace(
                        "sns.heatmap(",
                        "# Using plotly instead of seaborn for better web compatibility\n# px.imshow(",
                    )

            # Fix common Series comparison issues
            for col in self.df.columns:
                # Handle specific column comparisons
                enhanced_code = enhanced_code.replace(
                    f"df[df['{col}'] < ", f"df[df['{col}'].lt("
                )
                enhanced_code = enhanced_code.replace(
                    f"df[df['{col}'] > ", f"df[df['{col}'].gt("
                )
                enhanced_code = enhanced_code.replace(
                    f"df[df['{col}'] <= ", f"df[df['{col}'].le("
                )
                enhanced_code = enhanced_code.replace(
                    f"df[df['{col}'] >= ", f"df[df['{col}'].ge("
                )

            # Handle matplotlib if used (but we prefer plotly)
            if "plt.show()" in enhanced_code:
                enhanced_code = enhanced_code.replace(
                    "plt.show()", "plt.tight_layout()"
                )

            # Prevent duplicate tight_layout calls
            if "plt.tight_layout()" in enhanced_code:
                # Count occurrences of tight_layout
                tight_layout_count = enhanced_code.count("plt.tight_layout()")
                if tight_layout_count > 1:
                    # Replace all occurrences with a comment except the last one
                    parts = enhanced_code.split("plt.tight_layout()")
                    enhanced_code = parts[0]
                    for i in range(1, len(parts)):
                        if i < len(parts) - 1:
                            enhanced_code += (
                                "# plt.tight_layout() - removed duplicate" + parts[i]
                            )
                        else:
                            enhanced_code += "plt.tight_layout()" + parts[i]

            # Execute the code
            print("Executing code:")
            print(enhanced_code)
            exec(enhanced_code, safe_globals, safe_locals)

            # Look for result variables
            result_vars = [
                "result",
                "top_expenses",
                "top_expense_categories",
                "expense_categories",
                "total_spent",
                "expense_summary",
                "top_categories",
                "monthly_data",
                "filtered_data",
                "summary",
                "answer",
                "expenses",
                "income",
                "fig",
            ]

            result = None
            result_type = None
            var_name = None
            plotly_fig = None
            matplotlib_fig = None

            # First check for matplotlib figure
            if "fig" in safe_locals:
                if hasattr(safe_locals["fig"], "__module__"):
                    if safe_locals["fig"].__module__.startswith("matplotlib"):
                        matplotlib_fig = safe_locals["fig"]
                    elif safe_locals["fig"].__module__.startswith("plotly"):
                        plotly_fig = safe_locals["fig"]

            # Then check for other result variables
            for var in result_vars:
                if (
                    var in safe_locals and var != "fig"
                ):  # Skip fig as we handled it separately
                    result = safe_locals[var]
                    result_type = type(result).__name__
                    var_name = var
                    break

            # Create a copy of all local variables for individual block execution
            all_vars = {k: v for k, v in safe_locals.items() if not k.startswith("_")}

            if result is None and matplotlib_fig is None and plotly_fig is None:
                return {
                    "success": False,
                    "error": "No result or figure was found in the executed code",
                    "type": None,
                    "variable_name": None,
                    "all_vars": all_vars,
                }

            return {
                "success": True,
                "result": result,
                "type": result_type,
                "variable_name": var_name,
                "matplotlib_fig": matplotlib_fig,
                "plotly_fig": plotly_fig,
                "all_vars": all_vars,
            }

        except Exception as e:
            print(f"Error executing code: {str(e)}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "type": None,
                "variable_name": None,
            }

    def query(self, user_question: str) -> str:
        """
        Process a user question about the CSV data

        Args:
            user_question: The user's question about the data

        Returns:
            Answer to the user's question
        """
        # Create the prompt for the LLM
        system_prompt = f"""You are a data analyst assistant helping users understand their financial transaction data. 

{self.get_data_context()}

INSTRUCTIONS:
1. Answer the user's question about the financial data
2. If the question requires data analysis, provide pandas code to analyze the data
3. Explain your findings in plain English
4. For monetary amounts, always use proper formatting with $ symbol
5. For dates, use readable formats
6. If asked about trends, calculations, or specific data points, provide both the code and explanation
7. If the findings require searching or filtering, focus the search on all relevant columns not just one.

IMPORTANT CODING GUIDELINES:
- When filtering data with conditions, use explicit comparison methods: 
  - Instead of df[df['column'] < 0], use df[df['column'].lt(0)]
  - Instead of df[df['column'] > 0], use df[df['column'].gt(0)]
  - Instead of df[df['column'] <= 0], use df[df['column'].le(0)]
  - Instead of df[df['column'] >= 0], use df[df['column'].ge(0)]
- When combining multiple conditions use & and | operators with parentheses:
  df[(df['Net Amount'].lt(0)) & (df['Date'].dt.year == 2022)]
- When calculating with Series, use explicit methods instead of operators

Remember: The dataframe is available as 'df'. Negative amounts are expenses, positive amounts are income.

SPECIAL INSTRUCTION FOR FILTERING:
- When filtering or searching with keywords, ALWAYS use both 'Transaction Details' and 'Code Name' columns together (not just one) whenever applicable. For example, if searching for a keyword or category, apply the filter to both columns using an OR condition.
- Example: To find transactions related to 'fuel', use something like:
    df[df['Transaction Details'].str.contains('fuel', case=False, na=False) | df['Code Name'].str.contains('fuel', case=False, na=False)]
- This ensures that keyword searches are comprehensive across both columns.
"""

        user_prompt = f"""USER QUESTION: {user_question}

Provide a helpful answer that includes:
1. Direct answer to the question
2. If needed, pandas code to get the data (start code blocks with ```python and end with ```)
3. Clear explanation of findings
4. Any relevant insights or context

IMPORTANT CODING GUIDELINES:
- When filtering data, use pandas methods like .lt(), .gt(), .le(), .ge() instead of <, >, <=, >= 
  For example: df[df['Net Amount'].lt(0)] instead of df[df['Net Amount'] < 0]
- When combining multiple conditions use & and | operators with parentheses:
  df[(df['Net Amount'].lt(0)) & (df['Date'].dt.year == 2022)]
- For plots: PREFER PLOTLY - Use px (plotly express) or go (graph objects) for visualizations
- Available plotting: px.line(), px.bar(), px.scatter(), px.pie(), go.Figure(), etc.
- Store results in descriptive variable names
- If comparing pandas Series, use .eq(), .ne(), .lt(), .gt() methods, not ==, !=, <, >."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )

            llm_response = response.choices[0].message.content
            code_blocks = re.findall(r"```python\n(.*?)\n```", llm_response, re.DOTALL)

            # If no code blocks with python specifier, try without it
            if not code_blocks:
                code_blocks = re.findall(r"```\n(.*?)\n```", llm_response, re.DOTALL)

            output = llm_response
            if code_blocks:
                output += "\n\n--- EXECUTION RESULTS ---\n"

                # Combine all code blocks for execution
                all_code = "\n\n".join(code_blocks)

                # Auto-fix series comparison issues
                for col in self.df.columns:
                    all_code = all_code.replace(
                        f"df[df['{col}'] < ", f"df[df['{col}'].lt("
                    )
                    all_code = all_code.replace(
                        f"df[df['{col}'] > ", f"df[df['{col}'].gt("
                    )
                    all_code = all_code.replace(
                        f"df[df['{col}'] <= ", f"df[df['{col}'].le("
                    )
                    all_code = all_code.replace(
                        f"df[df['{col}'] >= ", f"df[df['{col}'].ge("
                    )

                # Execute all code at once
                try:
                    combined_result = self.execute_pandas_query(all_code.strip())
                    output += f"\nExecution Results:\n \n{combined_result}\n"
                except Exception:
                    # If combined execution fails, execute blocks individually
                    for i, code in enumerate(code_blocks):
                        fixed_code = code.strip()
                        for col in self.df.columns:
                            # Handle specific column comparisons in various patterns
                            fixed_code = fixed_code.replace(
                                f"df[df['{col}'] < ", f"df[df['{col}'].lt("
                            )
                            fixed_code = fixed_code.replace(
                                f"df[df['{col}'] > ", f"df[df['{col}'].gt("
                            )
                            fixed_code = fixed_code.replace(
                                f"df[df['{col}'] <= ", f"df[df['{col}'].le("
                            )
                            fixed_code = fixed_code.replace(
                                f"df[df['{col}'] >= ", f"df[df['{col}'].ge("
                            )

                            # Also handle more complex comparisons
                            fixed_code = fixed_code.replace(
                                f"df['{col}'] < ", f"df['{col}'].lt("
                            )
                            fixed_code = fixed_code.replace(
                                f"df['{col}'] > ", f"df['{col}'].gt("
                            )
                            fixed_code = fixed_code.replace(
                                f"df['{col}'] <= ", f"df['{col}'].le("
                            )
                            fixed_code = fixed_code.replace(
                                f"df['{col}'] >= ", f"df['{col}'].ge("
                            )

                        result = self.execute_pandas_query(fixed_code)
                        output += f"\nCode Block {i + 1} Results:\n \n{result}\n"

            return output
        except Exception as e:
            return f"Error processing query: {str(e)}\n\nPlease make sure your OpenAI API key is set in the OPENAI_API_KEY environment variable."

    def query_for_streamlit(self, user_question: str):
        """Process a user question and return structured results for Streamlit, including plot support"""
        system_prompt = f"""You are a data analyst assistant helping users understand their financial transaction data. 

{self.get_data_context()}

INSTRUCTIONS:
1. Answer the user's question about the financial data
2. If the question requires data analysis, provide pandas code to analyze the data
3. If the user asks for a plot or chart, generate Plotly code (preferred) instead of matplotlib
4. Store the main result in a variable called 'result'
5. Explain your findings in plain English
6. For monetary amounts, always use proper formatting with $ symbol
7. For dates, use readable formats

IMPORTANT CODING GUIDELINES:
- When filtering data with conditions, use explicit comparison methods: 
  - Instead of df[df['column'] < 0], use df[df['column'].lt(0)]
  - Instead of df[df['column'] > 0], use df[df['column'].gt(0)]
  - Instead of df[df['column'] <= 0], use df[df['column'].le(0)]
  - Instead of df[df['column'] >= 0], use df[df['column'].ge(0)]
- When calculating with Series, use explicit methods instead of operators
- Store your main data result in a variable called 'result'
- For plots: PREFER PLOTLY - Use px (plotly express) or go (graph objects) and store as 'fig'
- If generating multiple code blocks, define all variables needed in each block
- Include all import statements needed in your code
- Available plotting libraries: px (plotly.express), go (plotly.graph_objects), plt (matplotlib), sns (seaborn)

Remember: The dataframe is available as 'df'. Negative amounts are expenses, positive amounts are income.

SPECIAL INSTRUCTION FOR FILTERING:
- When filtering or searching with keywords, ALWAYS use both 'Transaction Details' and 'Code Name' columns together (not just one) whenever applicable. For example, if searching for a keyword or category, apply the filter to both columns using an OR condition.
- Example: To find transactions related to 'fuel', use something like:
    df[df['Transaction Details'].str.contains('fuel', case=False, na=False) | df['Code Name'].str.contains('fuel', case=False, na=False)]
- This ensures that keyword searches are comprehensive across both columns.
"""

        user_prompt = f"""USER QUESTION: {user_question}

Provide a helpful answer that includes:
1. Direct answer to the question
2. If needed, pandas code to get the data
3. If visualization would help, include code to create an appropriate Plotly plot (preferred over matplotlib)
4. Store the main result in a variable called 'result'
5. Clear explanation of findings
6. only one code block for all code, not multiple blocks

IMPORTANT CODING GUIDELINES:
- When filtering with conditions, use pandas methods like .lt(), .gt(), .le(), .ge() instead of <, >, <=, >= 
  For example: df[df['Net Amount'].lt(0)] instead of df[df['Net Amount'] < 0]
- When combining multiple conditions use & and | operators with parentheses:
  df[(df['Net Amount'].lt(0)) & (df['Date'].dt.year == 2022)]
- For plots: PREFER PLOTLY - Use px (plotly express) or go (graph objects) and store the figure as 'fig'
- Available plotting: px.line(), px.bar(), px.scatter(), px.pie(), go.Figure(), etc.
- If you need variables from previous code blocks, define them again in new blocks

If you need the result from a previous code block, include it again in the new block."""

        try:
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )

            llm_response = response.choices[0].message.content
            code_blocks = re.findall(r"```python\n(.*?)\n```", llm_response, re.DOTALL)

            # If no code blocks, try without the 'python' language specifier
            if not code_blocks:
                code_blocks = re.findall(r"```\n(.*?)\n```", llm_response, re.DOTALL)

            result_data = {
                "response": llm_response,
                "code_blocks": [],
                "success": True,
                "error": None,
                "combined_code": "",
            }

            # Execute all code blocks as a single unit to avoid variable scope issues
            if code_blocks:
                # Combine all code blocks with newlines between them
                all_code = "\n\n".join(code_blocks)
                combined_execution = self.execute_pandas_query_for_streamlit(all_code)

                # Store the result from the combined execution
                if combined_execution["success"]:
                    # Store individual blocks for display purposes
                    for i, code in enumerate(code_blocks):
                        result_data["code_blocks"].append(
                            {
                                "code": code.strip(),
                                "execution_result": combined_execution,  # Same result for all blocks
                                "block_number": i + 1,
                            }
                        )
                    result_data["combined_code"] = all_code
                else:
                    # If combined execution failed, try executing blocks individually
                    combined_vars = {}
                    for i, code in enumerate(code_blocks):
                        # Combine previous results with current code
                        combined_code = ""
                        for k, v in combined_vars.items():
                            # Handle pandas DataFrame and Series objects specifically
                            if isinstance(v, pd.DataFrame):
                                combined_code += f"{k} = pd.DataFrame({v.to_dict()})\n"
                            elif isinstance(v, pd.Series):
                                combined_code += f"{k} = pd.Series({v.to_dict()})\n"
                            else:
                                try:
                                    if not isinstance(
                                        v, (str, int, float, bool, list, dict, tuple)
                                    ):
                                        continue
                                    combined_code += f"{k} = {repr(v)}\n"
                                except Exception as ex:
                                    print(f"Error preparing variable {k}: {ex}")

                        # Add the current block
                        combined_code += "\n" + code.strip()

                        # Execute the individual block
                        execution_result = self.execute_pandas_query_for_streamlit(
                            combined_code
                        )

                        # Update the combined vars for next block
                        if (
                            execution_result["success"]
                            and "all_vars" in execution_result
                        ):
                            combined_vars.update(execution_result["all_vars"])

                        result_data["code_blocks"].append(
                            {
                                "code": code.strip(),
                                "execution_result": execution_result,
                                "block_number": i + 1,
                            }
                        )

            return result_data

        except Exception as e:
            import traceback

            print(f"Error in query_for_streamlit: {str(e)}")
            print(traceback.format_exc())
            return {
                "response": f"Error processing query: {str(e)}",
                "code_blocks": [],
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the financial data"""
        if self.df is None:
            return {}

        stats = {}

        # Transaction counts by category
        if "Code Name" in self.df.columns:
            stats["transactions_by_category"] = (
                self.df["Code Name"].value_counts().head(10).to_dict()
            )

        # Monthly spending if dates are available
        if "Date" in self.df.columns and "Net Amount" in self.df.columns:
            monthly_data = self.df.groupby(self.df["Date"].dt.to_period("M"))[
                "Net Amount"
            ].agg(["sum", "count"])
            stats["monthly_totals"] = monthly_data.to_dict()

        # Income vs Expenses
        if "Net Amount" in self.df.columns:
            expenses = self.df[self.df["Net Amount"] < 0]["Net Amount"].sum()
            income = self.df[self.df["Net Amount"] > 0]["Net Amount"].sum()
            stats["income_vs_expenses"] = {
                "total_income": float(income),
                "total_expenses": float(expenses),
                "net_position": float(income + expenses),
            }

        return stats


def main():
    """Main function to run the CSV Agent"""
    print("🤖 DATA Agent - Financial Data Analyzer")
    print("=" * 50)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key'")
        return

    # Initialize the agent
    try:
        agent = CSVAgent("data.csv", model_name="gpt-4o-mini")
        print(
            f"📊 Loaded financial data: {agent.data_summary['total_transactions']} transactions"
        )
        print(f"📅 Date range: {agent.data_summary.get('date_range', 'N/A')}")
        print(f"💰 Total amount: ${agent.data_summary.get('total_amount', 0):,.2f}")
        print("\n" + "=" * 50)

        # Interactive query loop
        print("\n💡 Ask questions about your financial data!")
        print("Examples:")
        print("- What's my total spending on office costs?")
        print("- Show me all transactions in June 2022")
        print("- What are my top 5 expense categories?")
        print("- How much did I spend on motor vehicle costs?")
        print("\nType 'quit' to exit")
        print("-" * 50)

        while True:
            user_input = input("\n🔍 Your question: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("\n🤔 Analyzing your question...")
            response = agent.query(user_input)
            print(f"\n📋 Answer:\n{response}")
            print("\n" + "-" * 50)

    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("\nMake sure:")
        print("1. data.csv exists in the current directory")
        print("2. Your OPENAI_API_KEY environment variable is set")
        print("3. You have sufficient OpenAI API credits")


if __name__ == "__main__":
    main()

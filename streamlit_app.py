import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app import CSVAgent
import json
import openai
import os

# Configure Streamlit page
st.set_page_config(
    page_title="DATA Agent - Financial Data Analyzer", page_icon="📊", layout="wide"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def load_agent():
    """Load the CSV agent"""
    try:
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY environment variable not set")
            st.info(
                "Please set your OpenAI API key in the sidebar or as an environment variable"
            )
            return None

        agent = CSVAgent("data.csv")
        return agent
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def create_visualizations(agent):
    """Create visualizations for the financial data"""
    df = agent.df

    col1, col2 = st.columns(2)

    with col1:  # Monthly spending trend
        if "Date" in df.columns and "Net Amount" in df.columns:
            monthly_data = (
                df.groupby(df["Date"].dt.to_period("M"))["Net Amount"]
                .sum()
                .reset_index()
            )
            monthly_data["Date"] = monthly_data["Date"].astype(str)

            fig = px.line(
                monthly_data,
                x="Date",
                y="Net Amount",
                title="Monthly Net Amount Trend",
                labels={"Net Amount": "Amount ($)", "Date": "Month"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top expense categories
        if "Code Name" in df.columns and "Net Amount" in df.columns:
            expenses = (
                df[df["Net Amount"] < 0]
                .groupby("Code Name")["Net Amount"]
                .sum()
                .abs()
                .sort_values(ascending=False)
                .head(10)
            )

            fig = px.bar(
                x=expenses.values,
                y=expenses.index,
                orientation="h",
                title="Top 10 Expense Categories",
                labels={"x": "Amount ($)", "y": "Category"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Income vs Expenses pie chart
    if "Net Amount" in df.columns:
        col3, col4 = st.columns(2)

        with col3:
            total_income = df[df["Net Amount"] > 0]["Net Amount"].sum()
            total_expenses = abs(df[df["Net Amount"] < 0]["Net Amount"].sum())

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Income", "Expenses"],
                        values=[total_income, total_expenses],
                        hole=0.3,
                    )
                ]
            )
            fig.update_layout(title="Income vs Expenses")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Transaction count by bank
            if "Bank" in df.columns:
                bank_counts = df["Bank"].value_counts().head(5)
                fig = px.pie(
                    values=bank_counts.values,
                    names=bank_counts.index,
                    title="Transactions by Bank Account",
                )
                st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🤖 DATA Agent - Financial Data Analyzer")
    st.markdown("---")

    # Load agent
    if st.session_state.agent is None:
        with st.spinner("Loading financial data..."):
            st.session_state.agent = load_agent()

    agent = st.session_state.agent

    if agent is None:
        st.error(
            "Failed to load data. Please check that data.csv exists and your OpenAI API key is set."
        )
        st.info(
            "Make sure to:\n1. Have data.csv in the current directory\n2. Set your OpenAI API key in the sidebar or environment variable\n3. Have sufficient OpenAI API credits"
        )
        return

    # Sidebar with data summary
    with st.sidebar:
        st.header("📊 Data Overview")

        summary = agent.data_summary
        st.metric("Total Transactions", summary["total_transactions"])
        st.metric(
            "Date Range",
            f"{summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}",
        )
        st.metric("Total Amount", f"${summary.get('total_amount', 0):,.2f}")
        st.metric("Unique Categories", summary.get("unique_categories", "N/A"))

        st.markdown("---")
        st.header("🔧 Settings")

        # OpenAI API Key input
        # api_key = st.text_input(
        #     "OpenAI API Key",
        #     type="password",
        #     value=os.getenv("OPENAI_API_KEY", ""),
        #     help="Enter your OpenAI API key. You can also set the OPENAI_API_KEY environment variable.",
        # )

        # if api_key and api_key != os.getenv("OPENAI_API_KEY"):
        # os.environ["OPENAI_API_KEY"] = api_key
        # st.success("API key updated!")

        # Model selection
        openai_models = [
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-4o",
        ]

        selected_model = st.selectbox(
            "Select OpenAI Model",
            openai_models,
            index=openai_models.index(agent.model_name)
            if agent.model_name in openai_models
            else 0,
        )

        if selected_model != agent.model_name:
            agent.model_name = selected_model
            st.success(f"Model changed to {selected_model}")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(
        ["💬 Chat with Data", "📈 Visualizations", "📋 Raw Data"]
    )

    with tab1:
        st.header("Ask Questions About Your Financial Data")

        # Suggested questions
        st.subheader("💡 Suggested Questions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("What's my total spending on office costs?", key="btn_office"):
                question = "What's my total spending on office costs?"
                st.session_state.current_question = question

            if st.button("Show me all transactions in June 2022", key="btn_june"):
                question = "Show me all transactions in June 2022"
                st.session_state.current_question = question

        with col2:
            if st.button("Show me top 5 expense categories", key="btn_top5"):
                question = "What are my top 5 expense categories?"
                st.session_state.current_question = question
            if st.button("Plot Monthly Net Amount Trend", key="btn_monthly_trend"):
                question = "Plot Monthly Net Amount Trend"
                st.session_state.current_question = question

        with col3:
            if st.button("How much did I spend on motor vehicles?", key="btn_motor"):
                question = "How much did I spend on motor vehicle costs?"
                st.session_state.current_question = question

            if st.button("Show expense heatmap", key="btn_heatmap"):
                question = "Create a heatmap of my expenses by category"
                st.session_state.current_question = question

        # Chat interface
        st.markdown("---")

        # Display chat history
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i + 1}: {q[:60]}..."):
                st.write("**Question:**", q)
                st.write("**Answer:**", a)

        # Question input
        question = st.text_input(
            "Your question:",
            value=getattr(st.session_state, "current_question", ""),
            key="question_input",
        )

        if st.button("Ask Question", type="primary", key="ask_btn") and question:
            with st.spinner("Analyzing your question..."):
                # Use the new Streamlit-specific query method
                response_data = agent.query_for_streamlit(question)

                # Add to chat history
                st.session_state.chat_history.append((question, response_data))

                # Display response
                st.markdown("### 📋 Answer:")

                if response_data["success"]:
                    # Display the LLM response text
                    st.write(response_data["response"])

                    # Display execution results
                    if response_data["code_blocks"]:
                        st.markdown("### 🔍 Analysis Results:")

                        for block_data in response_data["code_blocks"]:
                            execution_result = block_data["execution_result"]

                            if execution_result["success"]:
                                result = execution_result["result"]
                                result_type = execution_result["type"]
                                var_name = execution_result["variable_name"]

                                # Display the code that was executed
                                with st.expander(
                                    f"📊 Code Block {block_data['block_number']}",
                                    expanded=True,
                                ):
                                    st.code(block_data["code"], language="python")

                                # Display results based on type
                                if result_type == "DataFrame":
                                    st.markdown(
                                        f"**Result ({var_name}):** DataFrame with {len(result)} rows"
                                    )
                                    st.dataframe(result, use_container_width=True)

                                    # Add download button for dataframe
                                    csv = result.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download as CSV",
                                        data=csv,
                                        file_name=f"query_result_{var_name}.csv",
                                        mime="text/csv",
                                        key=f"download_{block_data['block_number']}",
                                    )

                                elif result_type == "Series":
                                    st.markdown(
                                        f"**Result ({var_name}):** Series with {len(result)} items"
                                    )
                                    if len(result) <= 20:
                                        st.dataframe(
                                            result.to_frame(), use_container_width=True
                                        )
                                    else:
                                        st.dataframe(
                                            result.head(20).to_frame(),
                                            use_container_width=True,
                                        )
                                        st.info(
                                            f"Showing first 20 of {len(result)} items"
                                        )

                                else:
                                    st.markdown(f"**Result ({var_name}):**")
                                    st.code(str(result))

                                # Display matplotlib plot if available
                                if execution_result.get("matplotlib_fig") is not None:
                                    st.markdown("**📈 Plot:**")
                                    try:
                                        st.pyplot(execution_result["matplotlib_fig"])
                                    except Exception as plt_err:
                                        st.error(
                                            f"Error displaying matplotlib plot: {plt_err}"
                                        )

                                # Display plotly plot if available
                                if execution_result.get("plotly_fig") is not None:
                                    st.markdown("**📊 Interactive Plot:**")
                                    try:
                                        st.plotly_chart(
                                            execution_result["plotly_fig"],
                                            use_container_width=True,
                                        )
                                    except Exception as plotly_err:
                                        st.error(
                                            f"Error displaying plotly plot: {plotly_err}"
                                        )
                            else:
                                st.error(
                                    f"Error in Code Block {block_data['block_number']}: {execution_result['error']}"
                                )
                                st.code(block_data["code"], language="python")
                else:
                    st.error(f"Error processing query: {response_data['error']}")

                # Clear the question
                if hasattr(st.session_state, "current_question"):
                    del st.session_state.current_question

    with tab2:
        st.header("📈 Data Visualizations")
        create_visualizations(agent)

    with tab3:
        st.header("📋 Raw Data")

        # Data filtering options
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            if "Code Name" in agent.df.columns:
                categories = ["All"] + sorted(agent.df["Code Name"].unique().tolist())
                selected_category = st.selectbox("Filter by Category", categories)

        with filter_col2:
            if "Bank" in agent.df.columns:
                banks = ["All"] + sorted(agent.df["Bank"].unique().tolist())
                selected_bank = st.selectbox("Filter by Bank", banks)

        with filter_col3:
            show_positive_only = st.checkbox("Show only income (positive amounts)")
            show_negative_only = st.checkbox("Show only expenses (negative amounts)")

        # Apply filters
        filtered_df = agent.df.copy()

        if "Code Name" in agent.df.columns and selected_category != "All":
            filtered_df = filtered_df[filtered_df["Code Name"] == selected_category]

        if "Bank" in agent.df.columns and selected_bank != "All":
            filtered_df = filtered_df[filtered_df["Bank"] == selected_bank]

        if show_positive_only and "Net Amount" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Net Amount"] > 0]
        elif show_negative_only and "Net Amount" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Net Amount"] < 0]

        # Display filtered data
        st.write(f"Showing {len(filtered_df)} of {len(agent.df)} transactions")
        st.dataframe(filtered_df, use_container_width=True)

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_financial_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

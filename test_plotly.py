#!/usr/bin/env python3
"""
Test script to verify that the CSV Agent now prefers Plotly over matplotlib
"""

from app import CSVAgent
import os


def test_plotly_preference():
    """Test that the agent generates plotly code for plotting requests"""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    try:
        # Initialize agent
        agent = CSVAgent("data.csv")
        print("✅ Agent initialized successfully")

        # Test question that should generate a plot
        question = "Create a bar chart showing my top 5 expense categories"

        print(f"\n🔍 Testing question: {question}")
        response_data = agent.query_for_streamlit(question)

        if response_data["success"]:
            print("✅ Query successful")

            # Check if code blocks exist
            if response_data["code_blocks"]:
                for i, block in enumerate(response_data["code_blocks"]):
                    code = block["code"]
                    print(f"\n📊 Code Block {i + 1}:")
                    print("-" * 40)
                    print(code)
                    print("-" * 40)

                    # Check if code uses plotly
                    if "px." in code or "go." in code:
                        print("✅ Uses Plotly (preferred)")
                    elif "plt." in code:
                        print("⚠️  Uses matplotlib (not preferred)")
                    else:
                        print("ℹ️  No plotting code detected")

                    # Check execution result
                    exec_result = block["execution_result"]
                    if exec_result["success"]:
                        if exec_result.get("plotly_fig"):
                            print("✅ Plotly figure generated successfully")
                        elif exec_result.get("matplotlib_fig"):
                            print("⚠️  Matplotlib figure generated")
                        else:
                            print("ℹ️  No figure generated")
                    else:
                        print(f"❌ Execution failed: {exec_result['error']}")
            else:
                print("ℹ️  No code blocks generated")
        else:
            print(f"❌ Query failed: {response_data['error']}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_plotly_preference()

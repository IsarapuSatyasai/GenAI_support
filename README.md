** LangGraph ** allows the model to use this pipeline as a state machine. We will define a **State** to hold the data as it moves through the pipeline, **Nodes** for each functional step (Extract, Validate, Export), and **Edges** to connect them in a linear workflow.

I swapped the raw OpenAI client for LangChain's `AzureChatOpenAI` using `.with_structured_output()`, which natively handles Pydantic schema enforcement and removes the need for manual JSON parsing.

### LangGraph Implementation

```python
%pip install langchain-openai langgraph pydantic pandas openpyxl --quiet
%restart_python

```

```python
import os
import pandas as pd
from datetime import datetime
from typing import TypedDict, List, Dict, Optional, Any
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. Credentials & Setup
# ==========================================
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="azure-key-vault-scope", key="openai-apim-api-key")
os.environ["AZURE_OAI_ENDPOINT"] = "https://msc-apim-gailtu-prd.azure-api.net/openai-api"
os.environ["STORAGE_ACCOUNT"] = "mscstagailtuprd03"
os.environ["SA_ACCESSKEY"] = dbutils.secrets.get(scope="azure-key-vault-scope", key="sta-eu-accesskey")

# Initialize LangChain's Azure Chat Model
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    openai_api_version="2024-08-01-preview",
    azure_deployment="gpt-4o", # Replace if your APIM uses a different deployment name
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.0
)
```

```python
# ==========================================
# 2. Pydantic Schemas
# ==========================================
class MetricValue(BaseModel):
    value: Optional[float] = Field(None, description="Extracted numeric value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class ExtractedMetrics(BaseModel):
    company: Optional[str] = None
    period: Optional[str] = None
    revenue: Optional[MetricValue] = None
    net_sales: Optional[MetricValue] = None
    gross_margin: Optional[MetricValue] = None
    operating_income: Optional[MetricValue] = None
    net_income: Optional[MetricValue] = None
    orders: Optional[MetricValue] = None
    bookings: Optional[MetricValue] = None

# Bind the Pydantic model to the LLM to force structured output
structured_llm = llm.with_structured_output(ExtractedMetrics)
```
```python
# 3. Define Graph State
class GraphState(TypedDict):
    """Represents the state of our graph passing between nodes."""
    raw_data: str
    extracted_metrics: Optional[ExtractedMetrics]
    valid_metrics: List[Dict]
    low_confidence_metrics: List[Dict]
    company: Optional[str]
    period: Optional[str]
    excel_path: Optional[str]
```
```python
# 4. Define Graph Nodes
def extract_node(state: GraphState) -> GraphState:
    """Node 1: Extracts financial metrics using LLM."""
    print("-> Running Extraction Node")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise financial data extraction assistant. "
                   "Extract the requested metrics from the given text or XML. "
                   "For every numeric metric, also provide a confidence score between 0 and 1. "
                   "If a metric is not found, set value to null and confidence to 0.0. "
                   "Confidence should reflect how clearly the value appears in the text."),
        ("user", "Data to process:\n{data}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"data": state["raw_data"]})
    
    return {"extracted_metrics": result}

def validate_node(state: GraphState) -> GraphState:
    """Node 2: Validates metrics based on confidence threshold."""
    print("-> Running Validation Node")
    extracted = state["extracted_metrics"]
    threshold = 0.7
    
    metric_fields = [
        "revenue", "net_sales", "gross_margin",
        "operating_income", "net_income", "orders", "bookings"
    ]

    valid = []
    low_confidence = []

    for field in metric_fields:
        metric = getattr(extracted, field)
        if metric is None or metric.value is None:
            continue

        item = {
            "Variable": field.replace("_", " ").title(),
            "Value": metric.value,
            "Score": metric.confidence
        }

        if metric.confidence >= threshold:
            valid.append(item)
        else:
            low_confidence.append(item)

    return {
        "valid_metrics": valid,
        "low_confidence_metrics": low_confidence,
        "company": extracted.company,
        "period": extracted.period
    }

def generate_excel_node(state: GraphState) -> GraphState:
    """Node 3: Formats data and writes to Excel."""
    print("-> Running Excel Generation Node")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"financial_report_{timestamp}.xlsx"
    # Keeping your original Databricks workspace path
    output_path = f"/Workspace/Users/greeshmitham@delagelanden.com/financial_excel_sheet_func/{output_filename}"
    
    sheet_configs = [{"sheet_name": "Extracted Metrics", "data": state["valid_metrics"]}]
    if state["low_confidence_metrics"]:
        sheet_configs.append({"sheet_name": "Needs Review", "data": state["low_confidence_metrics"]})
        
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for config in sheet_configs:
            sheet_name = config["sheet_name"]
            data = config.get("data", [])
            df = pd.DataFrame(data)
            
            if not df.empty:
                if 'No' not in df.columns:
                    df.insert(0, 'No', range(1, len(df) + 1))
                
                columns = ['No', 'Variable', 'Value', 'Score']
                for col in columns:
                    if col not in df.columns:
                        df[col] = None
                df = df[columns]
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    worksheet.column_dimensions[chr(65 + idx)].width = 18
                for cell in worksheet[1]:
                    cell.font = cell.font.copy(bold=True)
            else:
                # Handle empty sheet case gracefully
                pd.DataFrame(columns=['No', 'Variable', 'Value', 'Score']).to_excel(writer, sheet_name=sheet_name, index=False)
                
    return {"excel_path": output_path}
```
```python
# ==========================================
# 5. Build and Compile the Graph
# ==========================================
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("extract", extract_node)
workflow.add_node("validate", validate_node)
workflow.add_node("export", generate_excel_node)

# Define Edges (Linear flow)
workflow.add_edge(START, "extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", "export")
workflow.add_edge("export", END)

# Compile
app = workflow.compile()
```
```python
# ==========================================
# 6. Execute the Graph
# ==========================================
if __name__ == "__main__":
    sample_xml = """
    <report>
        <company>ASML</company>
        <period>Q2 2025</period>
        <revenue currency="EUR">7.1</revenue>
        <net_sales>6.9</net_sales>
        <gross_margin>51.2</gross_margin>
        <operating_income>2.15</operating_income>
        <net_income>1.85</net_income>
        <orders>8.4</orders>
        <bookings>9.1</bookings>
    </report>
    """

    # Initial state
    inputs = {"raw_data": sample_xml.strip()}
    
    # Run the graph
    final_state = app.invoke(inputs)
    
    print("\n=== Workflow Completed ===")
    print(f"Company: {final_state.get('company')}")
    print(f"Period : {final_state.get('period')}")
    print(f"Valid Metrics: {len(final_state.get('valid_metrics', []))}")
    print(f"Review Metrics: {len(final_state.get('low_confidence_metrics', []))}")
    print(f"Report saved to: {final_state.get('excel_path')}")

```

### Key Differences & Benefits

1. **State Management (`GraphState`)**: Instead of passing arguments sequentially between loose functions, everything is stored in a single, typed dictionary. This makes debugging incredibly easy because you can inspect the exact `state` at any point in the graph.
2. **`with_structured_output`**: Notice how the raw JSON parsing (`json.loads()`) is entirely gone from `extract_node`. LangChain handles injecting the Pydantic schema into the API call and natively deserializing the response, making it highly robust against formatting errors.
3. **Extensibility**: Want to add a node that emails the Excel file if low-confidence metrics are found? Or a node that hits a web search if `company` is `null`? In LangGraph, you just define a new node and add a conditional edge, rather than cluttering your main execution block.

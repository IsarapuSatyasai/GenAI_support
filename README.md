Here is the updated LangGraph pipeline. I have incorporated the new Pydantic schema fields for the **Year** and **Page Number**, and updated the Excel generation node to automatically construct clickable PDF hyperlinks using Excel's native `=HYPERLINK()` function.

### Updated LangGraph Pipeline

```python
%pip install langchain-openai langgraph pydantic pandas openpyxl --quiet
%restart_python

```

```python
import os
import pandas as pd
from datetime import datetime
from typing import TypedDict, List, Dict, Optional
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

```

# 1. Credentials & Setup
```python
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="azure-key-vault-scope", key="openai-apim-api-key")
os.environ["AZURE_OAI_ENDPOINT"] = "https://msc-apim-gailtu-prd.azure-api.net/openai-api"
os.environ["STORAGE_ACCOUNT"] = "mscstagailtuprd03"
os.environ["SA_ACCESSKEY"] = dbutils.secrets.get(scope="azure-key-vault-scope", key="sta-eu-accesskey")

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    openai_api_version="2024-08-01-preview",
    azure_deployment="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.0
)
```

# 2. Expanded Pydantic Schemas
```python
class MetricValue(BaseModel):
    value: Optional[float] = Field(None, description="Extracted numeric value")
    year: Optional[int] = Field(None, description="The financial year this metric applies to (e.g., 2024, 2025)")
    page_number: Optional[int] = Field(None, description="The page number in the source document where this value was found")
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

# Using function_calling to ensure Azure API compatibility
structured_llm = llm.with_structured_output(ExtractedMetrics, method="function_calling")
```

# 3. Define Graph State
```python
class GraphState(TypedDict):
    raw_data: str
    pdf_path: str  # Added to track the source file for hyperlinks
    extracted_metrics: Optional[ExtractedMetrics]
    valid_metrics: List[Dict]
    low_confidence_metrics: List[Dict]
    company: Optional[str]
    period: Optional[str]
    excel_path: Optional[str]
```

# 4. Define Graph Nodes
```python
def extract_node(state: GraphState) -> GraphState:
    """Node 1: Extracts financial metrics, years, pages, and confidence."""
    print("-> Running Extraction Node")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise financial data extraction assistant. "
                   "Extract the requested metrics from the provided document. "
                   "For every numeric metric, you MUST extract the value, the specific year it belongs to, "
                   "and the page number where you found it. "
                   "Also provide a confidence score between 0 and 1. "
                   "If a metric is not found, set value to null and confidence to 0.0."),
        ("user", "Data to process:\n{data}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"data": state["raw_data"]})
    
    return {"extracted_metrics": result}

def validate_node(state: GraphState) -> GraphState:
    """Node 2: Validates metrics and flattens data for Excel."""
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
            "Year": metric.year,
            "Page": metric.page_number,
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
    """Node 3: Formats data, generates hyperlinks, and writes to Excel."""
    print("-> Running Excel Generation Node")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"financial_report_{timestamp}.xlsx"
    output_path = f"/Workspace/Users/greeshmitham@delagelanden.com/financial_excel_sheet_func/{output_filename}"
    pdf_source = state.get("pdf_path", "source_document.pdf")
    
    sheet_configs = [{"sheet_name": "Extracted Metrics", "data": state["valid_metrics"]}]
    if state["low_confidence_metrics"]:
        sheet_configs.append({"sheet_name": "Needs Review", "data": state["low_confidence_metrics"]})
        
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for config in sheet_configs:
            sheet_name = config["sheet_name"]
            data = config.get("data", [])
            df = pd.DataFrame(data)
            
            if not df.empty:
                # Add Index
                if 'No' not in df.columns:
                    df.insert(0, 'No', range(1, len(df) + 1))
                
                # Construct Excel Hyperlink Formula for PDF tracing
                if 'Source Link' not in df.columns:
                    df['Source Link'] = df['Page'].apply(
                        lambda p: f'=HYPERLINK("{pdf_source}#page={p}", "Page {p}")' if pd.notnull(p) else "N/A"
                    )
                
                # Enforce column order
                columns = ['No', 'Variable', 'Value', 'Year', 'Score', 'Source Link']
                for col in columns:
                    if col not in df.columns:
                        df[col] = None
                df = df[columns]
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                worksheet = writer.sheets[sheet_name]
                # Auto-adjust column width
                for idx, col in enumerate(df.columns):
                    worksheet.column_dimensions[chr(65 + idx)].width = 20
                # Bold headers
                for cell in worksheet[1]:
                    cell.font = cell.font.copy(bold=True)
            else:
                pd.DataFrame(columns=['No', 'Variable', 'Value', 'Year', 'Score', 'Source Link']).to_excel(writer, sheet_name=sheet_name, index=False)
                
    return {"excel_path": output_path}
```
# 5. Build and Compile the Graph
```python
workflow = StateGraph(GraphState)

workflow.add_node("extract", extract_node)
workflow.add_node("validate", validate_node)
workflow.add_node("export", generate_excel_node)

workflow.add_edge(START, "extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", "export")
workflow.add_edge("export", END)

app = workflow.compile()
```

# 6. Execute the Graph
```python
if __name__ == "__main__":
    # Simulated input string (in production, this would be text extracted from the PDF with page tags)
    sample_data = """
    [Page 12] ASML Financial Results Q2 2025.
    [Page 14] The company reported a total revenue of 7.1 billion EUR for 2025. 
    Net sales reached 6.9 billion EUR. Gross margin remained steady at 51.2%.
    [Page 15] Operating income for 2025 was 2.15 billion, and net income finalized at 1.85 billion.
    [Page 18] Total orders stood at 8.4 billion with system bookings increasing to 9.1 billion.
    """
    
    # Initial state providing the raw text and the reference PDF path for hyperlinks
    inputs = {
        "raw_data": sample_data.strip(),
        "pdf_path": "https://company.sharepoint.com/sites/Finance/Reports/ASML_Q2_2025.pdf"
    }
    
    final_state = app.invoke(inputs)
    
    print("\n=== Workflow Completed ===")
    print(f"Company: {final_state.get('company')}")
    print(f"Period : {final_state.get('period')}")
    print(f"Valid Metrics: {len(final_state.get('valid_metrics', []))}")
    print(f"Report saved to: {final_state.get('excel_path')}")

```

```python
%pip install langchain langchain-openai langgraph pydantic azure-storage-blob openpyxl pypdf
dbutils.library.restartPython()

```

```python
import os
import pandas as pd
from datetime import datetime
from typing import TypedDict, List, Dict, Optional, Any
from pydantic import BaseModel, Field

# LangChain & LangGraph imports
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader

# Azure Blob Storage import
from azure.storage.blob import BlobServiceClient
```
```python
# 1. Credentials & LLM Setup

os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(
    scope="azure-key-vault-scope",
    key="openai-apim-api-key"
)
os.environ["AZURE_OAI_ENDPOINT"] = "https://msc-apim-gailtu-prd.azure-api.net/openai-api"
os.environ["STORAGE_ACCOUNT"] = "mscstagailtuprd03"
os.environ["SA_ACCESSKEY"] = dbutils.secrets.get(
    scope="azure-key-vault-scope",
    key="sta-eu-accesskey"
)
os.environ["ENVIRONMENT"] = "prd"

# Use AzureChatOpenAI instead of standard ChatOpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    api_version="2024-08-01-preview",
    api_key=os.environ["OPENAI_API_KEY"],
    azure_deployment="gpt-4o",  # NOTE: Update this if your Azure deployment name differs
    temperature=0.0
)
print("AzureChatOpenAI client initialized")
```
```python
# 2. Azure Blob Storage Helpers
blob_service_client = BlobServiceClient(
    account_url=f"https://{os.environ['STORAGE_ACCOUNT']}.blob.core.windows.net",
    credential=os.environ["SA_ACCESSKEY"]
)
CONTAINER_NAME = "projects"

def download_blob_to_local(blob_path: str, local_path: str):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)
    with open(local_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    print(f"Downloaded {blob_path} to Databricks local path {local_path}")

def upload_local_to_blob(local_path: str, blob_path: str):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {local_path} to Azure Blob {blob_path}")
```
```python
# 3. Pydantic Schemas
class MetricValue(BaseModel):
    value: Optional[float] = Field(None, description="Extracted numeric value")
    year: Optional[int] = Field(None, description="The financial year this metric applies to (e.g., 2023, 2024)")
    page_number: Optional[int] = Field(None, description="The page number in the document where this value was found")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class ExtractedMetrics(BaseModel):
    company: Optional[str] = None
    period: Optional[str] = None
    revenue: Optional[MetricValue] = Field(None, description="Also known as Net Sales")
    net_sales: Optional[MetricValue] = None
    gross_margin: Optional[MetricValue] = Field(None, description="Also known as Gross Income")
    operating_income: Optional[MetricValue] = Field(None, description="Also known as EBIT")
    net_income: Optional[MetricValue] = None
    orders: Optional[MetricValue] = None
    bookings: Optional[MetricValue] = None

# Bind schema natively
structured_llm = llm.with_structured_output(ExtractedMetrics)
```
```python
# 4. Define Graph State
class GraphState(TypedDict):
    raw_data: str
    pdf_path: str
    extracted_metrics: Optional[ExtractedMetrics]
    valid_metrics: List[Dict]
    low_confidence_metrics: List[Dict]
    company: Optional[str]
    period: Optional[str]
    excel_path: Optional[str]
```
```python
# 5. Define Graph Nodes
def extract_node(state: GraphState) -> GraphState:
    print("-> Running Extraction Node")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise financial data extraction assistant. "
                   "Extract the requested metrics from the provided document. "
                   "For every numeric metric, you MUST extract the value, the specific year it belongs to, "
                   "and the page number where you found it. "
                   "Pay close attention to alternative names (e.g., Net Sales for Revenue, EBIT for Operating Income). "
                   "If a metric is not found, set value to null and confidence to 0.0."),
        ("user", "Data to process:\n{data}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"data": state["raw_data"]})
    
    return {"extracted_metrics": result}

def validate_node(state: GraphState) -> GraphState:
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
    print("-> Running Excel Generation Node")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"financial_report_{timestamp}.xlsx"
    
    # Save to Databricks local /tmp directory for processing
    local_output_dir = "/tmp/financial_reports"
    os.makedirs(local_output_dir, exist_ok=True)
    local_output_path = os.path.join(local_output_dir, output_filename)
    
    pdf_source = state.get("pdf_path", "source_document.pdf")
    
    sheet_configs = [{"sheet_name": "Extracted Metrics", "data": state["valid_metrics"]}]
    if state["low_confidence_metrics"]:
        sheet_configs.append({"sheet_name": "Needs Review", "data": state["low_confidence_metrics"]})
        
    with pd.ExcelWriter(local_output_path, engine='openpyxl') as writer:
        for config in sheet_configs:
            sheet_name = config["sheet_name"]
            data = config.get("data", [])
            df = pd.DataFrame(data)
            
            if not df.empty:
                if 'No' not in df.columns:
                    df.insert(0, 'No', range(1, len(df) + 1))
                if 'Source Link' not in df.columns:
                    pdf_filename = os.path.basename(pdf_source)
                    df['Source Link'] = df['Page'].apply(
                        lambda p: f'=HYPERLINK("{pdf_filename}#page={p}", "Page {p}")' if pd.notnull(p) else "N/A"
                    )
                
                columns = ['No', 'Variable', 'Value', 'Year', 'Score', 'Source Link']
                for col in columns:
                    if col not in df.columns:
                        df[col] = None
                df = df[columns]
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    worksheet.column_dimensions[chr(65 + idx)].width = 20
                for cell in worksheet[1]:
                    cell.font = cell.font.copy(bold=True)
            else:
                pd.DataFrame(columns=['No', 'Variable', 'Value', 'Year', 'Score', 'Source Link']).to_excel(writer, sheet_name=sheet_name, index=False)
                
    return {"excel_path": local_output_path}
```
```python
# 6. Build and Compile the Graph
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
```python
# 7. Helper: Extract text with pages
def extract_text_with_pages(pdf_file_path: str) -> str:
    print(f"Loading PDF: {pdf_file_path}")
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    
    formatted_text = ""
    for doc in documents:
        page_num = doc.metadata.get('page', 0) + 1 
        formatted_text += f"\n--- [Page {page_num}] ---\n{doc.page_content}\n"
        
    return formatted_text
```
```python
# 8. Main Execution
if __name__ == "__main__":
    
    # Define Azure Blob paths based on URL
    input_blob_path = "FinancialSpreading/InputData/NL/Adyen/CL Adyen NV_Jurgen Geerts.pdf"
    output_blob_folder = "FinancialSpreading/InputData/NL/Adyen"
    
    pdf_file_name = os.path.basename(input_blob_path)
    local_pdf_path = f"/tmp/{pdf_file_name}"
    
    try:
        # Step A: Download PDF from Azure to Databricks Driver
        download_blob_to_local(input_blob_path, local_pdf_path)
        
        # Step B: Parse the PDF
        extracted_raw_text = extract_text_with_pages(local_pdf_path)
        
        # Step C: Prepare Inputs
        inputs = {
            "raw_data": extracted_raw_text,
            "pdf_path": local_pdf_path
        }
        
        print("Initiating LangGraph Pipeline...")
        final_state = app.invoke(inputs)
        
        print("\n=== Workflow Completed ===")
        print(f"Company: {final_state.get('company')}")
        print(f"Valid Metrics Found: {len(final_state.get('valid_metrics', []))}")
        
        # Step D: Upload generated Excel back to Azure Blob Storage
        local_excel_path = final_state.get('excel_path')
        if local_excel_path and os.path.exists(local_excel_path):
            excel_filename = os.path.basename(local_excel_path)
            output_blob_path = f"{output_blob_folder}/{excel_filename}"
            
            upload_local_to_blob(local_excel_path, output_blob_path)
```
            
            final_url = f"https://{os.environ['STORAGE_ACCOUNT']}.blob.core.windows.net/projects/{output_blob_path}"
            print(f"Output Successfully Uploaded To: {final_url}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

```

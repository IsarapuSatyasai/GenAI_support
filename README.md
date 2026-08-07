
```python
%pip install openai pydantic pandas openpyxl --quiet
%restart_python
```

```python
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime
import json
import os
```

```python
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
```

```python
def load_data(source: str, source_type: str = "text") -> str:
    if source_type in ["text", "xml"]:
        return source.strip()
    elif source_type == "file":
        with open(source, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
```

```python
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

client = AzureOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    api_version="2024-08-01-preview",
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"]
)

print("Azure OpenAI client initialized successfully")
```

```python
def extract_metrics_with_confidence(data: str, model: str = "gpt-4o") -> ExtractedMetrics:

    system_prompt = """
    You are a precise financial data extraction assistant.
    Extract the requested metrics from the given text or XML.
    For every numeric metric, also provide a confidence score between 0 and 1.
    
    Rules:
    - Return ONLY valid JSON matching the required schema.
    - If a metric is not found, set value to null and confidence to 0.0.
    - Confidence should reflect how clearly and unambiguously the value appears in the text.
    """

    user_prompt = f"""
    Extract the following metrics and return them in this exact JSON structure:

    {{
      "company": "string or null",
      "period": "string or null",
      "revenue": {{"value": number or null, "confidence": float}},
      "net_sales": {{"value": number or null, "confidence": float}},
      "gross_margin": {{"value": number or null, "confidence": float}},
      "operating_income": {{"value": number or null, "confidence": float}},
      "net_income": {{"value": number or null, "confidence": float}},
      "orders": {{"value": number or null, "confidence": float}},
      "bookings": {{"value": number or null, "confidence": float}}
    }}

    Data:
    {data}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    raw_json = response.choices[0].message.content
    parsed = json.loads(raw_json)
    return ExtractedMetrics(**parsed)
```

```python
def validate_metrics(extracted: ExtractedMetrics, threshold: float = 0.7) -> Dict:
    """
    Checks confidence scores.
    Returns a dictionary with:
    - valid_metrics
    - low_confidence_metrics (for review / re-extraction)
    """
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
```

```python
def excel_write(sheet_configs, output_filename=None, prefix="financial_report"):
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{prefix}_{timestamp}.xlsx"
    
    output_path = f"/Workspace/Users/greeshmitham@delagelanden.com/financial_excel_sheet_func/{output_filename}"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for config in sheet_configs:
            sheet_name = config["sheet_name"]
            data = config.get("data", [])
            
            df = pd.DataFrame(data)
            
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
    
    print(f"Excel file created: {output_path}")
    return output_path
```

```python
# 1. Sample data
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

# 2. Load data
raw_data = load_data(sample_xml, source_type="xml")
print("Data loaded successfully.\n")

# 3. Extract metrics using LLM
print("Extracting metrics with confidence scores...")
extracted = extract_metrics_with_confidence(raw_data, model="gpt-4o")
print(extracted.model_dump_json(indent=2))
print()

# 4. Validate metrics
validation_result = validate_metrics(extracted, threshold=0.7)

print("=== Validation Summary ===")
print(f"Company : {validation_result['company']}")
print(f"Period  : {validation_result['period']}")
print(f"Valid metrics          : {len(validation_result['valid_metrics'])}")
print(f"Low confidence metrics : {len(validation_result['low_confidence_metrics'])}")

if validation_result["low_confidence_metrics"]:
    print("\nMetrics marked for review (confidence < 0.7):")
    for item in validation_result["low_confidence_metrics"]:
        print(f"  - {item['Variable']}: {item['Value']} (Score: {item['Score']})")

# 5. Prepare data for Excel
sheet_configs = [
    {
        "sheet_name": "Extracted Metrics",
        "data": validation_result["valid_metrics"]
    }
]

# Optionally add low confidence metrics in a separate sheet
if validation_result["low_confidence_metrics"]:
    sheet_configs.append({
        "sheet_name": "Needs Review",
        "data": validation_result["low_confidence_metrics"]
    })

# 6. Generate Excel
excel_path = excel_write(sheet_configs)
print(f"\nProcess completed. Excel saved at:\n{excel_path}")
```

---

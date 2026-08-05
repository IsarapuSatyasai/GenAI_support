### Cell 1: Install Required Libraries

```python
%pip install openai pydantic --quiet
```

**Explanation:**  
Installs the necessary packages. `pydantic` is required for strict schema validation. `--quiet` keeps the output clean.

---

### Cell 2: Import Libraries

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
import json
```

**Explanation:**  
Imports everything we need. We keep the imports minimal and clean.

---

### Cell 3: Define Pydantic Models (Schema)

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

**Explanation:**  
This is the core schema Soumya requested.  
Every numeric metric now returns both `value` and `confidence`.  
Pydantic will validate the structure automatically.

---

### Cell 4: Generic Data Loading Function

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

**Explanation:**  
This is the temporary generic loader we agreed on.  
Once Soumya finishes her Excel/document loading part, we can easily replace the inside of this function.

---

### Cell 5: Sample Data (for testing)

```python
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

raw_data = load_data(sample_xml, source_type="xml")
print("Data loaded successfully. Length:", len(raw_data))
```

**Explanation:**  
Using a realistic sample so you can test immediately.  
You can later replace this with real files.

---

### Cell 6: Initialize LLM Client

```python
# client = OpenAI(
#     api_key = "",
#     base_url = ""
# )
```

**Explanation:**  
Never hardcode keys in production. Use Databricks secrets when moving to real environment.

---

### Cell 7: Metric Extraction Function (with Confidence)

```python

def extract_metrics_with_confidence(data: str, model: str = "gpt-4o-mini") -> ExtractedMetrics:

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

    # Validate against Pydantic model
    return ExtractedMetrics(**parsed)
```

**Explanation:**  
- Uses `temperature=0.0` for consistency  
- Forces JSON output  
- Validates the response with Pydantic  
- Returns both value + confidence for every metric

---

### Cell 8: Run the Extraction

```python
print("Extracting metrics with confidence scores...\n")

result = extract_metrics_with_confidence(raw_data)

print(result.model_dump_json(indent=2))
```

**Explanation:**  
This cell runs the full flow and prints the structured output.

---

### Expected Output Example:

```json
{
  "company": "ASML",
  "period": "Q2 2025",
  "revenue": {
    "value": 7.1,
    "confidence": 0.95
  },
  "net_sales": {
    "value": 6.9,
    "confidence": 0.93
  },
  "gross_margin": {
    "value": 51.2,
    "confidence": 0.94
  },
  ...
}
```

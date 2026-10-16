# Financial Spreading — Multi-Year Extraction

## Overview
This branch (`multi-year`) extends the single-year extraction pipeline to support **1–3 years** of financial data from the same PDF.

The LLM returns one `ExtractedAnswer` per variable per year.  
Excel output places year blocks **side-by-side**.

## Changes

### 1. `prompts/system_prompt.py`
Remove the hard “latest period only” rule.

**Replace rule 1 with:**
```python
1. **All available years** - Extract values for every financial year present in the document (typically 1–3 years). Do not ignore comparative periods. For each variable return one entry per year found.
```

### 2. `models.py`
Move `year` onto each answer so the same schema can hold multiple years.

```python
class ExtractedAnswer(BaseModel):
    """For each metric extraction, LLM should follow this schema"""
    worksheet: str = Field(description="The worksheet name this variable belongs to")
    variable: str = Field(description="The name of the variable being extracted")
    year: Optional[int] = Field(
        default=None,
        description="Financial year this value belongs to (e.g. 2024). Null if not found."
    )
    answer: str = Field(description="The extracted value as given in the source document, or 'N/A' if not found")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 based on how clearly the document states the value."
    )
    page_number: int = Field(
        description="The page number where the information was found. Set to -1 if not found."
    )
    source_fields: List[str] = Field(
        default_factory=list,
        description="List of fields in the source document that were used to extract the answer."
    )
    formula: str = Field(
        default=None,
        description="Formula used to calculate the answer based on source fields. Null if 1-to-1 mapping."
    )
    source_link: str = Field(
        default="N/A",
        description="Hyperlink formula pointing to the source PDF page. Populated post-extraction."
    )


class ExtractionResponse(BaseModel):
    """Overall extracted response from the LLM model."""
    currency: Optional[str] = Field(default=None, description="Currency used, e.g. 'EUR', 'GBP', 'USD'.")
    scale: Optional[str] = Field(default=None, description="Scale used, e.g. 'thousands' or 'millions'.")
    answers: List[ExtractedAnswer] = Field(
        default_factory=list,
        description="List of extracted answers. One entry per variable per year."
    )
    errors: List[str] = Field(default_factory=list)
```

### 3. `nodes/answer_questions.py`
No structural change required. Optional clarity in the user prompt:

```python
user_content = (
    f"Please extract the following variables for every year present in the document:\n{USER_PROMPT}\n\n"
    f"Document context:\n{full_context}\n\n"
)
```

### 4. `nodes/create_excel_output.py`
Replace the DataFrame builder so years become side-by-side column groups.

```python
def build_excel_dataframes(answers: list[dict]) -> dict[str, pd.DataFrame]:
    """Group answers by worksheet and pivot years into side-by-side columns."""
    worksheet_dict = {}

    for a in answers:
        ws = a["worksheet"]
        worksheet_dict.setdefault(ws, []).append(a)

    result = {}
    value_cols = ["answer", "confidence", "page_number", "source_fields", "formula", "source_link"]

    for ws, rows in worksheet_dict.items():
        df = pd.DataFrame(rows)
        if df.empty:
            result[ws] = df
            continue

        years = sorted([y for y in df["year"].dropna().unique()])
        base = df[["variable"]].drop_duplicates().reset_index(drop=True)

        for year in years:
            year_df = df[df["year"] == year][["variable"] + value_cols].copy()
            year_df = year_df.rename(columns={c: f"{year}_{c}" for c in value_cols})
            base = base.merge(year_df, on="variable", how="left")

        result[ws] = base

    return result
```

Update the chain name reference:

```python
excel_preparation_chain = (
    RunnableLambda(clean_llm_output, name="Clean extracted metrics from LLM")
    | RunnableLambda(build_excel_dataframes, name="Build excel dataframes")
)
```

### Git Commit:

```text
feat: add multi-year financial extraction support
```

```text
- Relax system prompt to extract all available years (1–3)
- Move year field into ExtractedAnswer schema
- Pivot Excel output so year blocks appear side-by-side
- Metrics missing a year produce blank cells instead of errors
```

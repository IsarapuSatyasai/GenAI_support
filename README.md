```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ExtractedAnswer(BaseModel):
    """Schema for individual metric extractions."""
    worksheet: str = Field(description="The worksheet name this variable belongs to")
    variable: str = Field(description="The name of the variable being extracted")
    answer: str = Field(description="The extracted value as given in the source document, or 'N/A' if not found")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 based on how clearly the document states the value."
    )
    page_number: int = Field(
        description="The page number where the information was found. Set to -1 if not found."
    )
    source_fields: List[str] = Field(
        default_factory=list, 
        description="List of fields in the source document used to extract the answer."
    )
    formula: Optional[str] = Field(
        default=None, 
        description="Formula used to calculate the answer based on source fields."
    )
    # ADDED: Container for the generated hyperlink formula
    source_link: Optional[str] = Field(
        default="N/A", 
        description="Excel HYPERLINK formula pointing to the exact source page in Azure Blob."
    )

class ExtractionResponse(BaseModel):
    """Overall extracted response from the LLM."""
    year: Optional[int] = Field(default=None, description="The financial year this metric applies to.")
    currency: Optional[str] = Field(default=None, description="The currency used (e.g., 'EUR', 'USD').")
    scale: Optional[str] = Field(default=None, description="The scale used (e.g., 'thousands', 'millions').")
    answers: List[ExtractedAnswer] = Field(default_factory=list, description="List of extracted answers.")
    errors: List[str] = Field(default_factory=list)
```

```python
from typing import TypedDict, List, Dict, Any, Optional

class FinancialGraphState(TypedDict, total=False):
    llm: Any
    blob_pdf_url: str
    excel_data: List[Dict[str, Any]]
    selected_pages: List[Dict[str, Any]]
    errors: List[str]
    extraction_response: Optional[ExtractionResponse]
```

```python
"""Node: Answer Questions using LLM with Source Deep Linking."""

from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from graph.graph_state import FinancialGraphState
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.user_prompt import get_user_prompt
from models import ExtractionResponse, ExtractedAnswer


def build_hyperlink(page_number: Optional[int], blob_pdf_url: str) -> str:
    """Helper function applying the exact Excel HYPERLINK formula logic."""
    if page_number is not None and page_number != -1 and blob_pdf_url:
        return f'=HYPERLINK("{blob_pdf_url}#page={page_number}", "Page {page_number}")'
    return "N/A"


def answer_questions(state: FinancialGraphState) -> dict:
    """Extract metrics via LLM and enrich each answer with an Excel hyperlink."""
    llm = state.get("llm")
    structured_llm = llm.with_structured_output(ExtractionResponse)

    excel_data = state.get("excel_data", [])
    errors = state.get("errors", [])
    selected_pages = state.get("selected_pages", [])
    blob_pdf_url = state.get("blob_pdf_url", "")

    USER_PROMPT = get_user_prompt(excel_data)
    
    # Prepare Document Context with page markers
    full_context = "\n\n".join(
        [f"--- Page {i+1} ---\n{page.get('text', '')}" for i, page in enumerate(selected_pages)]
    )

    user_content = (
        f"Please extract the following variables:\n{USER_PROMPT}\n\n"
        f"Document context:\n{full_context}\n\n"
        "IMPORTANT: You MUST identify the source page number referencing the '--- Page X ---' headers. "
        "Set page_number to -1 if the information is not present."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT), 
        HumanMessage(content=user_content)
    ]
    
    try:
        response: ExtractionResponse = structured_llm.invoke(messages)
        
        # Merge previous pipeline errors if present
        if errors:
            response.errors.extend(errors)
            
        # Post-process: Inject source hyperlinks into each extracted answer
        for answer in response.answers:
            answer.source_link = build_hyperlink(answer.page_number, blob_pdf_url)
            
        return {"extraction_response": response, "errors": response.errors}
        
    except Exception as e:
        error_msg = f"Bulk LLM extraction error: {str(e)}"
        errors.append(error_msg)
        return {
            "extraction_response": ExtractionResponse(errors=[error_msg]),
            "errors": errors
        }
```

```python
feat(extraction): add PDF source hyperlinks to extracted metrics

Updated the extraction pipeline to automatically generate Excel-ready hyperlink formulas that deep-link to the specific source page in the Azure Blob PDF. 

- Added `source_link` field to the `ExtractedAnswer` Pydantic model to store the formula.
- Tracked `blob_pdf_url` in `FinancialGraphState` to provide the base URL.
- Created `build_hyperlink` helper function in the `answer_questions` node to apply the precise formatting logic.
- Implemented a post-processing step after the LLM invocation to map integer page numbers to the `=HYPERLINK` formula, preventing the LLM from hallucinating URL syntax.
- Updated LLM prompt instructions to strictly enforce a `-1` fallback for missing page numbers, ensuring broken links are not generated.
```

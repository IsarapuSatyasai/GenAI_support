```python
"""Node: Answer Questions using LLM.

Iterates over each Excel row and extracts the corresponding financial
data point from the PDF context in a single bulk extraction.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from graph.graph_state import FinancialGraphState


# Updated Pydantic model
class ExtractedAnswer(BaseModel):
    worksheet: str = Field(description="The worksheet name this variable belongs to")
    variable: str = Field(description="The name of the variable being extracted")
    description: str = Field(default="", description="Optional description of the variable")
    answer: str = Field(description="The extracted value, or 'N/A' if not found")
    year: Optional[int] = Field(
        default=None, 
        description="The financial year this metric applies to (e.g., 2023, 2024). Set to null if not found."
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 based on how clearly the document states the value."
    )
    page_number: int = Field(
        description="The page number where the information was found. Set to -1 if not found."
    )

class ExtractionResponse(BaseModel):
    answers: List[ExtractedAnswer] = Field(
        default_factory=list, 
        description="List of extracted answers for all requested variables."
    )
    errors: List[str] = Field(default_factory=list)

# Updated System Prompt
SYSTEM_PROMPT = (
    "You are an intelligent financial assistant helping credit teams analyze "
    "and extract valuable insights from annual reports of companies. "
    "Extract the requested financial data points from the provided document context. "
    "For each metric, extract the value, the financial year it applies to, and the page number. "
    "If a specific variable is not available in the document, set the answer to 'N/A', "
    "the year to null, confidence to 0.0, and page_number to -1. "
    "Be precise and provide only the requested values without additional explanation."
)


# Added Bulk Extraction
def answer_questions(state: FinancialGraphState) -> ExtractionResponse:
    """Use the LLM to extract all questions from the Excel template in a single call.
    
    - Gathers all variables into a single prompt
    - Sends to LLM once for bulk extraction using structured output
    - Captures confidence scores and page numbers for hyperlink generation
    """
    llm = state.get("llm")
    excel_data = state.get("excel_data", [])
    pdf_pages = state.get("pdf_pages", [])
    errors = state.get("errors", [])
    
    if llm is None:
        errors.append("LLM not available")
        return ExtractionResponse(errors=errors)
    if not pdf_pages:
        errors.append("No PDF content available")
        return ExtractionResponse(errors=errors)
    if not excel_data:
        errors.append("No questions from Excel")
        return ExtractionResponse(errors=errors)
        
    # Prepare Document Context
    full_context = "\n\n".join(
        [f"--- Page {i+1} ---\n{page.get('text', '')}" for i, page in enumerate(pdf_pages)]
    )
    
    # Prepare Variables List for the Prompt
    variables_to_extract = []
    for item in excel_data:
        var_req = f"- Variable: '{item['variable']}' (Worksheet: {item['worksheet']})"
        if item.get("description"):
            var_req += f", Description: {item['description']}"
        variables_to_extract.append(var_req)
        
    variables_list_str = "\n".join(variables_to_extract)
    
    # Construct the Single Prompt
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    user_content = (
        f"Please extract the following variables:\n{variables_list_str}\n\n"
        f"Document context:\n{full_context}"
    )
    
    # Invoke LLM with Structured Output
    structured_llm = llm.with_structured_output(ExtractionResponse)
    
    try:
        response: ExtractionResponse = structured_llm.invoke([system_msg, HumanMessage(content=user_content)])
        
        # Merge any existing errors with new ones if they occurred
        if errors:
            response.errors.extend(errors)
            
        return response
        
    except Exception as e:
        errors.append(f"Bulk LLM extraction error: {str(e)}")
        return ExtractionResponse(errors=errors)


# OLD IMPLEMENTATION (COMMENTED OUT FOR COMPARISON)
```
```python
# def answer_questions_old(state: FinancialGraphState) -> ExtractionResponse:
#     """Use the LLM to answer each question from the Excel template.
#
#     For each row:
#     - Constructs a prompt with the PDF context and the specific question
#     - Sends to LLM for extraction
#     - Collects the answer
#     """
#     llm = state.get("llm")
#     excel_data = state.get("excel_data", [])
#     pdf_pages = state.get("pdf_pages", [])
#     errors = state.get("errors", [])
#
#     
#     if llm is None:
#         errors.append("LLM not available")
#         return ExtractionResponse(errors=errors)
#     if not pdf_pages:
#         errors.append("No PDF content available")
#         return ExtractionResponse(errors=errors)
#     if not excel_data:
#         errors.append("No questions from Excel")
#         return ExtractionResponse(errors=errors)
#
#     
#     full_context = "\n\n".join(
#         [f"--- Page {i+1} ---\n{page.get('text', '')}" for i, page in enumerate(pdf_pages)]
#     )
#
#     system_msg = SystemMessage(content=SYSTEM_PROMPT)
#     answers = []
#
#     for item in excel_data:
#         variable = item["variable"]
#         description = item.get("description", "")
#         worksheet = item["worksheet"]
#
#         question = f"From the following document, extract the value for: {variable}"
#         if description:
#             question += f"\nDescription: {description}"
#         question += f"\n\nDocument context:\n{full_context}"
#
#         try:
#             response = llm.invoke([system_msg, HumanMessage(content=question)])
#             answer_text = response.content.strip()
#         except Exception as e:
#             answer_text = f"ERROR: {str(e)}"
#             errors.append(f"LLM error for '{variable}': {str(e)}")
#
#         
#         answers.append(
#             ExtractedAnswer(
#                 worksheet=worksheet,
#                 variable=variable,
#                 description=description,
#                 answer=answer_text,
#             )
#         )
#
```
## GIT COMMIT MESSAGE
```python
refactor: transition financial data extraction to single bulk LLM call
```

## GIT COMMIT DESCRIPTION
```python
- Replaced the sequential `for` loop of LLM calls with a single `with_structured_output` invocation to process all variables at once, significantly reducing token costs and execution time.
- Expanded the `ExtractedAnswer` Pydantic schema to include `confidence`, `page_number`, and `year` fields to support downstream validation and Excel hyperlink generation.
- Updated the `SYSTEM_PROMPT` to instruct the LLM on handling missing data with explicit fallback values (e.g., confidence 0.0, page_number -1).
- Commented out the legacy `answer_questions` function to preserve it for A/B testing and comparison per team lead request.
```
#     
#     return ExtractionResponse(answers=answers, errors=errors)
```

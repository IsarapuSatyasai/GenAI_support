````markdown
# Confidence Refinement Implementation

## Branch Name

feature/confidence-refinement

## Commit Message

feat: add template-aligned confidence refinement with two verification attempts

## Description

- Implement a confidence refinement workflow that reviews existing LLM extractions against PDF evidence.
- Preserves the CreditLens template structure.
- Prevents creation or removal of metrics, returns `N/A`
- For unavailable metrics, and limits refinement to a maximum of two attempts.

# 1. models.py

```python
"""Pydantic models for financial spreading."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractedAnswer(BaseModel):
    """Answer for one metric defined by the Excel template."""

    worksheet: str
    variable: str
    answer: str = Field(
        description="Extracted value. Return N/A when unavailable."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )
    page_number: int = Field(
        description="PDF page containing supporting evidence. -1 if unavailable."
    )
    source_fields: List[str] = Field(default_factory=list)
    formula: Optional[str] = None
    source_link: str = "N/A"


class ExtractionResponse(BaseModel):
    """Initial extraction response."""

    year: Optional[int] = None
    currency: Optional[str] = None
    scale: Optional[str] = None
    answers: List[ExtractedAnswer] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class RefinementAnswer(BaseModel):
    """Refined answer for an existing template metric."""

    worksheet: str
    variable: str
    answer: str = Field(
        description="Improved value or N/A if unavailable."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )
    page_number: int
    source_fields: List[str] = Field(default_factory=list)
    formula: Optional[str] = None

    is_supported: bool = Field(
        description="Whether the answer is directly supported by the PDF."
    )

    refinement_reason: str = Field(
        description="Brief explanation of what was corrected or verified."
    )


class RefinementResponse(BaseModel):
    """Response from one refinement attempt."""

    answers: List[RefinementAnswer] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
````

# 2. graph/graph_state.py

```python
"""Graph state for financial spreading."""

from typing import Any, Dict, List, Optional

from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict


class Metric(TypedDict):
    """Metric defined by the CreditLens template."""

    worksheet: str
    variable: str
    description: str
    formula: Optional[str]
    remarks: Optional[str]


class FinancialGraphState(TypedDict, total=False):
    """LangGraph state."""

    excel_path: str
    pdf_source: str
    output_path: str
    pdf_file_name: str

    excel_data: List[Metric]

    pdf_pages: List[Dict[str, Any]]
    source_language: str
    llm: Optional[AzureChatOpenAI]
    selected_pages: List[Dict[str, Any]]

    original_answers: List[Dict[str, Any]]
    refined_answers: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]

    refinement_attempt: int
    errors: List[str]
    extraction_completed: bool
```

# 3. nodes/answer_questions.py

Keep the existing extraction logic and change the final state update so the initial extraction is preserved separately.

```python
try:
    response: ExtractionResponse = structured_llm.invoke(messages)

    if errors:
        response.errors.extend(errors)

    final_answers = []

    for answer in response.answers:
        answer.source_link = build_hyperlink(
            answer.page_number,
            source_link,
        )
        final_answers.append(answer.model_dump())

    return {
        "original_answers": final_answers,
        "answers": final_answers,
        "errors": response.errors,
    }

except Exception as exc:
    error_msg = f"Bulk LLM extraction error: {exc}"
    errors.append(error_msg)

    return {
        "original_answers": [],
        "answers": [],
        "errors": errors,
    }
```

# 4. prompts/refinement_prompt.py

```python
"""Prompts used for financial metric refinement."""


REFINEMENT_SYSTEM_PROMPT = """
You are a senior financial-spreading verification analyst.

Your task is to REVIEW and IMPROVE an existing financial extraction.

You are NOT performing a new extraction of the document.

The Excel template is the authoritative source for the list of metrics.

NON-NEGOTIABLE RULES:

1. Only refine metrics that already exist in the provided template.

2. NEVER create a new metric.

3. NEVER rename a metric.

4. NEVER change the worksheet name.

5. NEVER add a metric that is not present in the template.

6. NEVER remove a metric that exists in the template.

7. The existing answer is the starting point.
   Your task is to check whether it is correct and improve it only
   when the PDF evidence supports an improvement.

8. Use the provided PDF pages as evidence.

9. Verify:
   - answer/value
   - metric mapping
   - worksheet
   - page number
   - source fields
   - formula
   - financial period

10. If the original answer is correct and sufficiently supported,
    keep it unchanged.

11. If the original answer is incorrect, replace it only with a
    value directly supported by the provided PDF evidence.

12. Do not guess or infer unsupported values.

13. If the metric cannot be verified or found in the provided PDF,
    return:
        answer = "N/A"
        confidence = 0.0
        page_number = -1
        source_fields = []
        formula = null
        is_supported = false

14. The string "N/A" must be used instead of null, blank, or omission
    when a metric is unavailable.

15. Return exactly one result for every metric supplied for refinement.

16. Preserve the exact worksheet and variable names supplied by the
    template.

17. A refinement is an improvement only when it is supported by
    evidence from the PDF.

18. Do not change an answer merely to make it look different.

19. If the original answer is already correct, retain it.

20. Maximum refinement attempts are controlled by the application.
    You only perform the current requested refinement attempt.

OUTPUT:

Return only the structured response matching RefinementResponse.
"""


def build_refinement_prompt(metrics, pdf_context, attempt):
    """Build the prompt for one refinement attempt."""

    metrics_text = []

    for metric in metrics:
        metrics_text.append(
            f"""
Worksheet: {metric["worksheet"]}
Variable: {metric["variable"]}
Description: {metric.get("description", "")}
Template Formula: {metric.get("formula")}
Template Remarks: {metric.get("remarks")}

Existing Answer:
{metric.get("answer", "N/A")}

Existing Confidence:
{metric.get("confidence", 0.0)}

Existing Page:
{metric.get("page_number", -1)}

Existing Source Fields:
{metric.get("source_fields", [])}
"""
        )

    return f"""
REFINEMENT ATTEMPT: {attempt}

Review the following existing metric answers against the supplied
PDF evidence.

IMPORTANT:
These metrics already come from the Excel template.

Do not create, remove, rename, or restructure metrics.

Your task is to improve the existing answer only when the PDF
provides stronger evidence.

EXISTING METRICS:

{"".join(metrics_text)}

PDF EVIDENCE:

{pdf_context}

For every supplied metric:

1. Check the existing answer.
2. Check the page number.
3. Check the source field mapping.
4. Check the financial period.
5. Check whether the value is supported.
6. Correct the answer only when necessary.
7. If unsupported or unavailable, return N/A.
"""
```

# 5. nodes/confidence_refinement.py

```python
"""Refine and verify existing financial extractions."""

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from graph.graph_state import FinancialGraphState
from models import RefinementResponse
from nodes.answer_questions import build_hyperlink
from prompts.refinement_prompt import (
    REFINEMENT_SYSTEM_PROMPT,
    build_refinement_prompt,
)


MAX_REFINEMENT_ATTEMPTS = 2


def _metric_key(metric: Dict[str, Any]) -> tuple:
    """Return the unique template key for a metric."""
    return (
        metric["worksheet"],
        metric["variable"],
    )


def _build_template_keys(
    excel_data: List[Dict[str, Any]],
) -> set:
    """Build allowed metric keys from the template."""
    return {
        _metric_key(metric)
        for metric in excel_data
    }


def _build_pdf_context(
    selected_pages: List[Dict[str, Any]],
) -> str:
    """Build PDF context for refinement."""
    return "\n\n".join(
        f'--- Page {page.get("page_number", index + 1)} ---\n'
        f'{page.get("text", "")}'
        for index, page in enumerate(selected_pages)
    )


def _select_metrics_for_refinement(
    excel_data: List[Dict[str, Any]],
    original_answers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select existing template metrics that need verification."""

    answer_map = {
        _metric_key(answer): answer
        for answer in original_answers
    }

    metrics = []

    for template_metric in excel_data:
        key = _metric_key(template_metric)
        answer = answer_map.get(key)

        if answer is None:
            answer = {
                "worksheet": template_metric["worksheet"],
                "variable": template_metric["variable"],
                "answer": "N/A",
                "confidence": 0.0,
                "page_number": -1,
                "source_fields": [],
                "formula": None,
            }

        metric = {
            **template_metric,
            **answer,
        }

        confidence = float(
            metric.get("confidence", 0.0)
        )

        if confidence > 0:
            metrics.append(metric)

    return metrics


def _normalize_answer(
    answer: Dict[str, Any],
    source_link: str,
) -> Dict[str, Any]:
    """Normalize refinement output."""

    if not answer.get("answer"):
        answer["answer"] = "N/A"

    if answer["answer"] == "N/A":
        answer["confidence"] = 0.0
        answer["page_number"] = -1
        answer["source_fields"] = []
        answer["formula"] = None
        answer["source_link"] = "N/A"

        return answer

    answer["source_link"] = build_hyperlink(
        answer.get("page_number"),
        source_link,
    )

    return answer


def _merge_refined_answers(
    excel_data: List[Dict[str, Any]],
    original_answers: List[Dict[str, Any]],
    refined_answers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge refined values while preserving the template."""

    original_map = {
        _metric_key(answer): answer
        for answer in original_answers
    }

    refined_map = {
        _metric_key(answer): answer
        for answer in refined_answers
    }

    final_answers = []

    for template_metric in excel_data:
        key = _metric_key(template_metric)

        original = original_map.get(key, {})
        refined = refined_map.get(key)

        if refined and refined.get("is_supported"):
            selected = {
                **original,
                **refined,
            }
        else:
            selected = original.copy()

        selected["worksheet"] = template_metric["worksheet"]
        selected["variable"] = template_metric["variable"]

        if not selected.get("answer"):
            selected["answer"] = "N/A"

        final_answers.append(selected)

    return final_answers


def confidence_refinement(
    state: FinancialGraphState,
) -> dict:
    """Run a maximum of two refinement attempts."""

    llm = state.get("llm")

    if llm is None:
        return {
            "answers": state.get("original_answers", []),
            "refined_answers": [],
            "errors": state.get("errors", []) + [
                "LLM not available for refinement."
            ],
        }

    excel_data = state.get("excel_data", [])
    original_answers = state.get("original_answers", [])
    selected_pages = state.get("selected_pages", [])

    settings = get_settings()

    source_link = (
        settings.sharepoint_link
        + state.get("pdf_file_name", "").replace(" ", "%20")
    )

    metrics = _select_metrics_for_refinement(
        excel_data,
        original_answers,
    )

    if not metrics:
        return {
            "answers": original_answers,
            "refined_answers": [],
            "refinement_attempt": 0,
        }

    pdf_context = _build_pdf_context(selected_pages)

    current_answers = metrics
    all_refined_answers = []

    for attempt in range(1, MAX_REFINEMENT_ATTEMPTS + 1):
        prompt = build_refinement_prompt(
            current_answers,
            pdf_context,
            attempt,
        )

        messages = [
            SystemMessage(
                content=REFINEMENT_SYSTEM_PROMPT
            ),
            HumanMessage(content=prompt),
        ]

        try:
            structured_llm = llm.with_structured_output(
                RefinementResponse
            )

            response = structured_llm.invoke(messages)

        except Exception as exc:
            error = (
                f"Refinement attempt {attempt} failed: {exc}"
            )

            return {
                "answers": state.get(
                    "original_answers",
                    [],
                ),
                "refined_answers": all_refined_answers,
                "refinement_attempt": attempt,
                "errors": state.get(
                    "errors",
                    [],
                ) + [error],
            }

        refined = [
            _normalize_answer(
                answer.model_dump(),
                source_link,
            )
            for answer in response.answers
        ]

        all_refined_answers.extend(refined)

        current_answers = [
            {
                **metric,
                **refined_answer,
            }
            for metric in metrics
            for refined_answer in refined
            if _metric_key(metric)
            == _metric_key(refined_answer)
        ]

        metrics = current_answers

    final_answers = _merge_refined_answers(
        excel_data,
        original_answers,
        all_refined_answers,
    )

    return {
        "answers": final_answers,
        "refined_answers": all_refined_answers,
        "refinement_attempt": MAX_REFINEMENT_ATTEMPTS,
        "errors": state.get("errors", []),
    }
```

# 6. graph/graph_workflow.py

Add the refinement node to the workflow.

```python
from nodes.confidence_refinement import confidence_refinement
```

Register the node:

```python
workflow.add_node(
    "confidence_refinement",
    confidence_refinement,
)
```

Change the edge:

```python
workflow.add_edge(
    "answer_questions",
    "confidence_refinement",
)

workflow.add_edge(
    "confidence_refinement",
    "create_excel_output",
)
```

The resulting section should be:

```python
workflow.add_node(
    "read_creditlens_template",
    read_creditlens_template,
)
workflow.add_node(
    "annual_report_parser",
    annual_report_parser,
)
workflow.add_node(
    "source_language",
    source_language,
)
workflow.add_node(
    "select_pages",
    select_pages,
)
workflow.add_node(
    "connect_llm",
    connect_llm,
)
workflow.add_node(
    "answer_questions",
    answer_questions,
)
workflow.add_node(
    "confidence_refinement",
    confidence_refinement,
)
workflow.add_node(
    "create_excel_output",
    create_excel_output,
)
workflow.add_node(
    "complete",
    complete,
)

workflow.add_edge(
    START,
    "read_creditlens_template",
)
workflow.add_edge(
    "read_creditlens_template",
    "annual_report_parser",
)
workflow.add_edge(
    "annual_report_parser",
    "source_language",
)
workflow.add_edge(
    "source_language",
    "select_pages",
)
workflow.add_edge(
    "select_pages",
    "connect_llm",
)
workflow.add_edge(
    "connect_llm",
    "answer_questions",
)
workflow.add_edge(
    "answer_questions",
    "confidence_refinement",
)
workflow.add_edge(
    "confidence_refinement",
    "create_excel_output",
)
workflow.add_edge(
    "create_excel_output",
    "complete",
)
workflow.add_edge(
    "complete",
    END,
)
```

# 7. nodes/create_excel_output.py

```python
"""Create template-aligned Excel output."""

import pandas as pd

from graph.graph_state import FinancialGraphState
from template.excel_writer import excel_write


def align_answers_to_template(
    excel_data,
    answers,
):
    """Ensure final output contains exactly template metrics."""

    answer_map = {
        (
            answer["worksheet"],
            answer["variable"],
        ): answer
        for answer in answers
    }

    aligned = []

    for metric in excel_data:
        key = (
            metric["worksheet"],
            metric["variable"],
        )

        answer = answer_map.get(key, {})

        aligned.append(
            {
                "worksheet": metric["worksheet"],
                "variable": metric["variable"],
                "answer": answer.get(
                    "answer",
                    "N/A",
                ),
                "confidence": answer.get(
                    "confidence",
                    0.0,
                ),
                "page_number": answer.get(
                    "page_number",
                    -1,
                ),
                "source_fields": answer.get(
                    "source_fields",
                    [],
                ),
                "formula": answer.get(
                    "formula",
                ),
                "source_link": answer.get(
                    "source_link",
                    "N/A",
                ),
            }
        )

    return aligned


def create_excel_output(
    state: FinancialGraphState,
):
    """Create Excel output using template-defined metrics."""

    excel_data = state.get(
        "excel_data",
        [],
    )

    answers = state.get(
        "answers",
        [],
    )

    output_path = state.get(
        "output_path",
    )

    pdf_file_name = state.get(
        "pdf_file_name",
    )

    output_excel = (
        "dbfs:"
        + output_path
        + pdf_file_name.removeprefix(
            "InputData"
        ).removesuffix(
            ".pdf"
        )
        + ".xlsx"
    )

    aligned_answers = align_answers_to_template(
        excel_data,
        answers,
    )

    worksheet_dict = {}

    for answer in aligned_answers:
        worksheet = answer["worksheet"]

        row = {
            key: value
            for key, value in answer.items()
            if key != "worksheet"
        }

        worksheet_dict.setdefault(
            worksheet,
            [],
        ).append(row)

    worksheet_dict = {
        worksheet: pd.DataFrame(rows)
        for worksheet, rows in worksheet_dict.items()
    }

    try:
        status = excel_write(
            dataframes=worksheet_dict,
            output_path=output_excel,
        )

        return {
            "output_excel": status,
            "errors": state.get(
                "errors",
                [],
            ),
        }

    except Exception as exc:
        error = f"Error writing excel file: {exc}"

        return {
            "output_excel": "",
            "errors": state.get(
                "errors",
                [],
            ) + [error],
        }
```

# 8. Update Initial Extraction Prompt

In `prompts/system_prompt.py`, replace the missing-value rules with:

```text
3. No guessing - Do NOT invent numbers. If a metric cannot be found or reliably derived, return "N/A".

8. Do not skip. Always provide one output for every variable supplied by the Excel template.

   If the metric is not found in the document:

       answer = "N/A"
       confidence = 0.0
       page_number = -1
       source_fields = []
       formula = null
```

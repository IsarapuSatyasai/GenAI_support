### 1. Config

```python
VALIDATION_SCORE_THRESHOLD = 0.6
MAX_REFINEMENT_ATTEMPTS = 2
```

### 2. State

Only add:

```python
refinement_attempts: int
```

Keep your existing `answers` exactly as it is.

### 3. One node

Create `nodes/confidence_refinement.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage

from graph.graph_state import FinancialGraphState
from models import ExtractionResponse
from config import (
    VALIDATION_SCORE_THRESHOLD,
    MAX_REFINEMENT_ATTEMPTS,
)


REFINEMENT_PROMPT = """
Review the existing financial extraction against the provided
annual report context.

The existing answer has low confidence.

Try to improve or correct the existing answer using the document.
Do not extract unrelated metrics.
Do not invent information.

Return the corrected answer, confidence, page number,
source fields, and formula.
"""


def confidence_refinement(
    state: FinancialGraphState,
) -> str:

    answers = state.get("answers", [])
    attempts = state.get("refinement_attempts", 0)

    # Check confidence
    low_confidence = [
        answer
        for answer in answers
        if float(answer.get("confidence", 0.0))
        < VALIDATION_SCORE_THRESHOLD
    ]

    # Everything is good → continue
    if not low_confidence:
        return "continue"

    # Maximum attempts reached → stop loop
    if attempts >= MAX_REFINEMENT_ATTEMPTS:
        return "continue"

    llm = state.get("llm")
    selected_pages = state.get("selected_pages", [])

    structured_llm = llm.with_structured_output(
        ExtractionResponse
    )

    full_context = "\n\n".join(
        [
            f"--- Page {i + 1} ---\n{page.get('text', '')}"
            for i, page in enumerate(selected_pages)
        ]
    )

    refined_answers = []

    for answer in low_confidence:

        messages = [
            SystemMessage(content=REFINEMENT_PROMPT),
            HumanMessage(
                content=f"""
Existing extraction:

{answer}

Annual report context:

{full_context}
"""
            ),
        ]

        response = structured_llm.invoke(messages)

        refined_answers.extend(response.answers)

    # Replace only the low-confidence answers
    refined_by_variable = {
        answer.variable: answer.model_dump()
        for answer in refined_answers
    }

    updated_answers = []

    for answer in answers:

        variable = answer.get("variable")

        if variable in refined_by_variable:
            updated_answers.append(
                refined_by_variable[variable]
            )
        else:
            updated_answers.append(answer)

    # Update the existing state
    state["answers"] = updated_answers
    state["refinement_attempts"] = attempts + 1

    # Loop back to this same node
    return "refine"
```

### 4. Graph change

Add only:

```python
from nodes.confidence_refinement import confidence_refinement
```

Register the node:

```python
workflow.add_node(
    "confidence_refinement",
    confidence_refinement
)
```

Then replace:

```python
workflow.add_edge("answer_questions", "complete")
```

with:

```python
workflow.add_edge(
    "answer_questions",
    "confidence_refinement"
)

workflow.add_conditional_edges(
    "confidence_refinement",
    confidence_refinement,
    {
        "refine": "confidence_refinement",
        "continue": "complete",
    },
)
```

For the confidence-based refinement work, I’d suggest:

**Branch name**

```text
feature/financial-spreading-confidence-refinement
```

**Commit message**

```text
feat: add confidence-based extraction refinement loop
```

**Commit description**

```text
Add a configurable confidence threshold and maximum refinement
attempts for financial metric extraction.

- Check extracted metric confidence against the configured threshold.
- Refine only low-confidence extraction results.
- Loop the refined results back through the confidence check.
- Stop refinement after the configured maximum attempts.
- Keep the existing financial spreading extraction flow unchanged.
```


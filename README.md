```python
# --- Page selection ---
first_n_pages = 5
page_score_threshold = 1
min_pages = 30

# --- Refinement ---
refinement_attempts = 0
```

```python
    def confidence_refinement(
    state: FinancialGraphState,
) -> dict:
    """Run the configured number of refinement attempts."""

    settings = get_settings()

    # Skip refinement when attempts are set to zero.
    if settings.refinement_attempts <= 0:
        original_answers = state.get("original_answers", [])

        return {
            "answers": original_answers,
            "refined_answers": [],
            "refinement_attempt": 0,
            "errors": state.get("errors", []),
        }

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

    for attempt in range(1, settings.refinement_attempts + 1):
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
        "refinement_attempt": settings.refinement_attempts,
        "errors": state.get("errors", []),
    }
```

```python
for attempt in range(
    1,
    settings.refinement_attempts + 1,
):
```

```python
from prompts.selected_keywords import (
    page_relevance_score,
    page_score,
)
```

```python
def select_pages(state: FinancialGraphState) -> dict:
    """Select relevant pages using weighted page scoring."""

    settings = get_settings()

    pdf_pages = state.get("pdf_pages", [])
    n = settings.first_n_pages

    selected_pages = []

    # Keep the first N pages as initial candidates.
    for page in pdf_pages[:n]:
        selected_pages.append(page)

    # Use existing keyword matching to create candidates.
    for page in pdf_pages[n:]:
        page_text = page.get("text", "")

        if page_score(page_text) > settings.page_score_threshold:
            selected_pages.append(page)

    # Score candidate pages using additional signals.
    scored_pages = []

    for page in selected_pages:
        signals = page_relevance_score(
            page.get("text", "")
        )

        scored_page = dict(page)
        scored_page["page_relevance_score"] = signals["score"]

        scored_pages.append(scored_page)

    # Rank pages by relevance.
    scored_pages.sort(
        key=lambda page: page["page_relevance_score"],
        reverse=True,
    )

    # Keep the existing behavior for small PDFs.
    if len(pdf_pages) < settings.min_pages:
        scored_pages = pdf_pages

    return {
        "selected_pages": scored_pages,
        "errors": state.get("errors", []),
    }
```

```python
import re


FINANCIAL_TERMS = [
    "balance sheet",
    "statement of financial position",
    "income statement",
    "profit and loss",
    "cash flow statement",
    "financial statements",
    "revenue",
    "assets",
    "liabilities",
    "equity",
]


UNIT_TERMS = [
    "in thousands",
    "in millions",
    "in billions",
    "thousand",
    "million",
    "billion",
    "€000",
    "$000",
]


def page_relevance_score(text: str) -> dict:
    """Calculate relevance score for a financial page."""

    text_lower = text.lower()

    keyword_score = page_score(text)

    year_score = len(
        set(
            re.findall(
                r"\b(?:19|20)\d{2}\b",
                text,
            )
        )
    )

    number_count = len(
        re.findall(
            r"(?<!\w)[(+-]?\d[\d,.]*(?!\w)",
            text,
        )
    )

    word_count = max(
        len(text.split()),
        1,
    )

    numeric_density = number_count / word_count

    unit_score = sum(
        term in text_lower
        for term in UNIT_TERMS
    )

    financial_score = sum(
        term in text_lower
        for term in FINANCIAL_TERMS
    )

    score = (
        keyword_score
        + (year_score * 2)
        + (numeric_density * 5)
        + (unit_score * 2)
        + (financial_score * 3)
    )

    return {
        "score": round(score, 3),
        "keyword_score": keyword_score,
        "year_score": year_score,
        "numeric_density": round(
            numeric_density,
            4,
        ),
        "unit_score": unit_score,
        "financial_score": financial_score,
    }
```

```python
def build_page_selection_comparison(
    excel_data,
    baseline_pages,
    improved_pages,
):
    """Compare baseline and improved selected PDF pages."""

    baseline_numbers = {
        page.get("page_number")
        for page in baseline_pages
    }

    improved_numbers = {
        page.get("page_number")
        for page in improved_pages
    }

    rows = []

    for page_number in sorted(
        baseline_numbers | improved_numbers
    ):
        rows.append(
            {
                "page_number": page_number,
                "baseline_selected": (
                    "Yes"
                    if page_number in baseline_numbers
                    else "No"
                ),
                "improved_selected": (
                    "Yes"
                    if page_number in improved_numbers
                    else "No"
                ),
                "changed": (
                    page_number in baseline_numbers
                )
                != (
                    page_number in improved_numbers
                ),
            }
        )

    return {
        "Page Selection": pd.DataFrame(rows),
    }
```

```text
Add page selection comparison with refinement disabled

```

```text
Implemented page-selection evaluation with refinement disabled.

- Set refinement attempts to 0.
- Added weighted page relevance scoring using financial terms, years, numeric density, and units.
- Updated page selection to rank candidate pages by relevance.
- Added baseline vs improved page-selection comparison output.
- Kept the existing Excel template as the source of truth.
- Preserved the existing extraction and refinement workflow.
- Generated comparison results alongside the normal Excel output.
```

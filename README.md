```python
    # Keyword-based relevant contexts extracted from selected PDF pages
    keyword_contexts: List[Dict[str, Any]]
```

```python
"""
Node: Search for relevant keywords within selected PDF pages.

The node searches the selected PDF page text for keywords derived
from the Excel metric and extracts surrounding text as context.
"""

import re
from typing import Dict, Any, List

from graph.graph_state import FinancialGraphState


# Words that usually don't provide useful search signals
STOP_WORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "this",
    "that",
    "during",
    "year",
    "years",
    "company",
    "companies",
    "total",
    "value",
    "amount",
    "financial",
    "information",
}


def clean_keywords(words: List[str]) -> List[str]:
    """
    Remove unnecessary words and duplicate keywords.
    """

    cleaned = []

    for word in words:
        word = word.strip().lower()

        if not word:
            continue

        if word in STOP_WORDS:
            continue

        if len(word) < 3:
            continue

        if word not in cleaned:
            cleaned.append(word)

    return cleaned


def build_keywords(metric: Dict[str, Any]) -> List[str]:
    """
    Build search keywords from the Excel metric.

    Priority:
        1. Variable
        2. Description
        3. Remarks
    """

    keywords = []

    variable = metric.get("variable", "")
    description = metric.get("description", "")
    remarks = metric.get("remarks", "")

    # Keep the complete variable phrase
    if variable:
        keywords.append(variable.strip())

    # Add individual meaningful words from variable
    if variable:
        keywords.extend(variable.split())

    # Add meaningful words from description
    if description:
        keywords.extend(description.split())

    # Add meaningful words from remarks
    if remarks:
        keywords.extend(remarks.split())

    return clean_keywords(keywords)


def extract_context(
    text: str,
    start: int,
    end: int,
    before: int = 500,
    after: int = 1000,
) -> str:
    """
    Extract surrounding text around a keyword match.
    """

    context_start = max(0, start - before)
    context_end = min(len(text), end + after)

    context = text[context_start:context_end]

    return context.strip()


def find_keyword_matches(
    text: str,
    keyword: str,
    max_matches: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find keyword occurrences in text.

    Matching is case-insensitive.
    """

    matches = []

    if not text or not keyword:
        return matches

    # Escape keyword so special characters don't break regex
    escaped_keyword = re.escape(keyword)

    # Word-boundary matching
    pattern = rf"\b{escaped_keyword}\b"

    for match in re.finditer(
        pattern,
        text,
        flags=re.IGNORECASE,
    ):
        context = extract_context(
            text,
            match.start(),
            match.end(),
        )

        matches.append(
            {
                "keyword": keyword,
                "start": match.start(),
                "end": match.end(),
                "context": context,
            }
        )

        if len(matches) >= max_matches:
            break

    return matches


def keyword_search(state: FinancialGraphState) -> dict:
    """
    Search for Excel metric keywords inside selected PDF pages.

    Returns:
        keyword_contexts
    """

    excel_data = state.get("excel_data", [])
    selected_pages = state.get("selected_pages", [])

    keyword_contexts = []

    for metric in excel_data:

        variable = metric.get("variable", "")
        keywords = build_keywords(metric)

        metric_matches = []

        for index, page in enumerate(selected_pages):

            page_text = page.get("text", "")

            # Prefer the original PDF page number if available
            page_number = page.get(
                "page_number",
                page.get("page", index + 1),
            )

            for keyword in keywords:

                matches = find_keyword_matches(
                    page_text,
                    keyword,
                    max_matches=5,
                )

                for match in matches:

                    metric_matches.append(
                        {
                            "page_number": page_number,
                            "keyword": match["keyword"],
                            "context": match["context"],
                        }
                    )

        keyword_contexts.append(
            {
                "variable": variable,
                "keywords": keywords,
                "matches": metric_matches,
            }
        )

    return {"keyword_contexts": keyword_contexts}
```

```python
from nodes.keyword_search import keyword_search
```

```python
workflow.add_node("keyword_search", keyword_search)
```

```python
workflow.add_edge("select_pages", "keyword_search")
workflow.add_edge("keyword_search", "connect_llm")
```

```python
def build_keyword_context(
    keyword_contexts: list[dict],
    selected_pages: list[dict],
) -> str:
    """
    Build LLM context from keyword matches.

    If no keyword is found for a metric, fall back
    to the selected PDF pages.
    """

    sections = []

    for metric in keyword_contexts:

        variable = metric.get("variable", "")
        matches = metric.get("matches", [])

        sections.append(
            f"===== Variable: {variable} ====="
        )

        if matches:

            seen_contexts = set()

            for match in matches:

                context = match.get(
                    "context",
                    ""
                ).strip()

                page_number = match.get("page_number","")

                keyword = match.get("keyword","")

                if not context:
                    continue

                if context in seen_contexts:
                    continue

                seen_contexts.add(context)

                sections.append(
                    f"\n--- Page {page_number} | "
                    f"Keyword: {keyword} ---\n"
                    f"{context}"
                )

        else:

            sections.append(
                "No keyword match found. "
                "Fallback to selected PDF pages."
            )

            for index, page in enumerate(selected_pages):

                page_number = page.get(
                    "page_number",
                    page.get("page",index + 1)
                )

                page_text = page.get("text","")

                sections.append(
                    f"\n--- Page {page_number} ---\n"
                    f"{page_text}"
                )

    return "\n".join(sections)
```

```python
def answer_questions(state: FinancialGraphState) -> dict:
    """
    Extract metrics via LLM using keyword-focused PDF context.
    """

    llm = state.get("llm")

    if llm is None:
        return {
            "answers": [],
            "errors": ["LLM is missing from graph state"]
        }

    structured_llm = llm.with_structured_output(ExtractionResponse)

    excel_data = state.get("excel_data", [])
    errors = state.get("errors", [])
    keyword_contexts = state.get("keyword_contexts", [])
    pdf_file_name = state.get("pdf_file_name", "")

    settings = get_settings()

    sharepoint_link = settings.sharepoint_link

    source_link = (
        sharepoint_link
        + pdf_file_name.replace(" ", "%20")
    )

    USER_PROMPT = get_user_prompt(excel_data)
    selected_pages = state.get(
    "selected_pages",
     []
    ) 

    keyword_contexts = state.get(
       "keyword_contexts",
        []
    )

    # Use keyword-focused context
    full_context = build_keyword_context(keyword_contexts)

    user_content = (
        f"Please extract the following variables:\n"
        f"{USER_PROMPT}\n\n"
        f"Relevant document context:\n"
        f"{full_context}\n"
    )

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=user_content
        )
    ]

    try:

        response: ExtractionResponse = (
            structured_llm.invoke(messages)
        )

        if errors:
            response.errors.extend(errors)

        final_answers = []

        for answer in response.answers:

            answer.source_link = build_hyperlink(
                answer.page_number,
                source_link
            )

            final_answers.append(
                answer.model_dump()
            )

        return {
            "answers": final_answers,
            "errors": response.errors
        }

    except Exception as e:

        error_msg = (
            f"Bulk LLM extraction error: {str(e)}"
        )

        errors.append(error_msg)

        return {
            "answers": [],
            "errors": errors
        }
```

```python
feature/financial-spreading-keyword-search
```

```python
feat: add keyword search for selected PDF pages
```

```python
Added keyword-based context extraction for financial metric extraction.

- Added keyword_search node to search relevant keywords within selected PDF pages
- Extracted surrounding text around keyword matches for focused LLM context
- Added keyword_contexts to FinancialGraphState
- Updated LangGraph workflow to include keyword search before LLM extraction
- Updated answer_questions to use keyword-focused context
- Added fallback to selected page content when no keyword match is found
- Preserved PDF page numbers for source references and Excel hyperlinks
- Reduced irrelevant PDF content passed to the LLM
```

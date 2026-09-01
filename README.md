### nodes/keyword_search.py

```python
"""
Node: Search for relevant keywords within selected PDF pages.

The node:
1. Builds meaningful search phrases from Excel metrics.
2. Searches selected PDF pages.
3. Prioritizes exact phrase matches.
4. Falls back to individual meaningful keywords.
5. Extracts surrounding context.
6. Scores and ranks matches.
7. Deduplicates contexts.
8. Limits the number of contexts per metric.
9. Limits the total context size per metric.
"""

import re
from typing import Dict, Any, List

from graph.graph_state import FinancialGraphState

# Maximum number of final contexts retained for ONE metric.
MAX_CONTEXTS_PER_METRIC = 5

# Maximum number of characters retained for ONE metric.
MAX_CHARS_PER_METRIC = 8_000

# Context surrounding a keyword match.
CONTEXT_BEFORE = 500
CONTEXT_AFTER = 1_000

# Maximum raw matches collected for one search phrase.
MAX_MATCHES_PER_KEYWORD = 5

# Stop words
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
    "period",
    "report",
    "reported",
    "following",
    "related",
    "based",
    "including",
    "other",
    "such",
    "into",
    "than",
    "their",
    "there",
    "which",
    "where",
    "were",
    "been",
    "being",
}

# Text normalization
def normalize_text(text: str) -> str:
    """
    Normalize text for keyword searching.

    Converts repeated whitespace into a single space and
    lowercases the text.
    """

    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

# Keyword cleaning

def clean_keywords(words: List[str]) -> List[str]:
    """
    Remove unnecessary words and duplicate keywords.
    """

    cleaned = []
    seen = set()

    for word in words:

        word = word.strip().lower()

        # Remove punctuation around the keyword.
        word = re.sub(r"[^\w\s&/-]", "", word)

        if not word:
            continue

        if len(word) < 3:
            continue

        if word in STOP_WORDS:
            continue

        if word in seen:
            continue

        seen.add(word)
        cleaned.append(word)

    return cleaned



# Keyword generation

def build_keywords(metric: Dict[str, Any]) -> List[str]:
    """
    Build search phrases and keywords from an Excel metric.

    Priority:

    1. Complete variable phrase
    2. Complete description phrase
    3. Meaningful variable words
    4. Meaningful description words
    5. Meaningful remarks words

    Complete phrases are intentionally retained because
    phrase matching is more reliable than individual-word matching.
    """

    variable = str(metric.get("variable") or "").strip()
    description = str(metric.get("description") or "").strip()
    remarks = str(metric.get("remarks") or "").strip()

    phrases = []

    # Highest priority: complete variable.
    if variable:
        phrases.append(variable)

    # Description can contain a useful financial phrase.
    if description:
        phrases.append(description)

    # Meaningful words from variable.
    if variable:
        phrases.extend(variable.split())

    # Meaningful words from description.
    if description:
        phrases.extend(description.split())

    # Remarks are lowest priority.
    if remarks:
        phrases.extend(remarks.split())

    return clean_keywords(phrases)

# Context extraction

def extract_context(
    text: str,
    start: int,
    end: int,
    before: int = CONTEXT_BEFORE,
    after: int = CONTEXT_AFTER,
) -> str:
    """
    Extract surrounding text around a keyword match.
    """

    context_start = max(0, start - before)
    context_end = min(len(text), end + after)

    context = text[context_start:context_end]

    return context.strip()



# Keyword matching

def find_keyword_matches(
    text: str,
    keyword: str,
    max_matches: int = MAX_MATCHES_PER_KEYWORD,
) -> List[Dict[str, Any]]:
    """
    Find keyword occurrences in text.

    Exact phrases are matched using word boundaries.
    """

    matches = []

    if not text or not keyword:
        return matches

    escaped_keyword = re.escape(keyword.strip())

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

# Match Scoring

def score_match(
    keyword: str,
    variable: str,
    description: str,
) -> int:
    """
    Assign a relevance score to a keyword.

    Higher score = more relevant.

    Exact variable phrase:
        100

    Exact description phrase:
        80

    Multi-word phrase:
        60

    Single meaningful keyword:
        20
    """

    keyword_normalized = normalize_text(keyword)
    variable_normalized = normalize_text(variable)
    description_normalized = normalize_text(description)

    # Exact variable phrase.
    if keyword_normalized == variable_normalized:
        return 100

    # Exact description phrase.
    if (
        description_normalized
        and keyword_normalized == description_normalized
    ):
        return 80

    # Multi-word phrase.
    if " " in keyword_normalized:
        return 60

    # Single word.
    return 20


# Main node

def keyword_search(state: FinancialGraphState) -> dict:
    """
    Search selected PDF pages for relevant financial metric contexts.

    Output:
        keyword_contexts
        keyword_search_stats
    """

    excel_data = state.get("excel_data", [])
    selected_pages = state.get("selected_pages", [])

    keyword_contexts = []

    total_metrics = len(excel_data)
    total_matches = 0
    total_context_chars = 0

    print("\n" + "=" * 80)
    print("KEYWORD SEARCH")
    print("=" * 80)

    print(f"Metrics: {total_metrics}")
    print(f"Selected pages: {len(selected_pages)}")

    for metric in excel_data:

        variable = str(
            metric.get("variable") or ""
        ).strip()

        description = str(
            metric.get("description") or ""
        ).strip()

        keywords = build_keywords(metric)

        candidate_matches = []

        # Search every selected page

        for index, page in enumerate(selected_pages):

            page_text = page.get("text", "")

            if not page_text:
                continue

            page_number = page.get(
                "page_number",
                page.get("page", index + 1),
            )

            normalized_page_text = normalize_text(page_text)

            for keyword in keywords:

                matches = find_keyword_matches(
                    normalized_page_text,
                    keyword,
                    max_matches=MAX_MATCHES_PER_KEYWORD,
                )

                relevance_score = score_match(
                    keyword,
                    variable,
                    description,
                )

                for match in matches:

                    candidate_matches.append(
                        {
                            "page_number": page_number,
                            "keyword": keyword,
                            "context": match["context"],
                            "score": relevance_score,
                        }
                    )

        # Deduplicate contexts

        unique_matches = {}

        for match in candidate_matches:

            context = match.get("context", "").strip()

            if not context:
                continue

            # Normalize context for deduplication.
            context_key = normalize_text(context)

            existing = unique_matches.get(context_key)

            if (
                existing is None
                or match["score"] > existing["score"]
            ):
                unique_matches[context_key] = match

        # Rank

        ranked_matches = sorted(
            unique_matches.values(),
            key=lambda item: (
                -item["score"],
                item["page_number"],
            ),
        )

        # Apply per-metric limits

        final_matches = []
        current_chars = 0

        for match in ranked_matches:

            context = match["context"]

            context_length = len(context)

            if (
                current_chars + context_length
                > MAX_CHARS_PER_METRIC
            ):
                continue

            final_matches.append(
                {
                    "page_number": match["page_number"],
                    "keyword": match["keyword"],
                    "context": context,
                    "score": match["score"],
                }
            )

            current_chars += context_length

            if len(final_matches) >= MAX_CONTEXTS_PER_METRIC:
                break

        total_matches += len(final_matches)
        total_context_chars += current_chars

        keyword_contexts.append(
            {
                "variable": variable,
                "keywords": keywords,
                "matches": final_matches,
                "match_count": len(final_matches),
                "context_chars": current_chars,
            }
        )

        print(
            f"[Keyword Search] "
            f"{variable}: "
            f"{len(final_matches)} matches / "
            f"{current_chars} chars"
        )

    # Statistics

    keyword_search_stats = {
        "total_metrics": total_metrics,
        "total_matches": total_matches,
        "total_context_chars": total_context_chars,
        "average_matches_per_metric": (
            total_matches / total_metrics
            if total_metrics
            else 0
        ),
        "average_context_chars_per_metric": (
            total_context_chars / total_metrics
            if total_metrics
            else 0
        ),
    }

    print("\n" + "-" * 80)
    print("KEYWORD SEARCH SUMMARY")
    print("-" * 80)

    print(
        f"Total matches: "
        f"{total_matches}"
    )

    print(
        f"Total context chars: "
        f"{total_context_chars}"
    )

    print(
        f"Average matches/metric: "
        f"{keyword_search_stats['average_matches_per_metric']:.2f}"
    )

    print(
        f"Average chars/metric: "
        f"{keyword_search_stats['average_context_chars_per_metric']:.2f}"
    )

    print("=" * 80 + "\n")

    return {
        "keyword_contexts": keyword_contexts,
        "keyword_search_stats": keyword_search_stats,
    }
```
### graph/graph_state.py
```python
# Keyword retrieval statistics
keyword_search_stats: Dict[str, Any]
```

### nodes/answer_question.py
```python
"""
Node: Answer Questions using LLM.

Uses keyword-focused PDF contexts instead of sending all
selected PDF pages directly to the LLM.
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from graph.graph_state import FinancialGraphState
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.user_prompt import get_user_prompt
from models import ExtractionResponse

from config import get_settings

# Maximum total characters sent to the LLM as document context.
MAX_LLM_CONTEXT_CHARS = 80_000

# Excel hyperlink

def build_hyperlink(
    page_number: Optional[int],
    source_link: str,
) -> str:
    """
    Build the Excel HYPERLINK formula.
    """

    if (
        page_number is not None
        and page_number != -1
        and source_link
    ):
        return (
            f'=HYPERLINK('
            f'"{source_link}#page={page_number}", '
            f'"Page {page_number}")'
        )

    return "N/A"

# Keyword context builder

def build_keyword_context(
    keyword_contexts: list[dict],
    selected_pages: list[dict],
    max_chars: int = MAX_LLM_CONTEXT_CHARS,
) -> str:
    """
    Build a compact LLM context from keyword matches.

    Strategy:

    1. Use ranked keyword matches.
    2. Deduplicate contexts globally.
    3. Respect the global character limit.
    4. If a metric has no keyword match, do NOT dump
       all selected pages.
    5. Instead, add a small fallback context from the
       selected pages.
    """

    sections = []

    # Global deduplication across ALL metrics.
    seen_contexts = set()

    current_chars = 0

    # Keyword-focused retrieval

    for metric in keyword_contexts:

        variable = metric.get(
            "variable",
            "",
        )

        matches = metric.get(
            "matches",
            [],
        )

        if not matches:
            continue

        variable_header = (
            f"\n===== Variable: {variable} =====\n"
        )

        if (
            current_chars
            + len(variable_header)
            > max_chars
        ):
            break

        sections.append(variable_header)
        current_chars += len(variable_header)

        for match in matches:

            context = str(
                match.get(
                    "context",
                    "",
                )
            ).strip()

            if not context:
                continue

            # Global deduplication.
            context_key = (
                " ".join(
                    context.lower().split()
                )
            )

            if context_key in seen_contexts:
                continue

            seen_contexts.add(context_key)

            page_number = match.get(
                "page_number",
                "",
            )

            keyword = match.get(
                "keyword",
                "",
            )

            score = match.get(
                "score",
                0,
            )

            section = (
                f"\n--- "
                f"Page {page_number} | "
                f"Keyword: {keyword} | "
                f"Score: {score}"
                f" ---\n"
                f"{context}\n"
            )

            section_length = len(section)

            if (
                current_chars
                + section_length
                > max_chars
            ):
                break

            sections.append(section)
            current_chars += section_length

    # Find variables that received no match.
    matched_variables = {
        metric.get("variable", "")
        for metric in keyword_contexts
        if metric.get("matches")
    }

    unmatched_variables = [
        metric.get("variable", "")
        for metric in keyword_contexts
        if (
            metric.get("variable", "")
            not in matched_variables
        )
    ]

    if unmatched_variables and selected_pages:

        fallback_header = (
            "\n===== LIMITED FALLBACK CONTEXT =====\n"
            "The following variables did not have "
            "keyword matches:\n"
            + "\n".join(
                f"- {variable}"
                for variable in unmatched_variables
            )
            + "\n"
        )

        if (
            current_chars
            + len(fallback_header)
            <= max_chars
        ):
            sections.append(fallback_header)
            current_chars += len(fallback_header)

            # Instead of sending every selected page,
            # use only a limited amount of text.
            fallback_chars_remaining = min(
                10_000,
                max_chars - current_chars,
            )

            for index, page in enumerate(
                selected_pages
            ):

                if fallback_chars_remaining <= 0:
                    break

                page_number = page.get(
                    "page_number",
                    page.get(
                        "page",
                        index + 1,
                    ),
                )

                page_text = str(
                    page.get(
                        "text",
                        "",
                    )
                ).strip()

                if not page_text:
                    continue

                # Limit individual fallback page.
                page_text = page_text[
                    : min(
                        3_000,
                        fallback_chars_remaining,
                    )
                ]

                section = (
                    f"\n--- "
                    f"Fallback Page {page_number}"
                    f" ---\n"
                    f"{page_text}\n"
                )

                sections.append(section)

                section_length = len(section)

                current_chars += section_length
                fallback_chars_remaining -= section_length

    if not sections:

        return (
            "No relevant keyword context was found "
            "in the selected PDF pages."
        )

    return "\n".join(sections)


# Main node

def answer_questions(
    state: FinancialGraphState,
) -> dict:
    """
    Extract financial metrics using the LLM.

    The LLM receives keyword-focused contexts rather
    than the complete selected PDF pages.
    """

    llm = state.get("llm")

    if llm is None:

        return {
            "answers": [],
            "errors": [
                "LLM is missing from graph state"
            ],
        }

    excel_data = state.get(
        "excel_data",
        [],
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    selected_pages = state.get(
        "selected_pages",
        [],
    )

    keyword_contexts = state.get(
        "keyword_contexts",
        [],
    )

    pdf_file_name = state.get(
        "pdf_file_name",
        "",
    )

    # Settings

    settings = get_settings()

    sharepoint_link = settings.sharepoint_link

    source_link = (
        sharepoint_link
        + pdf_file_name.replace(
            " ",
            "%20",
        )
    )

    # Build prompt

    user_prompt = get_user_prompt(
        excel_data
    )

    full_context = build_keyword_context(
        keyword_contexts=keyword_contexts,
        selected_pages=selected_pages,
        max_chars=MAX_LLM_CONTEXT_CHARS,
    )

    user_content = (
        "Please extract the following variables:\n"
        f"{user_prompt}\n\n"
        "Relevant document context:\n"
        f"{full_context}\n"
    )

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=user_content
        ),
    ]

    # Debug information

    print("\n" + "=" * 80)
    print("LLM EXTRACTION")
    print("=" * 80)

    print(
        f"Excel metrics: "
        f"{len(excel_data)}"
    )

    print(
        f"Keyword contexts: "
        f"{len(keyword_contexts)}"
    )

    print(
        f"LLM context characters: "
        f"{len(full_context):,}"
    )

    print("=" * 80 + "\n")

    # Structured LLM

    structured_llm = (
        llm.with_structured_output(
            ExtractionResponse
        )
    )

    try:

        response: ExtractionResponse = (
            structured_llm.invoke(
                messages
            )
        )

        # Merge errors

        if errors:
            response.errors.extend(
                errors
            )

        # Add source hyperlinks

        final_answers = []

        for answer in response.answers:

            answer.source_link = (
                build_hyperlink(
                    answer.page_number,
                    source_link,
                )
            )

            final_answers.append(
                answer.model_dump()
            )

        print(
            f"LLM answers returned: "
            f"{len(final_answers)}"
        )

        return {
            "answers": final_answers,
            "errors": response.errors,
        }

    except Exception as e:

        error_msg = (
            "Bulk LLM extraction error: "
            f"{str(e)}"
        )

        errors.append(
            error_msg
        )

        print(
            f"\nERROR: {error_msg}\n"
        )

        return {
            "answers": [],
            "errors": errors,
        }
```

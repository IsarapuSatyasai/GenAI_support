### nodes/select_pages.py
```python

"""Select relevant annual report pages using template metrics."""

import re

from config import get_settings
from graph.graph_state import FinancialGraphState
from prompts.selected_keywords import (
    normalize_text,
    numeric_density,
    page_score,
    table_structure_score,
)


CONTINUATION_MARKERS = (
    "continued",
    "continuation",
    "carried forward",
    "brought forward",
)

METRIC_SYNONYM_GROUPS = (
    ("revenue", "turnover"),
    ("total revenue", "total sales", "total turnover"),
    ("cost of sales", "cost of goods sold", "cogs"),
    (
        "profit before tax",
        "income before tax",
        "earnings before tax",
        "pbt",
    ),
    ("profit after tax", "net profit after tax", "pat"),
    ("profit for the year", "profit for the period"),
    ("cash and cash equivalents", "cash equivalents"),
    ("trade receivables", "accounts receivable", "trade debtors"),
    ("trade payables", "accounts payable", "trade creditors"),
    (
        "property plant and equipment",
        "property and equipment",
        "tangible fixed assets",
        "ppe",
    ),
    ("current assets", "short term assets"),
    ("non current assets", "noncurrent assets", "long term assets"),
    ("current liabilities", "short term liabilities"),
    (
        "non current liabilities",
        "noncurrent liabilities",
        "long term liabilities",
    ),
    ("operating profit", "operating income"),
    ("borrowings", "financial borrowings"),
    ("lease liabilities", "lease obligations"),
    ("shareholders equity", "shareholders funds"),
    ("retained earnings", "accumulated profits"),
    (
        "depreciation and amortisation",
        "depreciation and amortization",
    ),
)


def _to_page_record(page, index):
    if isinstance(page, dict):
        record = dict(page)
        metadata = record.get("metadata") or {}
        text = record.get("text")

        if text is None:
            text = record.get("page_content", "")
    else:
        metadata = getattr(page, "metadata", {}) or {}
        text = getattr(page, "page_content", "")
        record = {}

    metadata = dict(metadata)
    text = str(text or "")

    existing_number = record.get("page_number")
    metadata_page = metadata.get("page")
    metadata_page_number = metadata.get("page_number")

    if isinstance(existing_number, int) and existing_number >= 1:
        page_number = existing_number
    elif isinstance(metadata_page, int) and metadata_page >= 0:
        page_number = metadata_page + 1
    elif isinstance(metadata_page_number, int) and metadata_page_number >= 1:
        page_number = metadata_page_number
    else:
        page_number = index + 1

    metadata["page_number"] = page_number
    record["metadata"] = metadata
    record["text"] = text
    record["page_content"] = text
    record["page_number"] = page_number

    return record


def _contains_term(normalized_text, term):
    if not normalized_text or not term:
        return False

    if term in ("current assets", "current liabilities"):
        pattern = (
            r"(?<!non )(?<!\w)"
            + re.escape(term)
            + r"(?!\w)"
        )
    else:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

    return re.search(pattern, normalized_text) is not None


def _metric_terms(variable):
    base_term = normalize_text(variable)

    if not base_term:
        return set()

    terms = {base_term}

    for group in METRIC_SYNONYM_GROUPS:
        normalized_group = {
            normalize_text(term)
            for term in group
        }

        if base_term in normalized_group:
            terms.update(normalized_group)

    return terms


def _build_template_metrics(excel_data):
    template_metrics = []

    for metric in excel_data:
        variable = str(metric.get("variable") or "").strip()
        worksheet = str(metric.get("worksheet") or "").strip()

        if not variable:
            continue

        terms = _metric_terms(variable)

        template_metrics.append(
            {
                "key": (worksheet, variable),
                "label": f"{worksheet}::{variable}",
                "terms": terms,
            }
        )

    return template_metrics


def _find_metric_matches(normalized_text, template_metrics):
    matched_keys = set()
    matched_labels = []

    for metric in template_metrics:
        if any(
            _contains_term(normalized_text, term)
            for term in metric["terms"]
        ):
            matched_keys.add(metric["key"])
            matched_labels.append(metric["label"])

    return matched_keys, matched_labels


def _page_relevance(text, matched_metric_count):
    normalized_text = normalize_text(text)

    years = set(
        re.findall(r"\b(?:19|20)\d{2}\b", text)
    )

    continuation_bonus = 1.5 if any(
        _contains_term(normalized_text, marker)
        for marker in CONTINUATION_MARKERS
    ) else 0.0

    score = (
        matched_metric_count * 10.0
        + min(numeric_density(text) * 30.0, 5.0)
        + table_structure_score(text) * 2.0
        + min(len(years) * 0.5, 2.0)
        + continuation_bonus
    )

    return round(score, 2)


def _baseline_selection(pdf_pages, settings):
    baseline_pages = []

    for page in pdf_pages[:settings.first_n_pages]:
        baseline_pages.append(page)

    for page in pdf_pages[settings.first_n_pages:]:
        if page_score(page.get("text", "")) > settings.page_score_threshold:
            baseline_pages.append(page)

    if len(pdf_pages) < settings.min_pages:
        baseline_pages = pdf_pages

    return baseline_pages


def select_pages(state: FinancialGraphState) -> dict:
    settings = get_settings()

    raw_pages = state.get("pdf_pages", [])
    excel_data = state.get("excel_data", [])
    errors = list(state.get("errors", []))

    pdf_pages = [
        _to_page_record(page, index)
        for index, page in enumerate(raw_pages)
    ]

    baseline_pages = _baseline_selection(
        pdf_pages,
        settings,
    )

    if not excel_data:
        errors.append(
            "Page selection requires template metrics in excel_data. "
            "Run read_creditlens_template before select_pages."
        )

        return {
            "pdf_pages": pdf_pages,
            "selected_pages": [],
            "baseline_selected_pages": baseline_pages,
            "errors": errors,
        }

    template_metrics = _build_template_metrics(excel_data)
    candidates = []

    for index, page in enumerate(pdf_pages):
        text = page.get("text", "")
        normalized_text = normalize_text(text)

        matched_keys, matched_labels = _find_metric_matches(
            normalized_text,
            template_metrics,
        )

        if not matched_keys:
            continue

        candidates.append(
            {
                "index": index,
                "page": page,
                "page_number": page["page_number"],
                "metric_keys": matched_keys,
                "matched_metrics": matched_labels,
                "score": _page_relevance(
                    text,
                    len(matched_keys),
                ),
            }
        )

    max_selected_pages = max(
        0,
        int(settings.max_selected_pages),
    )

    remaining = list(candidates)
    selected_candidates = []
    covered_metrics = set()

    while remaining and len(selected_candidates) < max_selected_pages:
        best = max(
            remaining,
            key=lambda candidate: (
                len(candidate["metric_keys"] - covered_metrics),
                candidate["score"],
                -candidate["page_number"],
            ),
        )

        selected_candidates.append(best)
        covered_metrics.update(best["metric_keys"])
        remaining.remove(best)

    selected_indices = {
        candidate["index"]
        for candidate in selected_candidates
    }

    selected_pages = [
        page
        for index, page in enumerate(pdf_pages)
        if index in selected_indices
    ]

    if not selected_pages:
        errors.append(
            "No PDF pages matched the template metric names or "
            "configured synonyms. Review the metric names and "
            "METRIC_SYNONYM_GROUPS."
        )

    print(
        "Improved selected pages:",
        [page["page_number"] for page in selected_pages],
    )
    print("Improved selected page count:", len(selected_pages))
    print(
        "Template metrics matched:",
        len(covered_metrics),
        "of",
        len({metric["key"] for metric in template_metrics}),
    )

    for candidate in sorted(
        selected_candidates,
        key=lambda item: item["page_number"],
    ):
        print(
            f"PDF page {candidate['page_number']}: "
            f"{candidate['matched_metrics']}"
        )

    return {
        "pdf_pages": pdf_pages,
        "selected_pages": selected_pages,
        "baseline_selected_pages": baseline_pages,
        "errors": errors,
    }

```
### prompts/selection_keywords.py
```python

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_terms(terms):
    return [
        (term, normalized)
        for term in terms
        if (normalized := normalize_text(term))
    ]


_NORMALIZED_KEYWORDS = _normalize_terms(all_keywords)
_NORMALIZED_STRONG_TERMS = _normalize_terms(
    STRONG_FINANCIAL_TERMS
)
_NORMALIZED_UNIT_TERMS = _normalize_terms(UNIT_TERMS)
_NORMALIZED_LOW_VALUE_TERMS = _normalize_terms(LOW_VALUE_TERMS)


def _contains_normalized_phrase(
    normalized_text: str,
    normalized_phrase: str,
) -> bool:
    if not normalized_text or not normalized_phrase:
        return False

    return re.search(
        r"(?<!\w)" + re.escape(normalized_phrase) + r"(?!\w)",
        normalized_text,
    ) is not None


def contains_phrase(text: str, phrase: str) -> bool:
    return _contains_normalized_phrase(
        normalize_text(text),
        normalize_text(phrase),
    )


def page_score(text: str) -> int:
    normalized_text = normalize_text(text)

    return sum(
        _contains_normalized_phrase(normalized_text, phrase)
        for _, phrase in _NORMALIZED_KEYWORDS
    )


def numeric_density(text: str) -> float:
    tokens = str(text).split()

    if not tokens:
        return 0.0

    numbers = re.findall(
        r"(?<!\w)\(?[-+]?\d[\d,]*(?:\.\d+)?%?\)?(?!\w)",
        str(text),
    )

    return min(len(numbers) / len(tokens), 1.0)


def table_structure_score(text: str) -> float:
    lines = [
        line.strip()
        for line in str(text).splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    numeric_lines = sum(
        1 for line in lines
        if len(re.findall(r"\d[\d,.]*", line)) >= 2
    )

    score = min(
        (numeric_lines / len(lines)) * 2,
        2.0,
    )

    if any(
        re.search(r"\s{2,}|\t|\|", line)
        for line in lines
    ):
        score += 1.0

    return min(score, 3.0)


def page_relevance_score(text: str) -> dict:
    normalized_text = normalize_text(text)

    if not normalized_text:
        return {
            "score": 0.0,
            "keyword_matches": [],
        }

    keyword_matches = [
        term
        for term, phrase in _NORMALIZED_KEYWORDS
        if _contains_normalized_phrase(
            normalized_text,
            phrase,
        )
    ]

    strong_matches = [
        term
        for term, phrase in _NORMALIZED_STRONG_TERMS
        if _contains_normalized_phrase(
            normalized_text,
            phrase,
        )
    ]

    unit_matches = [
        term
        for term, phrase in _NORMALIZED_UNIT_TERMS
        if _contains_normalized_phrase(
            normalized_text,
            phrase,
        )
    ]

    years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    number_density = numeric_density(text)
    table_score = table_structure_score(text)

    score = (
        min(len(strong_matches) * 3.0, 12.0)
        + min(len(keyword_matches) * 1.0, 10.0)
        + min(len(unit_matches) * 1.5, 4.5)
        + min(len(years) * 0.75, 3.0)
        + min(number_density * 30.0, 5.0)
        + table_score
    )

    penalty = sum(
        1.0
        for _, phrase in _NORMALIZED_LOW_VALUE_TERMS
        if _contains_normalized_phrase(
            normalized_text,
            phrase,
        )
    )

    if len(normalized_text.split()) < 30:
        penalty += 1.0

    score = max(score - min(penalty, 3.0), 0.0)

    return {
        "score": round(score, 2),
        "keyword_matches": keyword_matches,
    }

```

### nodes/answer_questions.py
```python

def answer_questions(state: FinancialGraphState) -> dict:
    """Extract metrics via LLM and enrich each answer with an Excel hyperlink."""

    excel_data = state.get("excel_data", [])
    errors = list(state.get("errors", []))
    selected_pages = state.get("selected_pages", [])

    if not selected_pages:
        final_answers = align_answers_to_template(
            excel_data,
            [],
        )

        return {
            "original_answers": final_answers,
            "answers": final_answers,
            "errors": errors + [
                "Extraction skipped because no template-relevant pages were selected."
            ],
        }

    llm = state.get("llm")
    structured_llm = llm.with_structured_output(ExtractionResponse)

    pdf_file_name = state.get("pdf_file_name", "")

```

```python

full_context = "\n\n".join(
    [
        f"--- PDF Page {page.get('page_number', i + 1)} ---\n"
        f"{page.get('text', '')}"
        for i, page in enumerate(selected_pages)
    ]
)
```

```python

user_content = (
    f"Please extract the following variables:\n{USER_PROMPT}\n\n"
    "Use only the supplied document context. "
    "For page_number, return the physical PDF page number shown "
    "in the context header. Do not use the selected-page position "
    "or context order as the page number.\n\n"
    f"Document context:\n{full_context}\n\n"
)
```

### config.py
```python
# --- Page selection ---
first_n_pages = 75
page_score_threshold = 1
min_pages = 30
max_selected_pages = 10
refinement_attempts = 0
```

```text
Improve template-based page selection and scoring
```
- Select relevant PDF pages using Excel template metrics and configured synonyms.
- Rank candidate pages by metric coverage, numeric density, and table structure.
- Limit improved page selection to 10 pages by default.
- Preserve original PDF page numbers in extraction context and source hyperlinks.
- Optimize keyword normalization and page relevance scoring.
- Keep baseline page selection separate for comparison.
- Skip LLM extraction when no relevant pages are selected.
- Preserve template-defined metrics and use N/A for unavailable answers.
- Keep refinement attempts disabled during page-selection testing.
```text
```

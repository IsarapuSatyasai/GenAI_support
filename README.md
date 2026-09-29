```python
baseline_selected_pages: List[Dict[str, Any]]
```

```python
def select_pages(state: FinancialGraphState) -> dict:
    settings = get_settings()
    pdf_pages = state.get("pdf_pages", [])
    n = settings.first_n_pages

    # Baseline page selection
    baseline_pages = []

    for page in pdf_pages[:n]:
        baseline_pages.append(page)

    for page in pdf_pages[n:]:
        page_text = page["text"]

        if page_score(page_text) > settings.page_score_threshold:
            baseline_pages.append(page)

    if len(pdf_pages) < settings.min_pages:
        baseline_pages = pdf_pages

    # Improved selection
    improved_pages = sorted(
    pdf_pages,
    key=lambda page: page_score(page.get("text", "")),
    reverse=True,
    )[:settings.max_selected_pages]

    return {
        "selected_pages": improved_pages,
        "baseline_selected_pages": baseline_pages,
        "errors": state.get("errors", []),
    }
```

```python
"confidence": original.get("confidence"),
```

```python
def build_page_selection_comparison(
    pdf_pages,
    baseline_pages,
    improved_pages,
):
    """Compare baseline and improved page selection."""

    baseline_numbers = {
        page.get("page_number")
        for page in baseline_pages
    }

    improved_numbers = {
        page.get("page_number")
        for page in improved_pages
    }

    rows = []

    for index, page in enumerate(pdf_pages, start=1):
        page_number = page.get("page_number", index)

        baseline_selected = page_number in baseline_numbers
        improved_selected = page_number in improved_numbers

        rows.append(
            {
                "page_number": page_number,
                "baseline_selected": baseline_selected,
                "improved_selected": improved_selected,
                "changed": (
                    baseline_selected != improved_selected
                ),
            }
        )

    return pd.DataFrame(rows)
```

```python
page_selection_comparison = build_page_selection_comparison(
    state.get("pdf_pages", []),
    state.get("baseline_selected_pages", []),
    state.get("selected_pages", []),
)

comparison_dataframes["Page Selection"] = page_selection_comparison
```

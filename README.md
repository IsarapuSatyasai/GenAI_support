```python
"""System prompt for financial spreading: LLM Agent will always follow these set of instructions."""

SYSTEM_PROMPT = """\
You are a senior credit analyst who works on financial spreading. You are specializing in extracting structured financial data from annual report PDFs.

## Critical extraction discipline:

- Strictly and deeply examine both the Excel template metrics and the full content of the input PDF before returning any value.
- Cross-check every template variable against the primary statements, notes, and any comparative columns. Do not stop at the first possible match.
- Prefer evidence that is explicitly present in the PDF over any assumption. If a number or mapping is visible, extract it.
- When in doubt, re-scan the relevant statement and notes rather than defaulting to N/A.
- Treat the template as the complete list of required outputs and the PDF as the sole source of truth.

## Non-negotiable rules:

1. **All available periods** - Extract values for every comparative year present in the document (typically the current and prior year). Return one answer object per metric per year. Do not drop prior-year columns. If a value exists for 2023, you MUST return it. Never leave a prior year as N/A when the number is visible on the statement.
2. **Consolidated preferred** - Extract consolidated financial statements if available. If not, extract the most detailed statements available.
3. **No guessing** - Do NOT invent numbers. If a metric cannot be found or reliably derived, return "N/A" and explain what is missing.
4. **Keep source details** - Report numerical values exactly as they appear. Always capture: currency (EUR, GBP, USD, etc.), scale (units, thousands, millions), fiscal year end, consolidation scope, and period label.
5. **English output** - Always provide output like remarks, explanations, and other text in English, even when the source document is in another language (e.g. Dutch, German, etc.). Translate faithfully and accurately.
6. **MetaData** - For text fields, always return the exact text as it appears in the PDF. Use description provided in the user prompt to check expected values.
7. **Evidence** - for every metric, provide a page reference (page number as shown in the PDF). Also provide hyperlink to the source document section.
8. Do not skip. Always provide one output for every variable supplied by the Excel template.
   If the metric is not found in the document:
       answer = "N/A"
       confidence = 0.0
       page_number = -1
       source_fields = []
       formula = null
       source_heading = null
       source_text = null
9. **Confidence score** - Rate each extraction on the scale between 0 to 10, where 0 means not found, 10 means 100% confident. And 1 means not confident at all.
10. **Preserve exact formatting** - Return numeric answers with the digits and thousand separators exactly as they appear in the source document (e.g. 13,950). Do not remove commas or convert to plain integers. The only change allowed is the sign, which must follow rule 11: never return parentheses; write a negative value with a leading minus (e.g. (74,929) in the source becomes -74,929 for a contra item).
11. **Consistency** - Keep sign conventions consistent:
- Assets: positive
- Liabilities & equity: positive
- Contra items (accumulated depreciation/amortization, provisions/revaluation of inventory, doubtful debtor provisions, treasury stock): negative
- P&L: revenues and costs/expenses are both positive. Report costs/expenses (cost of goods sold, salaries, labour costs, general costs, depreciation, amortisation, interest paid, tax, etc.) as positive amounts even when the source shows them in parentheses or with a minus sign (e.g. (606,898) in the source becomes 606,898). Only items that genuinely reduce cost or represent a loss/negative result (e.g. a net loss, negative EBIT, a cost reversal or release) are negative.
- Total assets: Not equal to 0
- Net fixed assets: Not less than 0
- Total liabilities: Not equal to 0
- Current assets: Not less than 0
- Inventory: Not less than 0
- Net worth: Not equal to 0
- Current liabilities: Not less than 0
- Net sales: Greater than 0
- Accounts Receivable Trade: Not less than 0
- Cash and Bank: Not less than 0
- Loans: Not less than 0
- Perform integrity checks and report results:
 a) Balance sheet must balance. Verify that the accounting equation holds: Total Assets = Total Equity & Reserves & Liabilities (or Assets = Liabilities + Equity)
 b) Cash flow net change equals change in Cash & Bank between periods (if cash flow statement present)
 c) Subtotals equal sum of components when components exist
12. **Source field mapping** - Always provide the source field name for each extracted value. This is the name of the field in the source document.
 - If a PDF line item matches directly with extracted metric and has 1 to 1 mapping, then provide name of original PDF line item as 'source_fields' and keep the 'formula' value as null.
 - If a extracted metric is sum of several PDF line items, then list every original source field name in 'source_fields' and set 'formula' to the sum.
 For e.g. if you need to extract value for 'Tax Payables & Social Security Contr.' which is sum of 'Corporation Tax' and 'Other taxation and Social Security', then set 'source_fields' to ['Corporation Tax', 'Other taxation and Social Security'] and 'formula' to 'Corporation Tax + Other taxation and Social Security'.
13. **Name mapping and primary-statement preference**
   Template variable names may not match PDF labels exactly. Always search for a clear semantic equivalent before returning N/A.

   Prefer the amount shown on the face of the primary consolidated statements (Balance Sheet, Income Statement, Cash Flow Statement) over note breakdowns when both exist and represent the same concept.

   Rules:
   - Prefer a matching or clearly equivalent line on the face of the primary statement.
   - Only use a note sub-line when the face total is not available or the template explicitly requires the breakdown.
   - When you map a different label, set source_fields to the original PDF label and include the heading + full source text.
   - Be consistent: if you map “Trade and other receivables” to Accounts Receivable Trade, use the face balance-sheet total, not a note component.
   - Prefer a matching PDF line over N/A when the label clearly represents the same metric.
14. **Mandatory mappings for these specific template variables** (do not return N/A if the equivalent exists):
   - Other Labour Costs: MUST map to “Social securities and pension costs”, “Total Labour Costs”, “Staff Costs”, or any combined labour line. Do not leave Other Labour Costs as N/A when any labour cost line is present.
   - General Costs: must map to “Other Expenses”, “Other operating expenses”, or “Administrative expenses”.
   - Interest Received - cash: MUST map to “Interest income” or “Interest Received” on the Income Statement or Cash Flow Statement. Do not return N/A if Interest income is shown.
   - Income from Other Finan Assets - cash: must map to “Finance income”, “Income From Other Financial Assets”, or “Finance income received”.
   When mapping, always fill source_heading and source_text with the original PDF heading and line.
15. **Source attribution** - For every metric, also provide:
   - source_heading: the section, table, or paragraph heading where the value was found (or the closest matching section if the exact metric is N/A).
   - source_text: the COMPLETE relevant paragraph or full table that contains the value or the closest related line.
     Even when the answer is "N/A", still return the nearest matching table/paragraph (e.g. the Total Labour Costs line when Salaries/Other Labour Costs cannot be split) so a human can see what context was considered.
   First locate the relevant heading/section, then extract the value (or N/A) and the complete source context.
16. **Primary statement totals and related items**
   - Depreciation: MUST map to the combined “Amortization and depreciation” (or “Depreciation and amortisation”) line from the face of the Income Statement. Do not return N/A if this line exists. Return the value for every year shown.
   - Amortisation: extract only the intangible amortisation amount (from the intangible assets note or cash-flow breakdown). Do not copy the combined Depreciation figure into Amortisation.
   - Accounts Receivable Trade → prefer “Trade and other receivables” from the Balance Sheet.
   - Account Payable Trade → prefer “Trade and other payables” from the Balance Sheet (not the “Trade payables” sub-line in the note).
   - Deferred Taxes Asset → “Deferred tax assets” from the Balance Sheet.
   - Lease Obligations – Current / LT Portion → current and non-current “Lease liability” from the Balance Sheet.
   - Tax Payables & Social Security Contr. → prefer “Current income tax payables” or the closest face total; if only a note breakdown exists, use the relevant total and document it.

   When both a face total and a note breakdown exist, report the face total unless the template variable clearly asks for a component.

17. **Multi-year requirement** - For every variable you extract, you MUST return a separate answer object for each year present in the document (e.g. 2024 AND 2023). Never return only the latest year. If a prior-year value is visible, include it. If a prior-year value is truly absent, still return an object for that year with answer = "N/A". Include the year field on every answer object.

## Output format:
Return a single JSON object matching the ExtractionResponse schema provided in the user message. Do not add anything outside JSON object.
"""
```

```python
"""Node: RAG recovery for N/A metrics.

Builds an in-memory FAISS index from the already-selected pages.
Retrieves top-k chunks per batch. No download, no persistent store.
Writes recovered values into the correct year columns.
"""

from typing import Any, Dict, List, Optional
import logging
import time
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from custom_lcel_components.llm import create_llm
from graph.graph_state import FinancialGraphState
from config import get_settings
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.user_prompt import get_user_prompt
from models import MultiYearExtractionResponse

logger = logging.getLogger(__name__)

def _is_na(answer: Dict[str, Any]) -> bool:
    val = str(answer.get("answer", "")).strip().upper()
    try:
        conf = float(answer.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        conf = 0.0
    return val in ("N/A", "NA", "", "NONE") or conf == 0.0

def _build_inmemory_index(
    selected_pages: List[Dict[str, Any]],
    embedding_model: AzureOpenAIEmbeddings,
) -> Optional[FAISS]:
    """Create an in-memory FAISS index from selected pages."""
    texts = []
    metadatas = []
    for page in selected_pages:
        text = page.get("text") or page.get("content") or ""
        if not text.strip():
            continue
        page_no = page.get("page_number") or page.get("page") or -1
        texts.append(text)
        metadatas.append({"page_number": page_no})

    if not texts:
        logger.warning("rag_recovery: no text found in selected_pages")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    docs = splitter.create_documents(texts, metadatas=metadatas)
    if not docs:
        logger.warning("rag_recovery: no chunks created from selected_pages")
        return None

    logger.info(f"Building in-memory FAISS index from {len(docs)} chunks")
    return FAISS.from_documents(docs, embedding_model)

def _update_answer_for_year(
    answers_by_year: Dict[Any, List[Dict[str, Any]]],
    variable: str,
    year: Any,
    new_answer: str,
    confidence: float,
    source_fields: list,
    source_text: str,
) -> bool:
    """Write a recovered value into the correct year bucket. Returns True if updated."""
    year_list = answers_by_year.get(year)
    if not year_list:
        return False
    for item in year_list:
        if item.get("variable") == variable and _is_na(item):
            item["answer"] = new_answer
            item["confidence"] = confidence
            item["source_fields"] = source_fields
            item["source_text"] = source_text[:500]
            item["source_link"] = "RAG"
            return True
    return False

def rag_recovery(state: FinancialGraphState) -> dict:
    answers: List[Dict[str, Any]] = state.get("answers", [])
    answers_by_year: Dict[Any, List[Dict[str, Any]]] = state.get("answers_by_year", {})
    excel_data = state.get("excel_data", [])
    selected_pages = state.get("selected_pages", [])
    errors = list(state.get("errors", []))

    if not answers:
        return {"errors": errors + ["rag_recovery skipped: no answers"]}
    if not selected_pages:
        return {"errors": errors + ["rag_recovery skipped: no selected_pages"]}

    metric_lookup = {
        (m.get("worksheet"), m.get("variable")): m for m in excel_data
    }

    # Collect N/A items across all years
    na_items = []
    for year, year_answers in answers_by_year.items():
        for a in year_answers:
            if _is_na(a):
                na_items.append((year, a))

    if not na_items:
        logger.info("rag_recovery: no N/A metrics to recover")
        return {"use_rag": False}

    logger.info(
        f"rag_recovery: recovering {len(na_items)} N/A metric-year pairs "
        f"from {len(selected_pages)} selected pages"
    )

    settings = get_settings()

    try:
        embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=settings.rag_embedding_model,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", settings.azure_openai_api_version
            ),
        )
        vectorstore = _build_inmemory_index(selected_pages, embedding_model)
        if vectorstore is None:
            return {"errors": errors + ["rag_recovery: failed to build in-memory index"]}

        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    except Exception as exc:
        logger.exception("Failed to build in-memory index")
        return {"errors": errors + [f"Failed to build index: {exc}"]}

    llm = create_llm(
        azure_deployment=settings.azure_openai_deployment,
        top_p=settings.llm_top_p,
    )
    structured_llm = (
        llm.with_structured_output(MultiYearExtractionResponse)
        .with_retry(wait_exponential_jitter=True, stop_after_attempt=3)
    )

    BATCH_SIZE = 6
    recovered = 0
    start_time = time.time()
    years = state.get("years") or list(answers_by_year.keys())

    # Group N/A items by variable so we ask for all years of a metric together
    from collections import defaultdict
    by_var = defaultdict(list)
    for year, item in na_items:
        by_var[item.get("variable")].append((year, item))

    var_list = list(by_var.keys())

    for batch_idx, i in enumerate(range(0, len(var_list), BATCH_SIZE)):
        batch_vars = var_list[i : i + BATCH_SIZE]

        user_metrics = []
        for var in batch_vars:
            sample = by_var[var][0][1]
            user_metrics.append({
                "worksheet": sample.get("worksheet"),
                "variable": var,
                "description": metric_lookup.get(
                    (sample.get("worksheet"), var), {}
                ).get("description"),
            })

        user_prompt = get_user_prompt(user_metrics)

        year_instruction = (
            f"You MUST return one answer object for EVERY year in this list: {years}. "
            "Do not omit any year. If a value exists for a prior year (e.g. 2023), include it. "
            "If a value does not exist for a year, still return an object for that year with answer = 'N/A'. "
            "Include the year field on every answer. "
            "Apply the mandatory mappings from the system prompt: "
            "Depreciation → Amortization and depreciation (both years), "
            "Other Labour Costs → Social securities and pension costs (both years), "
            "Interest Received - cash → Interest income (both years)."
        )

        query = " ".join(batch_vars)

        logger.info(
            f"rag_recovery: batch {batch_idx + 1}/"
            f"{(len(var_list) + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"({len(batch_vars)} metrics)"
        )

        try:
            docs = retriever.invoke(query)
            context = "\n\n".join(
                f"[Page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
                for d in docs
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Please extract only the following missing variables:\n"
                        f"{user_prompt}\n\n"
                        f"{year_instruction}\n\n"
                        f"Document context:\n{context}"
                    )
                ),
            ]

            response: MultiYearExtractionResponse = structured_llm.invoke(messages)

            for match in response.answers:
                var = match.variable
                year = getattr(match, "year", None)
                if not var or year is None:
                    continue
                if (
                    match.answer
                    and str(match.answer).strip().upper() not in ("N/A", "NA", "")
                ):
                    updated = _update_answer_for_year(
                        answers_by_year,
                        var,
                        year,
                        str(match.answer).strip(),
                        match.confidence or 0.65,
                        match.source_fields or [],
                        match.source_text or "",
                    )
                    if updated:
                        recovered += 1
                        logger.info(
                            f"  Recovered: {var} ({year}) → {str(match.answer)[:60]}"
                        )

        except Exception as exc:
            logger.warning(f"Batch {batch_idx + 1} failed: {exc}")
            errors.append(f"rag_recovery batch {batch_idx + 1} error: {exc}")

    # Rebuild the flat answers list from the latest year (for downstream nodes)
    latest = years[0] if years else None
    if latest is not None and latest in answers_by_year:
        answers = answers_by_year[latest]

    elapsed = time.time() - start_time
    logger.info(
        f"rag_recovery finished: recovered {recovered} metric-year pairs "
        f"in {elapsed:.1f}s (in-memory FAISS, batch size {BATCH_SIZE}, k=10)"
    )

    return {
        "answers": answers,
        "answers_by_year": answers_by_year,
        "errors": errors,
        "use_rag": True,
        "rag_collection_name": "in-memory-faiss",
    }
```

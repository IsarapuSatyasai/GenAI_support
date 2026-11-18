```python
"""Node: RAG recovery for N/A metrics.

Builds a temporary in-memory Chroma index from the pages already chosen
by the upstream selection step. For every metric that is still missing,
it retrieves the most relevant chunks and asks the model to fill the
value for each year present in the document. The index is discarded
when the node finishes.
"""

from typing import Any, Dict, List, Optional
import logging
import time
import os
from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from custom_lcel_components.llm import create_llm
from graph.graph_state import FinancialGraphState
from config import get_settings
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.user_prompt import get_user_prompt
from models import MultiYearExtractionResponse

logger = logging.getLogger(__name__)

SYNONYM_MAP = {
    "Depreciation": "amortization and depreciation depreciation and amortisation depreciation",
    "Amortisation": "amortization intangible assets amortisation",
    "Other Labour Costs": "social securities and pension costs staff costs total labour costs employee benefit expense wages",
    "Interest Received - cash": "interest income interest received interest income received",
    "General Costs": "other operating expenses administrative expenses other expenses",
    "Income from Other Finan Assets - cash": "finance income finance income received income from other financial assets",
    "3rd Party Svc's (Sub-Contr Work)": "costs incurred from financial institutions",
    "Accounts Receivable Trade": "trade and other receivables receivables from customers",
    "Account Payable Trade": "trade and other payables trade payables",
    "Deferred Taxes Asset": "deferred tax assets",
    "Other Income": "other income/(expenses) other income",
    "Salaries": "wages and salaries staff costs",
}

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
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> Optional[Chroma]:
    docs = []
    page_limit = 2500

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for page in selected_pages:
        text = page.get("text") or page.get("content") or ""
        if not text.strip():
            continue
        page_no = page.get("page_number") or page.get("page") or -1
        meta = {"page_number": page_no}

        if len(text) <= page_limit:
            docs.append(Document(page_content=text, metadata=meta))
        else:
            splits = recursive_splitter.create_documents([text], metadatas=[meta])
            docs.extend(splits)

    if not docs:
        logger.warning("rag_recovery: no text found in selected_pages")
        return None

    logger.info(f"Building in-memory Chroma index from {len(docs)} chunks")
    return Chroma.from_documents(docs, embedding_model)

def _update_answer_for_year(
    answers_by_year: Dict[Any, List[Dict[str, Any]]],
    variable: str,
    year: Any,
    new_answer: str,
    confidence: float,
    source_fields: list,
    source_text: str,
    page_number: Any = -1,
) -> bool:
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
            item["page_number"] = page_number
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
    chunk_size = getattr(settings, "rag_chunk_size", 1500)
    chunk_overlap = getattr(settings, "rag_chunk_overlap", 150)
    top_k = getattr(settings, "rag_similarity_top_k", 10)
    batch_size = getattr(settings, "rag_batch_size", 3)

    try:
        embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=settings.rag_embedding_model,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", settings.azure_openai_api_version
            ),
        )
        vectorstore = _build_inmemory_index(
            selected_pages, embedding_model, chunk_size, chunk_overlap
        )
        if vectorstore is None:
            return {"errors": errors + ["rag_recovery: failed to build in-memory index"]}

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
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

    recovered = 0
    start_time = time.time()
    years = state.get("years") or list(answers_by_year.keys())

    by_var = defaultdict(list)
    for year, item in na_items:
        by_var[item.get("variable")].append((year, item))

    var_list = list(by_var.keys())

    for batch_idx, i in enumerate(range(0, len(var_list), batch_size)):
        batch_vars = var_list[i : i + batch_size]

        user_metrics = []
        query_parts = []
        for var in batch_vars:
            sample = by_var[var][0][1]
            desc = metric_lookup.get(
                (sample.get("worksheet"), var), {}
            ).get("description") or ""
            remarks = metric_lookup.get(
                (sample.get("worksheet"), var), {}
            ).get("remarks") or ""
            user_metrics.append({
                "worksheet": sample.get("worksheet"),
                "variable": var,
                "description": desc,
            })
            query_parts.append(var)
            if var in SYNONYM_MAP:
                query_parts.append(SYNONYM_MAP[var])
            if desc:
                query_parts.append(desc)
            if remarks:
                query_parts.append(remarks)

        user_prompt = get_user_prompt(user_metrics)

        year_instruction = (
            f"You MUST return one answer object for EVERY year in this list: {years}. "
            "Strictly follow the template instructions, remarks, and descriptions. "
            "Search for synonyms and accounting equivalents of each variable before returning N/A. "
            "If an equivalent line exists on the primary statements or notes, extract it for every year shown. "
            "Include the page number where the value was found. "
            "Do not omit any year."
        )

        query = " ".join(query_parts)

        logger.info(
            f"rag_recovery: batch {batch_idx + 1}/"
            f"{(len(var_list) + batch_size - 1) // batch_size} "
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
                        getattr(match, "page_number", -1),
                    )
                    if updated:
                        recovered += 1
                        logger.info(
                            f"  Recovered: {var} ({year}) → {str(match.answer)[:60]}"
                        )

        except Exception as exc:
            logger.warning(f"Batch {batch_idx + 1} failed: {exc}")
            errors.append(f"rag_recovery batch {batch_idx + 1} error: {exc}")

    latest = years[0] if years else None
    if latest is not None and latest in answers_by_year:
        answers = answers_by_year[latest]

    elapsed = time.time() - start_time
    logger.info(
        f"rag_recovery finished: recovered {recovered} metric-year pairs "
        f"in {elapsed:.1f}s (in-memory Chroma, batch size {batch_size}, k={top_k})"
    )

    return {
        "answers": answers,
        "answers_by_year": answers_by_year,
        "errors": errors,
        "use_rag": True,
        "rag_collection_name": "in-memory-chroma",
    }
```

```python
"""System prompt for financial spreading: LLM Agent will always follow these set of instructions."""

SYSTEM_PROMPT = """\
You are a senior credit analyst who works on financial spreading. You are specializing in extracting structured financial data from annual report PDFs.

## Critical extraction discipline:

- Strictly and deeply examine both the Excel template metrics and the full content of the input PDF before returning any value.
- Cross-check every template variable against the primary statements, notes, and any comparative columns. Do not stop at the first possible match.
- Prefer evidence that is explicitly present in the PDF over any assumption. If a number or mapping is visible, extract it.
- For template variables that use different wording from the PDF (for example Other Labour Costs, Income from Other Financial Assets, Depreciation, General Costs, Interest Received), extract the value from the matching synonym or accounting equivalent on the primary statements or notes. Do not return N/A when a clear equivalent line is present.
- When in doubt, re-scan the relevant statement and notes rather than defaulting to N/A.
- Treat the template as the complete list of required outputs and the PDF as the sole source of truth.
- Search for synonyms, near-synonyms, and common accounting equivalents of every template variable before returning N/A.
- Follow every instruction, remark, and description given in the template for each variable.
- Returning N/A is allowed only when no equivalent line can be found after checking the primary statements, notes, and synonyms.
- Extract values explicitly as they appear in the document. Do not infer or approximate numbers beyond the required sign convention.

## Non-negotiable rules:

1. **All available periods** - Extract values for every comparative year present in the document. Return one answer object per metric per year. Do not drop prior-year columns. If a value exists for a prior year, you MUST return it.
2. **Consolidated preferred** - Extract consolidated financial statements if available. If not, extract the most detailed statements available.
3. **No guessing** - Do NOT invent numbers. If a metric cannot be found or reliably derived, return "N/A" and explain what is missing.
4. **Keep source details** - Report numerical values exactly as they appear. Always capture currency, scale, fiscal year end, consolidation scope, and period label.
5. **English output** - Always provide output in English, even when the source document is in another language.
6. **MetaData** - For text fields, always return the exact text as it appears in the PDF. Use the description provided in the user prompt to check expected values.
7. **Evidence** - For every metric, provide a page reference (page number as shown in the PDF).
8. Do not skip. Always provide one output for every variable supplied by the Excel template.
   If the metric is not found:
       answer = "N/A"
       confidence = 0.0
       page_number = -1
       source_fields = []
       formula = null
       source_heading = null
       source_text = null
9. **Confidence score** - Rate each extraction on a scale from 0 to 10.
10. **Preserve exact formatting** - Return numeric answers with the digits and thousand separators exactly as they appear. The only change allowed is the sign.
11. **Consistency** - Keep sign conventions consistent. Assets and liabilities positive. Contra items negative. P&L costs and revenues positive unless they represent a genuine loss or reversal.
12. **Source field mapping** - Provide the original PDF line-item name. For sums, list every source field and set the formula.
13. **Name mapping and primary-statement preference** - Prefer the face of the primary consolidated statements over note breakdowns. Search for a clear semantic equivalent before returning N/A.
14. **Source attribution** - Provide source_heading and source_text for every metric, including N/A.
15. **Multi-year requirement** - Return a separate answer object for each year. Include the year and page_number fields on every answer object.

## Output format:
Return a single JSON object matching the ExtractionResponse schema provided in the user message. Do not add anything outside the JSON object.
"""
```

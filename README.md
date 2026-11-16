```python
"""RAG recovery for remaining N/A metrics.

Builds a transient in-memory FAISS index from the already-selected pages
using sentence-window chunking, retrieves top-k chunks, and writes
recovered values into the correct year buckets.
Uses the same SYSTEM_PROMPT as the primary LLM extraction.
No persistent store is used.
"""

from typing import Any, Dict, List, Optional
import logging
import time
import os
from collections import defaultdict

from langchain_community.vectorstores import FAISS
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
    """Create a transient in-memory FAISS index using sentence-window chunking."""
    docs: List[Document] = []

    for page in selected_pages:
        text = page.get("text") or page.get("content") or ""
        if not text.strip():
            continue
        page_no = page.get("page_number") or page.get("page") or -1

        # Split into sentences
        raw_sentences = text.replace("\n", " ").split(". ")
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            continue

        window_size = 5      # sentences in each window
        step = 3             # overlap of 2 sentences

        for i in range(0, len(sentences), step):
            window = sentences[i:i + window_size]
            chunk_text = ". ".join(window)
            if len(chunk_text) < 40:
                continue
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={"page_number": page_no},
                )
            )

    if not docs:
        logger.warning("rag_recovery: no sentence-window chunks created")
        return None

    logger.info("Building in-memory FAISS index from %d sentence-window chunks", len(docs))
    return FAISS.from_documents(docs, embedding_model)


def _update_answer_for_year(
    answers_by_year: Dict[Any, List[Dict[str, Any]]],
    variable: str,
    year: Any,
    new_answer: str,
    confidence: float,
    source_fields: list,
    source_text: str,
    source_heading: Optional[str] = None,
    page_number: Optional[int] = None,
) -> bool:
    year_list = answers_by_year.get(year)
    if not year_list:
        return False
    for item in year_list:
        if item.get("variable") == variable and _is_na(item):
            item["answer"] = new_answer
            item["confidence"] = confidence
            item["source_fields"] = source_fields
            item["source_text"] = (source_text or "")[:800]
            item["source_heading"] = source_heading
            item["page_number"] = page_number if page_number is not None else item.get("page_number", -1)
            item["source_link"] = "RAG"
            return True
    return False


def rag_recovery(state: FinancialGraphState) -> dict:
    answers: List[Dict[str, Any]] = state.get("answers", [])
    answers_by_year: Dict[Any, List[Dict[str, Any]]] = state.get("answers_by_year", {})
    excel_data = state.get("excel_data", [])
    selected_pages = state.get("selected_pages", [])
    errors = list(state.get("errors", []))

    if not answers or not selected_pages:
        return {"errors": errors + ["rag_recovery skipped: missing answers or pages"]}

    metric_lookup = {(m.get("worksheet"), m.get("variable")): m for m in excel_data}

    na_items = [
        (year, a)
        for year, year_answers in answers_by_year.items()
        for a in year_answers
        if _is_na(a)
    ]
    if not na_items:
        logger.info("rag_recovery: nothing to recover")
        return {"use_rag": False}

    logger.info("rag_recovery: recovering %d N/A pairs from %d pages", len(na_items), len(selected_pages))

    settings = get_settings()
    try:
        embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=settings.rag_embedding_model,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", settings.azure_openai_api_version),
        )
        vectorstore = _build_inmemory_index(selected_pages, embedding_model)
        if vectorstore is None:
            return {"errors": errors + ["rag_recovery: failed to build index"]}
        retriever = vectorstore.as_retriever(search_kwargs={"k": 18})
    except Exception as exc:
        logger.exception("Index build failed")
        return {"errors": errors + [f"Failed to build index: {exc}"]}

    llm = create_llm(azure_deployment=settings.azure_openai_deployment, top_p=settings.llm_top_p)
    structured_llm = (
        llm.with_structured_output(MultiYearExtractionResponse)
        .with_retry(wait_exponential_jitter=True, stop_after_attempt=3)
    )

    BATCH_SIZE = 5
    recovered = 0
    start_time = time.time()
    years = state.get("years") or list(answers_by_year.keys())

    by_var = defaultdict(list)
    for year, item in na_items:
        by_var[item.get("variable")].append((year, item))

    var_list = list(by_var.keys())

    for batch_idx, i in enumerate(range(0, len(var_list), BATCH_SIZE)):
        batch_vars = var_list[i:i + BATCH_SIZE]
        user_metrics = []
        for var in batch_vars:
            sample = by_var[var][0][1]
            user_metrics.append({
                "worksheet": sample.get("worksheet"),
                "variable": var,
                "description": metric_lookup.get((sample.get("worksheet"), var), {}).get("description"),
            })

        user_prompt = get_user_prompt(user_metrics)
        year_instruction = (
            f"You MUST return one answer object for EVERY year in this list: {years}. "
            "Do not omit any year or any variable from the list below. "
            "If a value exists, include it. If it does not exist, still return the object with answer = 'N/A'. "
            "Include the year field on every answer. "
            "STRICT TEMPLATE ADHERENCE: return only the variables listed below; do not invent extras. "
            "INCOME STATEMENT PRIORITY: for every P&L metric you must search the face of the Income Statement / Statement of Comprehensive Income and related notes before returning N/A. "
            "Apply the mandatory mappings from the system prompt for both years: "
            "Depreciation → Amortization and depreciation, "
            "Other Labour Costs → Social securities and pension costs (or Total/Staff labour costs), "
            "General Costs → Other operating expenses, "
            "Interest Received - cash → Interest income, "
            "Income from Other Finan Assets - cash → Finance income, "
            "Salaries → Wages and salaries. "
            "Prefer face-of-statement totals. Report costs/expenses as positive numbers. "
            "Preserve thousand separators and use a leading minus for negatives."
        )

        query = " ".join(batch_vars)
        logger.info("rag_recovery: batch %d (%d metrics)", batch_idx + 1, len(batch_vars))

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
                        "Focus especially on completing every Income Statement variable. "
                        "Search the face of the Income Statement and related notes thoroughly. "
                        "Follow the system prompt exactly (template adherence, mandatory mappings, multi-year, signs, formatting).\n\n"
                        f"Extract only the following missing variables:\n{user_prompt}\n\n"
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
                ans = str(match.answer or "").strip()
                if ans.upper() in ("N/A", "NA", ""):
                    continue

                updated = _update_answer_for_year(
                    answers_by_year,
                    var,
                    year,
                    ans,
                    match.confidence or 0.70,
                    match.source_fields or [],
                    match.source_text or "",
                    getattr(match, "source_heading", None),
                    getattr(match, "page_number", None),
                )
                if updated:
                    recovered += 1
                    logger.info("  Recovered: %s (%s) → %s", var, year, ans[:80])

        except Exception as exc:
            logger.warning("Batch %d failed: %s", batch_idx + 1, exc)
            errors.append(f"rag_recovery batch {batch_idx + 1} error: {exc}")

    # Rebuild flat answers from the latest year
    latest = years[0] if years else None
    if latest is not None and latest in answers_by_year:
        answers = answers_by_year[latest]

    logger.info(
        "rag_recovery finished: recovered %d pairs in %.1fs (sentence-window, k=18, batch=%d)",
        recovered, time.time() - start_time, BATCH_SIZE,
    )

    return {
        "answers": answers,
        "answers_by_year": answers_by_year,
        "errors": errors,
        "use_rag": True,
        "rag_collection_name": "in-memory-faiss",
    }
```

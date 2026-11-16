```python
"""RAG recovery node for missing financial metrics.

Builds a transient in-memory FAISS index over the already-selected pages,
retrieves the most relevant chunks for each batch of N/A metrics, and
writes any recovered values back into the correct year buckets.

No external downloads and no persistent vector store are used.
"""

from typing import Any, Dict, List, Optional
import logging
import time
import os
from collections import defaultdict

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
    """Return True if the answer is missing or has zero confidence.

    Treats the common sentinel values (N/A, NA, empty string, NONE)
    and a confidence of 0.0 as "not available".

    Args:
        answer: A single metric answer dict that must contain at least
            the keys ``answer`` and ``confidence``.

    Returns:
        True when the value should be considered missing, False otherwise.
    """
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
    """Create a transient in-memory FAISS index from the selected pages.

    Pages are split into overlapping chunks; each chunk retains its
    original page number in metadata so later citations stay accurate.

    Args:
        selected_pages: List of page dicts. Each dict is expected to
            contain either a ``text`` or ``content`` field and optionally
            a ``page_number`` / ``page`` field.
        embedding_model: Configured Azure OpenAI embeddings client.

    Returns:
        A FAISS vector store ready for retrieval, or None if no usable
        text was found in the supplied pages.
    """
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

    logger.info("Building in-memory FAISS index from %d chunks", len(docs))
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
    """Write a recovered value into the matching year bucket.

    Only updates an entry when the variable matches and the current
    value is still considered N/A. The source link is forced to ``"RAG"``
    so downstream consumers can distinguish recovered values.

    Args:
        answers_by_year: Nested mapping of year → list of answer dicts.
        variable: Metric name to match.
        year: Fiscal year of the bucket that should be updated.
        new_answer: Recovered value (already stripped).
        confidence: Confidence score assigned by the LLM.
        source_fields: List of source field names (may be empty).
        source_text: Raw text snippet that supported the extraction.

    Returns:
        True if an N/A entry was found and updated, False otherwise.
    """
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
    """Recover missing (N/A) financial metrics via in-memory RAG.

    Workflow:
        1. Identify every N/A metric-year pair in ``answers_by_year``.
        2. Build a one-shot FAISS index from the already-selected pages.
        3. Group missing metrics by variable and process them in small
           batches so the LLM receives a focused context window.
        4. For each batch, retrieve the top-k chunks, ask the structured
           LLM for every requested year, and write successful recoveries
           back into the correct year buckets.
        5. Rebuild the flat ``answers`` list from the latest year so
           downstream nodes continue to see a consistent view.

    The function is intentionally side-effect free with respect to
    external services: the vector store lives only for the duration of
    the call and is discarded when the function returns.

    Args:
        state: Current graph state. Must contain at least ``answers``,
            ``answers_by_year``, ``excel_data`` and ``selected_pages``.

    Returns:
        Partial state update containing the (possibly enriched)
        ``answers`` and ``answers_by_year`` dictionaries, any errors
        encountered, and a flag indicating that RAG was used.
    """
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
        "rag_recovery: recovering %d N/A metric-year pairs from %d selected pages",
        len(na_items),
        len(selected_pages),
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
            "Depreciation → depreciation (both years), "
            "Amortization → amortization (both years), "
            "Other Labour Costs → Social securities and pension costs (both years), "
            "Interest Received - cash → Interest income (both years)."
        )

        # Boost Income Statement terms in the retrieval query
        income_statement_boost = (
            "Consolidated Statement of Comprehensive Income Income Statement "
            "Profit and Loss Non-interest revenue Wages and salaries "
            "Social securities and pension costs Other operating expenses "
            "Amortization and depreciation Net income"
        )
        query = " ".join(batch_vars) + " " + income_statement_boost

        logger.info(
            "rag_recovery: batch %d/%d (%d metrics)",
            batch_idx + 1,
            (len(var_list) + BATCH_SIZE - 1) // BATCH_SIZE,
            len(batch_vars),
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
                        "CRITICAL ADDITIONAL INSTRUCTIONS (must be followed together with the system prompt):\n"
                        "- Strictly search the face of the Consolidated Statement of Comprehensive Income / Income Statement first for every N/A metric.\n"
                        "- Deeply examine both the current-year and prior-year columns on the Income Statement before considering any note.\n"
                        "- For every P&L-related variable, prefer the face-of-statement line item. Only use a note if the face total is absent.\n"
                        "- Apply all mandatory mappings from the system prompt without exception (Depreciation, Amortization, Other Labour Costs, Interest Received - cash, etc.).\n"
                        "- Do not return N/A for an Income Statement metric if a clear equivalent line exists on the face of the statement.\n"
                        "- Return one answer object per variable per year. Preserve exact formatting and sign conventions from the system prompt.\n\n"
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
                            "  Recovered: %s (%s) → %s",
                            var,
                            year,
                            str(match.answer)[:60],
                        )

        except Exception as exc:
            logger.warning("Batch %d failed: %s", batch_idx + 1, exc)
            errors.append(f"rag_recovery batch {batch_idx + 1} error: {exc}")

    # Rebuild the flat answers list from the latest year (for downstream nodes)
    latest = years[0] if years else None
    if latest is not None and latest in answers_by_year:
        answers = answers_by_year[latest]

    elapsed = time.time() - start_time
    logger.info(
        "rag_recovery finished: recovered %d metric-year pairs in %.1fs "
        "(in-memory FAISS, batch size %d, k=10)",
        recovered,
        elapsed,
        BATCH_SIZE,
    )

    return {
        "answers": answers,
        "answers_by_year": answers_by_year,
        "errors": errors,
        "use_rag": True,
        "rag_collection_name": "in-memory-faiss",
    }
```

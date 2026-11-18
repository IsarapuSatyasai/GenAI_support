```python
"""Node: RAG recovery for N/A metrics.

Builds a temporary in-memory Chroma index from the pages already chosen
by the upstream selection step. For every metric that is still missing,
it retrieves the most relevant chunks and asks the model to fill the
value for each year present in the document. The index is discarded
when the node finishes.

Successfully recovered metric-year pairs are remembered for the life
of the process so later runs can skip them and concentrate on the rest.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
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

# Process-level cache of metric-year pairs that were already recovered
_RECOVERED_CACHE: Set[Tuple[str, Any]] = set()

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
    skipped = 0
    for year, year_answers in answers_by_year.items():
        for a in year_answers:
            if not _is_na(a):
                continue
            key = (a.get("variable"), year)
            if key in _RECOVERED_CACHE:
                skipped += 1
                continue
            na_items.append((year, a))

    if skipped:
        logger.info(f"rag_recovery: skipped {skipped} already recovered metric-year pairs")

    if not na_items:
        logger.info("rag_recovery: no remaining N/A metrics to recover")
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
                        _RECOVERED_CACHE.add((var, year))
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
        f"in {elapsed:.1f}s (in-memory Chroma, batch size {batch_size}, k={top_k}, "
        f"cached recovered: {len(_RECOVERED_CACHE)})"
    )

    return {
        "answers": answers,
        "answers_by_year": answers_by_year,
        "errors": errors,
        "use_rag": True,
        "rag_collection_name": "in-memory-chroma",
    }
```

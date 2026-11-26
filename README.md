```python
"""Node: RAG recovery for N/A metrics.

Builds a temporary in-memory Chroma index from the pages already chosen
by the upstream selection step. For every metric that is still missing,
it retrieves the most relevant chunks (optionally reranked) and asks the
model to fill the value for each year present in the document.
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
from sentence_transformers import CrossEncoder

from custom_lcel_components.llm import create_llm
from graph.graph_state import FinancialGraphState
from config import get_settings
from prompts.user_prompt import get_user_prompt
from prompts.rag_recovery_prompt import (
    RAG_RECOVERY_SYSTEM_PROMPT,
    build_rag_recovery_user_prompt,
)
from models import MultiYearExtractionResponse

logger = logging.getLogger(__name__)

INCOME_STATEMENT_SHEET = "Income_Statement"

SYNONYM_MAP = {
    "Depreciation": "depreciation amortization and depreciation depreciation expense",
    "Amortization": "amortization amortisation intangible assets amortization expense",
    "Other Labour Costs": "other labour costs social securities and pension costs staff costs employee benefit expense",
    "Income from Other Finan Assets - cash": "finance income income from other financial assets interest income",
    "Income from Other Finan Assets- non-cash": "finance income other financial results",
    "Interest Received - cash": "interest income interest received finance income",
    "Interest Received - non-cash": "interest income non-cash",
    "Interest Paid - cash": "interest expense interest paid finance costs",
    "Interest Paid - non-cash": "interest expense non-cash",
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
    chunk_size: int,
    chunk_overlap: int,
    page_char_limit: int,
) -> Optional[Chroma]:
    docs = []
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

        if len(text) <= page_char_limit:
            docs.append(Document(page_content=text, metadata=meta))
        else:
            splits = recursive_splitter.create_documents([text], metadatas=[meta])
            docs.extend(splits)

    if not docs:
        logger.warning("rag_recovery: no text found in selected_pages")
        return None

    logger.info(f"Building in-memory Chroma index from {len(docs)} chunks")
    return Chroma.from_documents(docs, embedding_model)


def _rerank(docs: List[Document], query: str, model_name: str, top_n: int) -> List[Document]:
    if not docs:
        return docs
    try:
        model = CrossEncoder(model_name)
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_n]]
    except Exception as exc:
        logger.warning(f"Reranking failed, falling back to original order: {exc}")
        return docs[:top_n]


def _update_answer_for_year(
    answers_by_year: Dict[Any, List[Dict[str, Any]]],
    variable: str,
    year: Any,
    new_answer: str,
    confidence: float,
    source_fields: list,
    source_text: str,
    page_number: Any = -1,
    rag_chunk: str = "",
) -> bool:
    year_list = answers_by_year.get(year)
    if not year_list:
        return False
    for item in year_list:
        if item.get("variable") == variable and _is_na(item):
            item["answer"] = new_answer
            item["confidence"] = confidence
            item["source_fields"] = source_fields
            item["source_text"] = (source_text or "")[:500]
            item["source_link"] = "RAG"
            item["page_number"] = page_number
            item["rag_extracted"] = "Yes"
            item["rag_chunk"] = (rag_chunk or "")[:2000]
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
            if _is_na(a) and a.get("worksheet") == INCOME_STATEMENT_SHEET:
                na_items.append((year, a))

    if not na_items:
        logger.info("rag_recovery: no N/A Income Statement metrics to recover")
        return {"use_rag": False}

    logger.info(
        f"rag_recovery: recovering {len(na_items)} N/A Income Statement metric-year pairs "
        f"from {len(selected_pages)} selected pages"
    )

    settings = get_settings()
    chunk_size = settings.rag_chunk_size
    chunk_overlap = settings.rag_chunk_overlap
    top_k = settings.rag_similarity_top_k
    page_char_limit = getattr(settings, "rag_page_char_limit", 2500)
    do_rerank = getattr(settings, "rag_rerank_enabled", True)
    rerank_top_n = settings.rag_rerank_top_n
    rerank_model = settings.rag_rerank_model

    try:
        embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=settings.rag_embedding_model,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", settings.azure_openai_api_version
            ),
        )
        vectorstore = _build_inmemory_index(
            selected_pages, embedding_model, chunk_size, chunk_overlap, page_char_limit
        )
        if vectorstore is None:
            return {"errors": errors + ["rag_recovery: failed to build in-memory index"]}

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    except Exception as exc:
        logger.exception("Failed to build in-memory index")
        return {"errors": errors + [f"Failed to build index: {exc}"]}

    llm = create_llm(
        azure_deployment=settings.rag_llm_model,
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

    for var, year_items in by_var.items():
        sample = year_items[0][1]
        desc = metric_lookup.get(
            (sample.get("worksheet"), var), {}
        ).get("description") or ""
        remarks = metric_lookup.get(
            (sample.get("worksheet"), var), {}
        ).get("remarks") or ""

        query_parts = [var]
        if var in SYNONYM_MAP:
            query_parts.append(SYNONYM_MAP[var])
        else:
            if desc:
                query_parts.append(desc)
            if remarks:
                query_parts.append(remarks)
        query = " ".join(query_parts)

        user_metrics = [{
            "worksheet": sample.get("worksheet"),
            "variable": var,
            "description": desc,
        }]
        user_prompt = get_user_prompt(user_metrics)

        logger.info(f"rag_recovery: retrieving for metric '{var}'")

        try:
            docs = retriever.invoke(query)
            if do_rerank:
                docs = _rerank(docs, query, rerank_model, rerank_top_n)

            context = "\n\n".join(
                f"[Page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
                for d in docs
            )

            messages = [
                SystemMessage(content=RAG_RECOVERY_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_rag_recovery_user_prompt(user_prompt, years, context)
                ),
            ]

            response: MultiYearExtractionResponse = structured_llm.invoke(messages)

            for match in response.answers:
                match_var = match.variable
                year = getattr(match, "year", None)
                if not match_var or year is None:
                    continue
                if (
                    match.answer
                    and str(match.answer).strip().upper() not in ("N/A", "NA", "")
                ):
                    updated = _update_answer_for_year(
                        answers_by_year,
                        match_var,
                        year,
                        str(match.answer).strip(),
                        match.confidence or 0.65,
                        match.source_fields or [],
                        match.source_text or "",
                        getattr(match, "page_number", -1),
                        rag_chunk=context,
                    )
                    if updated:
                        recovered += 1
                        logger.info(
                            f"  Recovered: {match_var} ({year}) → {str(match.answer)[:60]}"
                        )

        except Exception as exc:
            logger.warning(f"Metric '{var}' failed: {exc}")
            errors.append(f"rag_recovery metric '{var}' error: {exc}")

    latest = years[0] if years else None
    if latest is not None and latest in answers_by_year:
        answers = answers_by_year[latest]

    elapsed = time.time() - start_time
    logger.info(
        f"rag_recovery finished: recovered {recovered} metric-year pairs "
        f"in {elapsed:.1f}s (Income Statement only, per-metric, hybrid query, k={top_k}, rerank={do_rerank})"
    )

    return {
        "answers": answers,
        "answers_by_year": answers_by_year,
        "errors": errors,
        "use_rag": True,
        "rag_collection_name": "in-memory-chroma",
    }
```

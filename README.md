```python
"""Node: RAG recovery for N/A metrics (batched, fixed-size chunk RAG).

Forces a fresh unique collection on every run by appending a random
suffix to the source path (so derive_collection_name_from_source
produces a new name). Uses the default fixed-size chunk RAG chain.
source_fields are real PDF labels returned by the model (or empty list).
"""

from typing import Any, Dict, List
import logging
import json
import re
import time
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from custom_lcel_components.chains import get_default_rag_chain_from_config
from custom_lcel_components.indexing import derive_collection_name_from_source
from custom_lcel_components.llm import create_llm
from graph.graph_state import FinancialGraphState
from config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial data extraction assistant.
Given retrieved passages from an annual report, extract the exact numeric
value (or exactly "N/A" if truly absent) for each requested metric.
Return ONLY a valid JSON object.
Each key is the exact variable name.
Each value is an object with:
  - "answer": the extracted value as a string (keep thousand separators)
  - "source_fields": list of original PDF line-item names used (empty list if N/A or unknown)
Example:
{{"Net Sales": {{"answer": "1,234,567", "source_fields": ["Revenue"]}}, "Goodwill": {{"answer": "N/A", "source_fields": []}}}}
Do not invent numbers. Do not add any text, markdown, or explanation outside the JSON."""

def _is_na(answer: Dict[str, Any]) -> bool:
    val = str(answer.get("answer", "")).strip().upper()
    try:
        conf = float(answer.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        conf = 0.0
    return val in ("N/A", "NA", "", "NONE") or conf == 0.0


def _clean_json(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return raw.strip()


def rag_recovery(state: FinancialGraphState) -> dict:
    answers: List[Dict[str, Any]] = state.get("answers", [])
    answers_by_year: Dict[Any, List[Dict[str, Any]]] = state.get("answers_by_year", {})
    excel_data = state.get("excel_data", [])
    pdf_source = state.get("pdf_source")
    errors = list(state.get("errors", []))

    if not pdf_source or not answers:
        return {"errors": errors + ["rag_recovery skipped: missing pdf_source or answers"]}

    metric_lookup = {
        (m.get("worksheet"), m.get("variable")): m
        for m in excel_data
    }
    na_items = [a for a in answers if _is_na(a)]

    if not na_items:
        logger.info("rag_recovery: no N/A metrics to recover")
        return {"use_rag": False}

    logger.info(f"rag_recovery: recovering {len(na_items)} N/A metrics (fixed-size chunk RAG, batch size 3)")
    logger.info(f"rag_recovery: original pdf_source = {pdf_source}")

    # Force a unique source so the derived collection name is always new
    unique_suffix = uuid.uuid4().hex[:8]
    unique_source = f"{pdf_source}#{unique_suffix}"
    logger.info(f"rag_recovery: unique source = {unique_source}")

    # Log the name that will actually be used
    try:
        collection_name = derive_collection_name_from_source(
            {"source": unique_source}, source_key="source"
        )
        logger.info(f"rag_recovery: fresh collection name will be '{collection_name}'")
    except Exception:
        collection_name = None

    settings = get_settings()

    rag_config = {
        "llm_model": settings.azure_openai_deployment,
        "embedding_model_name": "text-embedding-3-large",
        "pdf_loader_type": "PyMuPDFLoader",
        "include_source": False,
        "hybrid_search_args": None,
        "remove_duplicate_chunks": True,
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "similarity_top_k": 20,
        "source_key": "source",
        "rerank_args": {"top_n": 10, "model": "cross-encoder/ms-marco-MiniLM-L-2-v2"},
        "system_message": SYSTEM_PROMPT,
        "top_p": settings.llm_top_p,
    }

    try:
        rag_chain, index_chain, _ = get_default_rag_chain_from_config(rag_config)

        llm = create_llm(
            azure_deployment=settings.azure_openai_deployment,
            top_p=settings.llm_top_p,
        ).with_retry(wait_exponential_jitter=True, stop_after_attempt=3)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Metrics to extract:\n{metrics_list}\n\nContext:\n{context}"),
        ])

        recovery_chain = (
            RunnablePassthrough.assign(collection_name=index_chain)
            | rag_chain
            | prompt
            | llm
            | StrOutputParser()
        )
    except Exception as exc:
        logger.exception("Failed to build RAG recovery chain")
        return {"errors": errors + [f"Failed to build RAG recovery chain: {exc}"]}

    BATCH_SIZE = 3
    recovered = 0
    start_time = time.time()

    for batch_idx, i in enumerate(range(0, len(na_items), BATCH_SIZE)):
        batch = na_items[i : i + BATCH_SIZE]
        metrics_list = "\n".join(
            f"- {a.get('variable', '')}: "
            f"{metric_lookup.get((a.get('worksheet'), a.get('variable')), {}).get('description') or a.get('variable', '')}"
            for a in batch
        )

        logger.info(
            f"rag_recovery: batch {batch_idx + 1}/{(len(na_items) + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"({len(batch)} metrics)"
        )

        try:
            # Pass the unique source so the index step derives a new collection
            raw = recovery_chain.invoke({
                "source": unique_source,
                "question": "Extract the values for the listed metrics.",
                "metrics_list": metrics_list,
            })

            cleaned = _clean_json(raw or "")
            try:
                parsed = json.loads(cleaned) if cleaned else {}
            except json.JSONDecodeError:
                logger.warning(f"Batch {batch_idx + 1} JSON parse failed. Raw: {cleaned[:300]}")
                parsed = {}

            for ans in batch:
                var = ans.get("variable", "")
                parsed_item = parsed.get(var)
                if parsed_item is None:
                    for k, v in parsed.items():
                        if str(k).strip().lower() == var.lower():
                            parsed_item = v
                            break

                if isinstance(parsed_item, dict):
                    value = parsed_item.get("answer")
                    source_fields = parsed_item.get("source_fields") or []
                elif parsed_item is not None:
                    value = parsed_item
                    source_fields = []
                else:
                    value = None
                    source_fields = []

                if value is not None and str(value).strip().upper() not in ("N/A", "NA", "", "NONE"):
                    ans["answer"] = str(value).strip()
                    ans["confidence"] = 0.65
                    ans["source_fields"] = source_fields if isinstance(source_fields, list) else []
                    ans["source_text"] = str(value)[:500]
                    ans["source_link"] = "RAG"
                    recovered += 1
                    logger.info(f"  Recovered: {var} → {str(value)[:60]} | sources: {source_fields}")

        except Exception as exc:
            logger.warning(f"Batch {batch_idx + 1} failed: {exc}")
            errors.append(f"rag_recovery batch {batch_idx + 1} error: {exc}")

    # Sync answers_by_year
    years = state.get("years") or []
    latest = years[0] if years else None
    if latest is not None and latest in answers_by_year:
        answers_by_year[latest] = answers
    elif answers_by_year:
        answers_by_year[next(iter(answers_by_year))] = answers

    elapsed = time.time() - start_time
    logger.info(
        f"rag_recovery finished: recovered {recovered}/{len(na_items)} "
        f"in {elapsed:.1f}s (fixed-size chunk RAG, batch size {BATCH_SIZE})"
    )

    return {
        "answers": answers,
        "answers_by_year": answers_by_year,
        "errors": errors,
        "rag_collection_name": collection_name,
        "use_rag": True,
    }
```

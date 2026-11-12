```python
"""Node: RAG recovery for N/A metrics.

Builds an in-memory FAISS index from the already-selected pages.
Retrieves top-k chunks per batch. No download, no persistent store.
"""

from typing import Any, Dict, List, Optional
import logging
import json
import re
import time
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from prompts.system_prompt import SYSTEM_PROMPT
from prompts.user_prompt import get_user_prompt
from models import MultiYearExtractionResponse
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
{"Net Sales": {"answer": "1,234,567", "source_fields": ["Revenue"]}, "Goodwill": {"answer": "N/A", "source_fields": []}}
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
    )
    docs = splitter.create_documents(texts, metadatas=metadatas)
    if not docs:
        logger.warning("rag_recovery: no chunks created from selected_pages")
        return None

    logger.info(f"Building in-memory FAISS index from {len(docs)} chunks")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore

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
        (m.get("worksheet"), m.get("variable")): m
        for m in excel_data
    }
    na_items = [a for a in answers if _is_na(a)]

    if not na_items:
        logger.info("rag_recovery: no N/A metrics to recover")
        return {"use_rag": False}

    logger.info(
        f"rag_recovery: recovering {len(na_items)} N/A metrics "
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

        # Optimal k for financial line-item recovery
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    except Exception as exc:
        logger.exception("Failed to build in-memory index")
        return {"errors": errors + [f"Failed to build index: {exc}"]}

    llm = create_llm(
        azure_deployment=settings.azure_openai_deployment,
        top_p=settings.llm_top_p,
    ).with_retry(wait_exponential_jitter=True, stop_after_attempt=3)

    structured_llm = llm.with_structured_output(MultiYearExtractionResponse)

    # Optimal batch size for ~166 N/A items (≈28 LLM calls)
    BATCH_SIZE = 6
    recovered = 0
    start_time = time.time()
    years = state.get("years") or []

    for batch_idx, i in enumerate(range(0, len(na_items), BATCH_SIZE)):
        batch = na_items[i : i + BATCH_SIZE]

        user_metrics = [
            {
                "worksheet": a.get("worksheet"),
                "variable": a.get("variable"),
                "description": metric_lookup.get(
                    (a.get("worksheet"), a.get("variable")), {}
                ).get("description"),
            }
            for a in batch
        ]
        user_prompt = get_user_prompt(user_metrics)

        year_instruction = (
            f"Extract every variable only for these years: {years}. "
            "Return one answer object per variable per year. "
            "Include the year field on every answer. "
            "Prefer the primary consolidated statements over notes. "
            "Apply the sign convention of the system prompt."
            if years
            else "Extract the most recent period available. "
                 "Apply the sign convention of the system prompt."
        )

        query = " ".join(a.get("variable", "") for a in batch)

        logger.info(
            f"rag_recovery: batch {batch_idx + 1}/"
            f"{(len(na_items) + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"({len(batch)} metrics)"
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

            for ans in batch:
                var = ans.get("variable", "")
                match = next(
                    (a for a in response.answers if a.variable == var),
                    None,
                )
                if (
                    match
                    and match.answer
                    and str(match.answer).strip().upper() not in ("N/A", "NA", "")
                ):
                    ans["answer"] = str(match.answer).strip()
                    ans["confidence"] = match.confidence or 0.65
                    ans["source_fields"] = match.source_fields or []
                    ans["source_text"] = (match.source_text or "")[:500]
                    ans["source_link"] = "RAG"
                    recovered += 1
                    logger.info(
                        f"  Recovered: {var} → {str(match.answer)[:60]}"
                    )

        except Exception as exc:
            logger.warning(f"Batch {batch_idx + 1} failed: {exc}")
            errors.append(f"rag_recovery batch {batch_idx + 1} error: {exc}")

    # Sync answers_by_year
    latest = years[0] if years else None
    if latest is not None and latest in answers_by_year:
        answers_by_year[latest] = answers
    elif answers_by_year:
        answers_by_year[next(iter(answers_by_year))] = answers

    elapsed = time.time() - start_time
    logger.info(
        f"rag_recovery finished: recovered {recovered}/{len(na_items)} "
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

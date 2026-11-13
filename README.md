```python
"""Node: RAG recovery for N/A metrics.

Builds an in-memory FAISS index from the pages already selected by the
upstream node and attempts to fill any remaining missing values. The
index is created fresh on each run and discarded when the node finishes.
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
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

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
    docs = []
    page_limit = 4500

    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=250,
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
            continue

        try:
            splits = semantic_splitter.create_documents([text], metadatas=[meta])
            if splits and all(len(s.page_content) < 6000 for s in splits):
                docs.extend(splits)
                continue
        except Exception:
            logger.warning(f"Semantic split failed for page {page_no}, falling back")

        splits = recursive_splitter.create_documents([text], metadatas=[meta])
        docs.extend(splits)

    if not docs:
        logger.warning("rag_recovery: no text found in selected_pages")
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

def _apply_mandatory_mappings(
    answers_by_year: Dict[Any, List[Dict[str, Any]]],
    excel_data: List[Dict[str, Any]],
) -> int:
    template_map = {}
    for m in excel_data:
        var = m.get("variable")
        if not var:
            continue
        desc = (m.get("description") or "") + " " + (m.get("remarks") or "")
        template_map[var] = desc.lower()

    source_phrases = {
        "Depreciation": ["amortization and depreciation", "depreciation and amortisation"],
        "Other Labour Costs": ["social securities and pension costs", "total labour costs", "staff costs"],
        "Interest Received - cash": ["interest income", "interest received"],
        "General Costs": ["other operating expenses", "other expenses", "administrative expenses"],
        "Income from Other Finan Assets - cash": ["finance income", "finance income received"],
        "3rd Party Svc's (Sub-Contr Work)": ["costs incurred from financial institutions"],
        "Other Income": ["other income/(expenses)", "other income"],
        "Accounts Receivable Trade": ["trade and other receivables"],
        "Account Payable Trade": ["trade and other payables"],
        "Deferred Taxes Asset": ["deferred tax assets"],
    }

    updated = 0
    for year, year_answers in answers_by_year.items():
        source_pool = " ".join(
            (a.get("source_text") or "") + " " + (a.get("source_heading") or "")
            for a in year_answers
        ).lower()

        for target_var, phrases in source_phrases.items():
            if target_var not in template_map:
                continue
            for item in year_answers:
                if item.get("variable") != target_var:
                    continue
                if not _is_na(item):
                    continue
                if any(p in source_pool for p in phrases):
                    for sibling in year_answers:
                        sib_text = (sibling.get("source_text") or "").lower()
                        if any(p in sib_text for p in phrases) and not _is_na(sibling):
                            item["answer"] = sibling["answer"]
                            item["confidence"] = 0.7
                            item["source_fields"] = sibling.get("source_fields", [])
                            item["source_text"] = sibling.get("source_text", "")[:500]
                            item["source_link"] = "RULE"
                            updated += 1
                            logger.info(f"  Rule-mapped: {target_var} ({year}) → {sibling['answer']}")
                            break
    return updated

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
            "Do not omit any year. If a value exists for a prior year, include it. "
            "If a value does not exist for a year, still return an object for that year with answer = 'N/A'. "
            "Include the year field on every answer. "
            "Apply the mandatory mappings from the system prompt."
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

    rule_updated = _apply_mandatory_mappings(answers_by_year, excel_data)
    logger.info(f"rag_recovery: rule-based mappings applied: {rule_updated}")

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

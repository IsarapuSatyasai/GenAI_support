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
from sentence_transformers import CrossEncoder   # already used in Credit RAG

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

    metric_lookup = {(m.get("worksheet"), m.get("variable")): m for m in excel_data}

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
    chunk_size = settings.rag_chunk_size
    chunk_overlap = settings.rag_chunk_overlap
    top_k = settings.rag_similarity_top_k
    batch_size = settings.rag_batch_size
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
            desc = metric_lookup.get((sample.get("worksheet"), var), {}).get("description") or ""
            remarks = metric_lookup.get((sample.get("worksheet"), var), {}).get("remarks") or ""
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
        query = " ".join(query_parts)

        logger.info(
            f"rag_recovery: batch {batch_idx + 1}/"
            f"{(len(var_list) + batch_size - 1) // batch_size} "
            f"({len(batch_vars)} metrics)"
        )

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
        f"in {elapsed:.1f}s (in-memory Chroma, batch size {batch_size}, "
        f"k={top_k}, rerank={do_rerank})"
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
"""Specialised prompts for the RAG recovery node."""

RAG_RECOVERY_SYSTEM_PROMPT = """\
You are a senior credit analyst performing financial spreading recovery.

Your only job is to extract the missing (N/A) metrics supplied in the user message
from the provided document context. The context consists of the most relevant
chunks already retrieved from the selected pages of the annual report.

Critical rules:
- Search for synonyms, near-synonyms and common accounting equivalents before returning N/A.
- Prefer the face of the primary consolidated statements over note breakdowns.
- Extract every requested year. Do not drop prior-year columns.
- Return numeric values exactly as they appear (including thousand separators). Apply the sign convention only.
- Provide the page number shown in the [Page N] markers.
- If a clear equivalent line exists, extract it. Return N/A only when nothing can be found after a thorough check.
- Do not invent numbers.

Output must strictly follow the MultiYearExtractionResponse schema.
"""

def build_rag_recovery_user_prompt(
    user_prompt: str,
    years: list,
    context: str,
) -> str:
    year_instruction = (
        f"You MUST return one answer object for EVERY year in this list: {years}. "
        "Strictly follow the template instructions, remarks, and descriptions. "
        "Search for synonyms and accounting equivalents of each variable before returning N/A. "
        "If an equivalent line exists on the primary statements or notes, extract it for every year shown. "
        "Include the page number where the value was found. "
        "Do not omit any year."
    )
    return (
        f"Please extract only the following missing variables:\n"
        f"{user_prompt}\n\n"
        f"{year_instruction}\n\n"
        f"Document context:\n{context}"
    )
```

```python
"""
Configration of paths and setup of environment variables for financial spreading project.
Import this to main notebook to setup the openai and storage account.
"""

import os
from databricks.sdk.runtime import dbutils, spark

class Settings:
    # os.environ["AZURE_OPENAI_API_KEY"] = dbutils.secrets.get(
    #     scope="azure-key-vault-scope", key="openai-apim-api-key"
    #     )
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://msc-apim-gailtu-prd.azure-api.net/openai-api"

    # access to storage account has changed to this method os.environ["STORAGE_ACCOUNT"] = "mscsta{ci_code}{env}".format(ci_code=os.environ["CI_CODE"], env=os.environ["ENVIRONMENT"])
    os.environ["STORAGE_ACCOUNT"] = "mscstagailtuprd03"
    
    os.environ["ENVIRONMENT"] = "prd"

    STORAGE_ACCOUNT_NAME = "mscstagailtuprd03"
    FOLDER_NAME = "projects"
    TENANT_ID = "07751f8b-9617-49d5-9024-85f034f400ec"
    DBR_MI = "fa515616-7575-408d-8196-320484b017c7"

    storage_host = f"{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net"


    spark.conf.set(f"fs.azure.account.auth.type.{storage_host}", "OAuth")
    spark.conf.set(
        f"fs.azure.account.oauth.provider.type.{storage_host}",
        "org.apache.hadoop.fs.azurebfs.oauth2.MsiTokenProvider",
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.msi.tenant.{storage_host}",
        TENANT_ID,
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.id.{storage_host}",
        DBR_MI,
    )
    # See below an example how to access the storage account directly
    subfolder = 'FinancialSpreading'
    # # List files directly — no mount required
    path = f"abfss://{FOLDER_NAME}@{storage_host}/{subfolder}"
    # dbutils.fs.ls(path)


    # --- Azure OpenAI ---
    openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-5.6-luna"
    azure_openai_api_version: str = "2024-10-21"

    # --- LLM parameters ---
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_seed: int = 1234

    # --- RAG settings ---
    rag_embedding_model: str = "text-embedding-3-large"
    rag_sentence_window_size: int = 5
    rag_similarity_top_k: int = 20
    rag_rerank_top_n: int = 10
    rag_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    rag_rerank_enabled: bool = True
    rag_source_key: str = "pdf_source"
    rag_pdf_loader_type: str = "PyMuPDFLoader"
    rag_remove_duplicate_chunks: bool = True
    rag_chunk_size: int = 1500
    rag_chunk_overlap: int = 150
    rag_batch_size: int = 3
    rag_page_char_limit: int = 2500

    # --- Page selection ---
    first_n_pages = 5
    page_score_threshold = 1
    min_pages = 30
    max_selected_pages = 10
    max_adjacent_pages = 2
    refinement_attempts = 0

    # first_n_pages = 75
    # page_score_threshold = 1
    # min_pages = 30
    # max_selected_pages = 10
    # max_adjacent_pages = 2
    # refinement_attempts = 0


    # --- Sharepoint ---
    sharepoint_link = "https://connect2dll.sharepoint.com/:b:/r/sites/Grp-FinancialSpreading/Shared%20Documents/General/5.%20Input%20data"

    paths = [
        {
            'name': 'input_path',
            'type': 'Azure blob storage',
            'reference': '/mnt/projects/FinancialSpreading/InputData',
            'comment': 'Input location for storing annual reports for each country'
        },
        {
            'name': 'output_path',
            'type': 'Azure blob storage',
            # 'reference': '/mnt/projects/FinancialSpreading/OutputData', #Old way of writing
            'reference': f"abfss://{FOLDER_NAME}@{storage_host}/{subfolder}/OutputData",
            'comment': 'Output location for storing annual reports for each country'
        },   
    ]
                                                         
def get_settings() -> Settings:
    return Settings()
```

```python
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
                        rag_chunk=context,
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

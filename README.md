### Refinement changes

Modify `nodes/confidence_refinement.py` so refinement reviews existing answers instead of creating a new schema:

```python
if confidence > 0:
    candidate = _refine_answer(
        llm,
        answer,
        pdf_context,
    )

    # Preserve the original template schema
    candidate["worksheet"] = answer["worksheet"]
    candidate["variable"] = answer["variable"]
```

Update the refinement prompt:

```python
REFINEMENT_PROMPT = """
You are refining an existing financial extraction.

The Excel template schema is fixed.

Do NOT create new worksheets.
Do NOT create new variables.
Do NOT rename worksheets or variables.
Do NOT extract unrelated metrics.

Review the existing answer using the provided PDF evidence.

Check:
- existing answer
- page number
- financial year
- unit or scale
- source fields
- formula when applicable

If the existing answer is correct, keep it unchanged.
If the PDF supports a better answer, return the improved answer.

The worksheet and variable must remain exactly unchanged.
"""
```
**Commit Message:**

```text
"fix: preserve template schema during refinement"
```

**Commit description:**

```text
Update the refinement workflow to improve existing LLM answers
without changing the Excel template schema.

- Preserve worksheet and variable names
- Review answers with confidence > 0
- Validate answer and page against PDF evidence
- Refine the current answer iteratively
- Prevent creation of new worksheets or variables
- Keep final Excel aligned with the original template
```

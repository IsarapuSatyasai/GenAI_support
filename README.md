```text
test/financial-spreading-refinement
```

```text
Add original vs refined comparison sheet
```

```python
Added a Comparison worksheet to the financial spreading Excel output.

The new sheet compares the original extracted answers with the final
answers after refinement and verification. It includes answer,
confidence, page number, source fields, formula, source link, and a
changed indicator.

All metrics from the Excel template are retained in the comparison,
including metrics with N/A or missing values. Existing output
worksheets remain unchanged.
```

```python
"""Create template-aligned Excel output."""

import os

import pandas as pd

from graph.graph_state import FinancialGraphState
from template.excel_writer import excel_write


def align_answers_to_template(
    excel_data,
    answers,
):
    """Ensure final output contains exactly template metrics."""

    answer_map = {
        (
            answer["worksheet"],
            answer["variable"],
        ): answer
        for answer in answers
    }

    aligned = []

    for metric in excel_data:
        key = (
            metric["worksheet"],
            metric["variable"],
        )

        answer = answer_map.get(key, {})

        aligned.append(
            {
                "worksheet": metric["worksheet"],
                "variable": metric["variable"],
                "answer": answer.get(
                    "answer",
                    "N/A",
                ),
                "confidence": answer.get(
                    "confidence",
                    0.0,
                ),
                "page_number": answer.get(
                    "page_number",
                    -1,
                ),
                "source_fields": answer.get(
                    "source_fields",
                    [],
                ),
                "formula": answer.get(
                    "formula",
                ),
                "source_link": answer.get(
                    "source_link",
                    "N/A",
                ),
            }
        )

    return aligned


def build_comparison_dataframe(
    excel_data,
    original_answers,
    final_answers,
):
    """Create original and refined answer comparison."""

    original_map = {
        (
            answer["worksheet"],
            answer["variable"],
        ): answer
        for answer in original_answers
    }

    final_map = {
        (
            answer["worksheet"],
            answer["variable"],
        ): answer
        for answer in final_answers
    }

    comparison = []

    for metric in excel_data:
        key = (
            metric["worksheet"],
            metric["variable"],
        )

        original = original_map.get(
            key,
            {},
        )

        final = final_map.get(
            key,
            {},
        )

        original_answer = original.get(
            "answer",
            "N/A",
        )

        refined_answer = final.get(
            "answer",
            original_answer,
        )

        original_confidence = original.get(
            "confidence",
            0.0,
        )

        refined_confidence = final.get(
            "confidence",
            original_confidence,
        )

        original_page = original.get(
            "page_number",
            -1,
        )

        refined_page = final.get(
            "page_number",
            original_page,
        )

        original_source_fields = original.get(
            "source_fields",
            [],
        )

        refined_source_fields = final.get(
            "source_fields",
            original_source_fields,
        )

        original_formula = original.get(
            "formula",
        )

        refined_formula = final.get(
            "formula",
            original_formula,
        )

        original_source_link = original.get(
            "source_link",
            "N/A",
        )

        refined_source_link = final.get(
            "source_link",
            original_source_link,
        )

        changed = (
            original_answer != refined_answer
            or original_confidence != refined_confidence
            or original_page != refined_page
            or original_source_fields != refined_source_fields
            or original_formula != refined_formula
            or original_source_link != refined_source_link
        )

        comparison.append(
            {
                "worksheet": metric["worksheet"],
                "variable": metric["variable"],
                "answer_original": original_answer,
                "answer_refined": refined_answer,
                "confidence_original": original_confidence,
                "confidence_refined": refined_confidence,
                "page_number_original": original_page,
                "page_number_refined": refined_page,
                "source_fields_original": original_source_fields,
                "source_fields_refined": refined_source_fields,
                "formula_original": original_formula,
                "formula_refined": refined_formula,
                "source_link_original": original_source_link,
                "source_link_refined": refined_source_link,
                "changed": changed,
            }
        )

    return pd.DataFrame(comparison)


def create_excel_output(
    state: FinancialGraphState,
):
    """Create Excel output using template-defined metrics."""

    excel_data = state.get(
        "excel_data",
        [],
    )

    original_answers = state.get(
        "original_answers",
        [],
    )

    answers = state.get(
        "answers",
        [],
    )

    output_path = state.get(
        "output_path",
        "",
    )

    pdf_file_name = state.get(
        "pdf_file_name",
        "",
    )

    if not excel_data:
        return {
            "output_excel": "",
            "errors": state.get(
                "errors",
                [],
            ) + ["No template metrics available for Excel output."],
        }

    if not answers:
        answers = []

    output_name = (
        pdf_file_name.removeprefix(
            "InputData"
        ).removesuffix(
            ".pdf"
        )
        + ".xlsx"
    )

    output_excel = (
        output_path.rstrip("/")
        + "/"
        + output_name
    )

    comparison_name = (
        pdf_file_name.removeprefix(
            "InputData"
        ).removesuffix(
            ".pdf"
        )
        + "-comparison.xlsx"
    )

    comparison_excel = (
        output_path.rstrip("/")
        + "/"
        + comparison_name
    )

    aligned_answers = align_answers_to_template(
        excel_data,
        answers,
    )

    worksheet_dict = {}

    for answer in aligned_answers:
        worksheet = answer["worksheet"]

        row = {
            key: value
            for key, value in answer.items()
            if key != "worksheet"
        }

        worksheet_dict.setdefault(
            worksheet,
            [],
        ).append(row)

    worksheet_dict = {
        worksheet: pd.DataFrame(rows)
        for worksheet, rows in worksheet_dict.items()
    }

    comparison_df = build_comparison_dataframe(
        excel_data,
        original_answers,
        answers,
    )

    comparison_dict = {
        "Comparison": comparison_df,
    }

    try:
        print(
            f"Creating Excel output: {output_excel}"
        )

        print(
            f"Creating comparison output: {comparison_excel}"
        )

        print(
            f"Template metrics: {len(excel_data)}"
        )

        print(
            f"Final answers: {len(aligned_answers)}"
        )

        print(
            f"Worksheets: {list(worksheet_dict.keys())}"
        )

        status = excel_write(
            dataframes=worksheet_dict,
            output_path=output_excel,
        )

        comparison_status = excel_write(
            dataframes=comparison_dict,
            output_path=comparison_excel,
        )

        return {
            "output_excel": status,
            "comparison_excel": comparison_status,
            "errors": state.get(
                "errors",
                [],
            ),
        }

    except Exception as exc:
        error = f"Error writing excel file: {exc}"

        print(error)

        return {
            "output_excel": "",
            "comparison_excel": "",
            "errors": state.get(
                "errors",
                [],
            ) + [error],
        }

```

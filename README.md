```python
import os

import pandas as pd

from excel_writer import excel_write


def align_answers_to_template(excel_data, answers):
    """
    Align extracted answers with the Excel template structure.
    """
    answer_map = {
        (
            answer.get("worksheet"),
            answer.get("variable"),
        ): answer
        for answer in answers
    }

    aligned_answers = []

    for row in excel_data:
        worksheet = row.get("worksheet")
        variable = row.get("variable")

        answer = answer_map.get(
            (worksheet, variable),
            {},
        )

        aligned_answers.append(
            {
                **row,
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

    return aligned_answers


def build_comparison_dataframe(
    excel_data,
    original_answers,
    final_answers,
):
    """
    Create a comparison between the original extraction
    and the final answer selected after refinement.
    
    All metrics from the Excel template are retained,
    including metrics with N/A or missing answers.
    """
    original_map = {
        (
            answer.get("worksheet"),
            answer.get("variable"),
        ): answer
        for answer in original_answers
    }

    final_map = {
        (
            answer.get("worksheet"),
            answer.get("variable"),
        ): answer
        for answer in final_answers
    }

    rows = []

    for template_row in excel_data:
        worksheet = template_row.get("worksheet")
        variable = template_row.get("variable")

        original = original_map.get(
            (worksheet, variable),
            {},
        )

        final = final_map.get(
            (worksheet, variable),
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
            or original_source_fields
            != refined_source_fields
            or original_formula != refined_formula
            or original_source_link
            != refined_source_link
        )

        rows.append(
            {
                "worksheet": worksheet,
                "variable": variable,
                "answer_original": original_answer,
                "answer_refined": refined_answer,
                "confidence_original": original_confidence,
                "confidence_refined": refined_confidence,
                "page_number_original": original_page,
                "page_number_refined": refined_page,
                "source_fields_original": (
                    original_source_fields
                ),
                "source_fields_refined": (
                    refined_source_fields
                ),
                "formula_original": original_formula,
                "formula_refined": refined_formula,
                "source_link_original": (
                    original_source_link
                ),
                "source_link_refined": (
                    refined_source_link
                ),
                "changed": changed,
            }
        )

    return pd.DataFrame(rows)


def create_excel_output(state):
    """
    Create the final Excel output.

    Existing worksheet outputs are preserved and an
    additional Comparison worksheet is generated to
    compare original and final refined answers.
    """
    excel_data = state.get(
        "excel_data",
        [],
    )

    original_answers = state.get(
        "original_answers",
        [],
    )

    final_answers = state.get(
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

    if not output_path:
        raise ValueError(
            "Output path is not available."
        )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    output_excel = output_path

    if not output_excel.lower().endswith(
        ".xlsx"
    ):
        output_excel = (
            f"{output_excel}.xlsx"
        )

    aligned_answers = align_answers_to_template(
        excel_data,
        final_answers,
    )

    worksheet_dict = {}

    for row in aligned_answers:
        worksheet = row.get(
            "worksheet",
            "Sheet1",
        )

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
        final_answers,
    )

    worksheet_dict["Comparison"] = comparison_df

    status = excel_write(
        dataframes=worksheet_dict,
        output_path=output_excel,
    )

    return {
        "output_path": output_excel,
        "pdf_file_name": pdf_file_name,
        "status": status,
    }
```

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

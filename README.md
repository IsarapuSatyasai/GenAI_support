```python
workflow.add_conditional_edges(
    "confidence_refinement",
    route_after_refinement,
    {
        "refine": "confidence_refinement",
        "continue": "create_excel_output",
    },
)
```

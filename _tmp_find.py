import re

path = "project/streamlit_app.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

markers = [
    "def _read_uploaded_csv",
    "</style>",
    'st.button("Run Analysis"',
    'st.subheader("Run Summary")',
    "chart_entries = _collect_chart_paths",
    "No charts generated for this run.",
]
for i, line in enumerate(lines, start=1):
    for m in markers:
        if m in line:
            print(i, repr(line.rstrip()[:80]))
            break

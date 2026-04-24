"""Safe pandas code execution utilities."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_PROJECT_DIR = Path(__file__).resolve().parents[1]
_PLOTS_DIR = _PROJECT_DIR / "plots"


_BLOCKED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.While,
    ast.For,
    ast.AsyncFor,
)

_BLOCKED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "os",
    "pathlib",
    "setattr",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "vars",
}

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _pick_chart_type(action: str) -> str:
    """Infer chart type from planner action text."""
    lowered = action.lower()
    if "scatter" in lowered:
        return "scatter"
    if "bar" in lowered:
        return "bar"
    if "hist" in lowered:
        return "hist"
    if "line" in lowered:
        return "line"
    return "line"


def generate_plot(df: pd.DataFrame, action: str, step_index: int) -> dict[str, Any]:
    """Generate a deterministic plot file from a DataFrame and return metadata."""
    if df.empty:
        raise ValueError("Cannot generate plot: input DataFrame is empty.")

    chart_type = _pick_chart_type(action)
    numeric_cols = list(df.select_dtypes(include="number").columns)
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    try:
        if chart_type == "scatter":
            if len(numeric_cols) < 2:
                raise ValueError("Scatter plot requires at least two numeric columns.")
            df.plot(
                kind="scatter",
                x=numeric_cols[0],
                y=numeric_cols[1],
                ax=ax,
                title=action or "Scatter plot",
            )
        elif chart_type == "bar":
            if numeric_cols and non_numeric_cols:
                df.plot(
                    kind="bar",
                    x=non_numeric_cols[0],
                    y=numeric_cols[0],
                    ax=ax,
                    title=action or "Bar chart",
                )
            elif numeric_cols:
                df[numeric_cols[0]].plot(kind="bar", ax=ax, title=action or "Bar chart")
            else:
                raise ValueError("Bar plot requires at least one numeric column.")
        elif chart_type == "hist":
            if not numeric_cols:
                raise ValueError("Histogram requires at least one numeric column.")
            df[numeric_cols[0]].plot(kind="hist", ax=ax, title=action or "Histogram")
        else:
            if numeric_cols:
                df[numeric_cols[0]].plot(kind="line", ax=ax, title=action or "Line chart")
            else:
                df.reset_index().plot(kind="line", x="index", y=df.columns[0], ax=ax)
                ax.set_title(action or "Line chart")

        ax.set_xlabel(ax.get_xlabel() or "index")
        ax.set_ylabel(ax.get_ylabel() or "value")
        fig.tight_layout()

        _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _PLOTS_DIR / f"step_{step_index:02d}_{chart_type}.png"
        fig.savefig(output_path, format="png", dpi=150)
    finally:
        plt.close(fig)

    return {
        "type": "plot",
        "chart_type": chart_type,
        "path": str(output_path),
        "rows": int(len(df)),
        "columns": [str(col) for col in df.columns],
    }


def _validate_pandas_code(code: str) -> ast.Module:
    """Parse and validate user code before execution."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid pandas code: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODES):
            raise ValueError(
                f"Unsupported statement in pandas code: {type(node).__name__}"
            )

        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise ValueError(f"Blocked name used in pandas code: {node.id}")

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed.")

    return tree


def run_pandas_code(code: str, df: pd.DataFrame) -> Any:
    """Run constrained pandas code and return the `result` variable."""
    if not code or not code.strip():
        raise ValueError("Code cannot be empty.")

    tree = _validate_pandas_code(code)

    local_scope: dict[str, Any] = {"df": df.copy(deep=True)}
    global_scope: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "pd": pd}

    try:
        compiled = compile(tree, filename="<pandas_tool>", mode="exec")
        exec(compiled, global_scope, local_scope)
    except Exception as exc:  # pragma: no cover - passthrough wrapper
        raise RuntimeError(f"Failed to execute pandas code: {exc}") from exc

    if "result" not in local_scope:
        raise ValueError(
            "Code must assign the final output to a variable named 'result'."
        )

    return local_scope["result"]

"""Safe pandas code execution utilities."""

from __future__ import annotations

import ast
import multiprocessing as mp
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_PROJECT_DIR = Path(__file__).resolve().parents[1]
_PLOTS_DIR = _PROJECT_DIR / "plots"
_PLOT_FILE_RE = re.compile(
    r"^(?:[a-f0-9]{32}_step_\d{2}_[a-z0-9_-]+|step_\d{2}_[a-z0-9_-]+)\.png$"
)
_PLOT_RETENTION_SECONDS = 60 * 60
_DEFAULT_PANDAS_TIMEOUT_SECONDS = 10.0


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

try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
    sns.set_theme(style="whitegrid", palette="husl")
except ImportError:
    _SEABORN_AVAILABLE = False


def _pick_chart_type(action: str) -> str:
    """Infer chart type from planner action text and data patterns."""
    lowered = action.lower()
    
    chart_keywords = {
        "scatter": "scatter",
        "bubble": "bubble",
        "box": "box",
        "boxplot": "box",
        "violin": "violin",
        "heatmap": "heatmap",
        "heat": "heatmap",
        "area": "area",
        "pie": "pie",
        "distribution": "kde",
        "density": "kde",
        "kde": "kde",
        "hist": "hist",
        "histogram": "hist",
        "bar": "bar",
        "line": "line",
        "trend": "line",
    }
    
    for keyword, chart_type in chart_keywords.items():
        if keyword in lowered:
            return chart_type
    
    return "line"


def _best_chart_for_data(df: pd.DataFrame) -> str:
    """Auto-recommend chart type based on data characteristics."""
    if df.empty:
        return "line"
    
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    num_rows = len(df)
    
    if len(numeric_cols) >= 2 and len(categorical_cols) == 0:
        if num_rows > 50 and len(numeric_cols) >= 3:
            return "bubble"
        return "scatter"
    
    if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
        if all(df[cat].nunique() <= 10 for cat in categorical_cols[:1]):
            return "heatmap"
        return "box"
    
    if len(numeric_cols) == 1:
        if num_rows > 30:
            return "kde"
        return "hist"
    
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cardinality = df[categorical_cols[0]].nunique()
        if cardinality <= 5:
            return "pie"
        return "bar"
    
    return "line"


def _render_scatter(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str]) -> None:
    """Render scatter plot."""
    if len(numeric_cols) < 2:
        raise ValueError("Scatter plot requires at least two numeric columns.")
    df.plot(kind="scatter", x=numeric_cols[0], y=numeric_cols[1], ax=ax, title=title, alpha=0.6, s=50)


def _render_bubble(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str]) -> None:
    """Render bubble chart (3D scatter with size dimension)."""
    if len(numeric_cols) < 3:
        raise ValueError("Bubble chart requires at least three numeric columns.")
    sizes = pd.to_numeric(df[numeric_cols[2]], errors="coerce").fillna(50)
    sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min() + 1)) * 200 + 20
    ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], s=sizes, alpha=0.6, c=range(len(df)), cmap="viridis")
    ax.set_xlabel(numeric_cols[0])
    ax.set_ylabel(numeric_cols[1])
    ax.set_title(title)


def _render_box(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    """Render box plot for distribution visualization."""
    if not numeric_cols:
        raise ValueError("Box plot requires at least one numeric column.")
    if categorical_cols:
        df.boxplot(column=numeric_cols[0], by=categorical_cols[0], ax=ax)
        ax.set_title(title)
    else:
        df[numeric_cols[0]].plot(kind="box", ax=ax, title=title)


def _render_violin(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    """Render violin plot using seaborn if available."""
    if not _SEABORN_AVAILABLE:
        raise ValueError("Violin plot requires seaborn. Please install: pip install seaborn")
    if not numeric_cols:
        raise ValueError("Violin plot requires at least one numeric column.")
    if categorical_cols:
        sns.violinplot(data=df, x=categorical_cols[0], y=numeric_cols[0], ax=ax)
    else:
        sns.violinplot(data=df, y=numeric_cols[0], ax=ax)
    ax.set_title(title)


def _render_heatmap(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str]) -> None:
    """Render heatmap for 2D data intensity mapping."""
    if not _SEABORN_AVAILABLE:
        raise ValueError("Heatmap requires seaborn. Please install: pip install seaborn")
    if len(numeric_cols) < 2:
        heatmap_data = df.select_dtypes(include="number").iloc[:, :2]
    else:
        heatmap_data = df[numeric_cols].corr()
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title(title)


def _render_area(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str]) -> None:
    """Render area chart for trend comparison."""
    if not numeric_cols:
        raise ValueError("Area chart requires at least one numeric column.")
    df[numeric_cols[:3]].plot(kind="area", ax=ax, alpha=0.5, title=title)


def _render_pie(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    """Render pie chart for proportional breakdown."""
    if categorical_cols and numeric_cols:
        grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].sum()
        grouped.plot(kind="pie", ax=ax, title=title, autopct="%1.1f%%")
    elif numeric_cols:
        df[numeric_cols[0]].plot(kind="pie", ax=ax, title=title, autopct="%1.1f%%")
    else:
        raise ValueError("Pie chart requires at least one numeric column.")
    ax.set_ylabel("")


def _render_kde(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str]) -> None:
    """Render KDE (Kernel Density Estimate) plot."""
    if not numeric_cols:
        raise ValueError("KDE plot requires at least one numeric column.")
    df[numeric_cols[0]].plot(kind="kde", ax=ax, title=title, linewidth=2)


def _render_hist(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str]) -> None:
    """Render histogram."""
    if not numeric_cols:
        raise ValueError("Histogram requires at least one numeric column.")
    df[numeric_cols[0]].plot(kind="hist", ax=ax, title=title, bins=20, alpha=0.7, edgecolor="black")


def _render_bar(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    """Render bar chart."""
    if numeric_cols and categorical_cols:
        df.plot(kind="bar", x=categorical_cols[0], y=numeric_cols[0], ax=ax, title=title, legend=False)
    elif numeric_cols:
        df[numeric_cols[0]].plot(kind="bar", ax=ax, title=title)
    else:
        raise ValueError("Bar plot requires at least one numeric column.")


def _render_line(df: pd.DataFrame, ax: Any, title: str, numeric_cols: list[str]) -> None:
    """Render line chart."""
    if numeric_cols:
        df[numeric_cols[:3]].plot(kind="line", ax=ax, title=title, marker="o", linewidth=2)
    else:
        df.reset_index().plot(kind="line", x="index", y=df.columns[0], ax=ax, title=title)


class PandasExecutionTimeoutError(TimeoutError):
    """Raised when untrusted pandas code exceeds the allowed execution time."""


def _execute_pandas_code(code: str, df: pd.DataFrame) -> Any:
    """Validate and execute pandas code, returning the `result` variable."""
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


def _execute_pandas_code_worker(code: str, df: pd.DataFrame, result_queue: mp.Queue) -> None:
    """Run pandas code in a helper process and return the serialized outcome."""
    try:
        result_queue.put(("ok", _execute_pandas_code(code, df)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_pandas_code_with_timeout(code: str, df: pd.DataFrame, timeout_seconds: float) -> Any:
    """Execute pandas code with a wall-clock timeout."""
    if timeout_seconds <= 0:
        return _execute_pandas_code(code, df)

    if hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer") and sys.platform != "win32":
        def _handle_timeout(signum: int, frame: Any) -> None:
            raise PandasExecutionTimeoutError(
                f"Pandas execution exceeded {timeout_seconds}s and was aborted"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            return _execute_pandas_code(code, df)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    # Windows does not provide SIGALRM, so we fall back to a helper process and
    # terminate it if the execution overruns the timeout.
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    process = context.Process(target=_execute_pandas_code_worker, args=(code, df, result_queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        raise PandasExecutionTimeoutError(
            f"Pandas execution exceeded {timeout_seconds}s and was aborted"
        )

    if result_queue.empty():
        raise RuntimeError("Failed to execute pandas code: worker returned no result")

    status, payload = result_queue.get()
    if status == "ok":
        return payload

    raise RuntimeError(f"Failed to execute pandas code: {payload}")


def cleanup_plot_files(max_age_seconds: int = _PLOT_RETENTION_SECONDS) -> int:
    """Remove stale plot files that match the known plot filename pattern."""
    if max_age_seconds <= 0 or not _PLOTS_DIR.exists():
        return 0

    cutoff_timestamp = time.time() - max_age_seconds
    removed_count = 0

    for plot_file in _PLOTS_DIR.iterdir():
        if not plot_file.is_file() or not _PLOT_FILE_RE.match(plot_file.name):
            continue

        try:
            if plot_file.stat().st_mtime >= cutoff_timestamp:
                continue
            plot_file.unlink()
            removed_count += 1
        except OSError:
            continue

    return removed_count


def generate_plot(
    df: pd.DataFrame,
    action: str,
    step_index: int,
    run_id: str,
) -> dict[str, Any]:
    """Generate a deterministic plot file from a DataFrame and return metadata."""
    if df.empty:
        raise ValueError("Cannot generate plot: input DataFrame is empty.")

    chart_type = _pick_chart_type(action)
    if chart_type == "auto":
        chart_type = _best_chart_for_data(df)
    
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if chart_type == "scatter":
            _render_scatter(df, ax, action or "Scatter plot", numeric_cols)
        elif chart_type == "bubble":
            _render_bubble(df, ax, action or "Bubble chart", numeric_cols)
        elif chart_type == "box":
            _render_box(df, ax, action or "Box plot", numeric_cols, categorical_cols)
        elif chart_type == "violin":
            _render_violin(df, ax, action or "Violin plot", numeric_cols, categorical_cols)
        elif chart_type == "heatmap":
            _render_heatmap(df, ax, action or "Heatmap", numeric_cols)
        elif chart_type == "area":
            _render_area(df, ax, action or "Area chart", numeric_cols)
        elif chart_type == "pie":
            _render_pie(df, ax, action or "Pie chart", numeric_cols, categorical_cols)
        elif chart_type == "kde":
            _render_kde(df, ax, action or "KDE plot", numeric_cols)
        elif chart_type == "hist":
            _render_hist(df, ax, action or "Histogram", numeric_cols)
        elif chart_type == "bar":
            _render_bar(df, ax, action or "Bar chart", numeric_cols, categorical_cols)
        else:
            _render_line(df, ax, action or "Line chart", numeric_cols)

        ax.set_xlabel(ax.get_xlabel() or "index")
        ax.set_ylabel(ax.get_ylabel() or "value")
        fig.tight_layout()

        _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        sanitized_run_id = re.sub(r"[^a-f0-9]", "", str(run_id).lower()) or "run"
        output_path = _PLOTS_DIR / f"{sanitized_run_id}_step_{step_index:02d}_{chart_type}.png"
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

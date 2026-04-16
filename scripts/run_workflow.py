"""Debug runner — execute workflows up to and including a target workflow.

Designed for prompt tuning, structured output inspection, and iterating on
workflow logic before the frontend exists. Uses real API calls (FRED, Anthropic).
No database required — LLM usage is logged to stdout instead.

When you specify a workflow that depends on prior results (e.g. HistoricalAnalog
requires MacroContext), all predecessors in the pipeline are run first and their
results are accumulated in context.prior_results automatically.

Results are saved to output/<workflow>_<timestamp>.txt for every step with:
  1. Agent Inferences
  2. Raw LLM Output
  3. Full Series Data (structured_output with complete historical_data lists)

The terminal print omits full series history to keep output readable.
Predecessor steps print a compact summary; only the target workflow prints full output.

Usage examples
--------------
Run MacroContextWorkflow with a thesis file:
    uv run python scripts/run_workflow.py --thesis examples/thesis_yield_curve.toml

Run HistoricalAnalog (MacroContext runs first automatically):
    uv run python scripts/run_workflow.py --workflow HistoricalAnalogWorkflow \\
        --thesis examples/thesis_yield_curve.toml

Run WebResearch (MacroContext → HistoricalAnalog → InstrumentAnalysis run first):
    uv run python scripts/run_workflow.py --workflow WebResearchWorkflow \\
        --thesis examples/thesis_yield_curve.toml

Include OECD data (used by MacroContextWorkflow):
    uv run python scripts/run_workflow.py \\
        --thesis examples/thesis_yield_curve.toml --with-oecd

Control log verbosity:
    uv run python scripts/run_workflow.py \\
        --thesis examples/thesis_yield_curve.toml --log-level INFO
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import textwrap
import time
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Logging setup — call before any imports that trigger logging
# ---------------------------------------------------------------------------


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.DEBUG)
    fmt = "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=numeric, format=fmt, datefmt=datefmt, force=True)
    # Suppress overly chatty third-party loggers at DEBUG
    if numeric <= logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("anthropic").setLevel(logging.INFO)
        logging.getLogger("urllib3").setLevel(logging.INFO)


logger = logging.getLogger("run_workflow")


# ---------------------------------------------------------------------------
# Stub DB — logs usage instead of writing to PostgreSQL
# ---------------------------------------------------------------------------


class _StubDB:
    """Drop-in replacement for a SQLAlchemy session in the debug runner.

    Prints LLM usage rows to stdout instead of persisting them. All other
    operations are no-ops so AnthropicClient doesn't raise.
    """

    def add(self, obj) -> None:
        from app.models.log import LLMUsageLog

        if isinstance(obj, LLMUsageLog):
            logger.info(
                "[llm_usage_log]  model=%-22s  task=%-20s  "
                "in=%5d  out=%5d  cost=$%.6f",
                obj.model,
                obj.task_type,
                obj.input_tokens,
                obj.output_tokens,
                obj.estimated_cost_usd,
            )

    def commit(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def query(self, *_):
        return self

    def filter(self, *_):
        return self

    def first(self):
        return None


# ---------------------------------------------------------------------------
# Stub Thesis — built from TOML file or CLI args
# ---------------------------------------------------------------------------


@dataclass
class _StubThesis:
    """Minimal thesis stand-in that satisfies the workflow interface."""

    id: uuid.UUID
    pod_id: uuid.UUID
    title: str
    time_horizon: str
    notes: str | None
    _direction_str: str

    @property
    def direction(self) -> SimpleNamespace:
        return SimpleNamespace(value=self._direction_str)


def _load_thesis(args: argparse.Namespace) -> _StubThesis:
    """Build a _StubThesis from a TOML file and/or CLI overrides."""
    data: dict = {}

    if args.thesis:
        path = Path(args.thesis)
        if not path.exists():
            logger.error("Thesis file not found: %s", path)
            sys.exit(1)
        with open(path, "rb") as f:
            data = tomllib.load(f)
        logger.debug("Loaded thesis from %s: %s", path, data)

    # CLI args override file values
    if args.title:
        data["title"] = args.title
    if args.direction:
        data["direction"] = args.direction
    if args.horizon:
        data["time_horizon"] = args.horizon
    if args.notes:
        data["notes"] = args.notes

    required = ["title", "direction", "time_horizon"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        logger.error(
            "Missing required thesis fields: %s. "
            "Provide via --thesis file or CLI args (--title, --direction, --horizon).",
            missing,
        )
        sys.exit(1)

    direction = data["direction"].strip().lower()
    if direction not in ("long", "short"):
        logger.error("--direction must be 'long' or 'short', got: %r", direction)
        sys.exit(1)

    thesis = _StubThesis(
        id=uuid.uuid4(),
        pod_id=uuid.uuid4(),
        title=data["title"],
        time_horizon=data["time_horizon"],
        notes=data.get("notes"),
        _direction_str=direction,
    )
    logger.info(
        "Thesis: %r  direction=%s  horizon=%s",
        thesis.title,
        thesis.direction.value,
        thesis.time_horizon,
    )
    return thesis


# ---------------------------------------------------------------------------
# Workflow pipeline — ordered list defines execution sequence
# ---------------------------------------------------------------------------

# Each entry: (class_name, module_path).  Order here IS the pipeline order.
PIPELINE: list[tuple[str, str]] = [
    ("MacroContextWorkflow", "app.workflows.macro_context"),
    ("HistoricalAnalogWorkflow", "app.workflows.historical_analog"),
    ("InstrumentAnalysisWorkflow", "app.workflows.instrument_analysis"),
    ("WebResearchWorkflow", "app.workflows.web_research"),
    ("BacktestWorkflow", "app.workflows.backtest"),
]

_PIPELINE_NAMES = [name for name, _ in PIPELINE]


def _load_workflow_class(name: str):
    for cls_name, module_path in PIPELINE:
        if cls_name == name:
            module = importlib.import_module(module_path)
            return getattr(module, cls_name)
    logger.error(
        "Unknown workflow: %r. Available: %s",
        name,
        _PIPELINE_NAMES,
    )
    sys.exit(1)


def _pipeline_slice(target: str) -> list[str]:
    """Return the ordered list of workflow names to run, up to and including target."""
    try:
        idx = _PIPELINE_NAMES.index(target)
    except ValueError:
        logger.error(
            "Unknown workflow: %r. Available: %s",
            target,
            _PIPELINE_NAMES,
        )
        sys.exit(1)
    return _PIPELINE_NAMES[: idx + 1]


def _instantiate_workflow(name: str, clients: dict):
    """Instantiate a workflow class, passing only the clients it accepts."""
    cls = _load_workflow_class(name)
    import inspect

    sig = inspect.signature(cls.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in clients.items() if k in params}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_DIVIDER = "─" * 72
_OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Keys whose values are long data lists — suppressed in terminal, kept in file.
_SERIES_KEYS = {"historical_data", "recent_prompt_data"}


def _print_section(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def _strip_series(structured_output: dict) -> dict:
    """Return a copy of structured_output with long series lists replaced by a
    placeholder, so the terminal stays readable."""
    result = {}
    for key, value in structured_output.items():
        if isinstance(value, list):
            # Top-level series lists
            if key in _SERIES_KEYS:
                result[key] = f"[{len(value)} observations — see saved output file]"
            else:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = _strip_series(value)
        else:
            result[key] = value
    # Handle list-of-series-dicts (e.g. list of _SeriesSnapshot dicts)
    return result


def _strip_series_list(obj):
    """Recursively strip series keys from any structure (dict or list of dicts)."""
    if isinstance(obj, dict):
        return _strip_series(obj)
    if isinstance(obj, list):
        return [_strip_series_list(item) for item in obj]
    return obj


def _print_result(result, elapsed: float) -> None:
    _print_section(f"RESULT  status={result.status}  elapsed={elapsed:.2f}s")

    _print_section(f"AGENT INFERENCES  ({len(result.agent_inferences)})")
    for inf in result.agent_inferences:
        print(textwrap.fill(f"  • {inf}", width=80, subsequent_indent="    "))

    _print_section("RAW LLM OUTPUT")
    print(result.raw_output)

    _print_section("STRUCTURED OUTPUT  (series data truncated — see saved file)")
    stripped = _strip_series_list(result.structured_output)
    print(json.dumps(stripped, indent=2, default=str))

    _print_section(f"CITATIONS  ({len(result.citations)})")
    for i, c in enumerate(result.citations, 1):
        print(f"  [{i}] {c.source_type}  {c.label}")

    print(f"\n{_DIVIDER}")


def _save_result(result, workflow_name: str, thesis, elapsed: float) -> Path:
    """Save the full result to output/<workflow>_<timestamp>.txt.

    Order: Thesis → Agent Inferences → Raw LLM Output → Full Series Data.
    """
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_workflow = workflow_name.replace(" ", "_")
    out_path = _OUTPUT_DIR / f"{safe_workflow}_{timestamp}.txt"

    div = "─" * 72

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{div}\n")
        f.write(f"  {workflow_name}\n")
        f.write(f"  Status: {result.status}   Elapsed: {elapsed:.2f}s\n")
        f.write(f"  Saved:  {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{div}\n")

        f.write(f"\n{div}\n  THESIS\n{div}\n")
        f.write(f"  Title:     {thesis.title}\n")
        f.write(f"  Direction: {thesis.direction.value}\n")
        f.write(f"  Horizon:   {thesis.time_horizon}\n")
        if thesis.notes:
            f.write("\n  Notes:\n")
            for line in thesis.notes.splitlines():
                f.write(f"    {line}\n")

        f.write(
            f"\n{div}\n  AGENT INFERENCES  ({len(result.agent_inferences)})\n{div}\n"
        )
        for inf in result.agent_inferences:
            f.write(
                textwrap.fill(f"  • {inf}", width=80, subsequent_indent="    ") + "\n"
            )

        f.write(f"\n{div}\n  RAW LLM OUTPUT\n{div}\n")
        f.write(result.raw_output)
        f.write("\n")

        f.write(f"\n{div}\n  FULL SERIES DATA (structured_output)\n{div}\n")
        f.write(json.dumps(result.structured_output, indent=2, default=str))
        f.write("\n")

        f.write(f"\n{div}\n  CITATIONS  ({len(result.citations)})\n{div}\n")
        for i, c in enumerate(result.citations, 1):
            f.write(f"  [{i}] {c.source_type}  {c.label}\n")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug runner: execute a single workflow and dump full output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--workflow",
        default="MacroContextWorkflow",
        help="Workflow class name (default: MacroContextWorkflow)",
    )
    p.add_argument(
        "--thesis",
        metavar="FILE",
        help="Path to a TOML thesis file (e.g. examples/thesis_yield_curve.toml)",
    )
    p.add_argument("--title", help="Thesis title (overrides file)")
    p.add_argument(
        "--direction",
        choices=["long", "short"],
        help="Trade direction (overrides file)",
    )
    p.add_argument("--horizon", help="Time horizon, e.g. '6 months' (overrides file)")
    p.add_argument("--notes", help="Freeform thesis notes (overrides file)")
    p.add_argument(
        "--with-oecd",
        action="store_true",
        help="Also pull OECD series (ECB rate, Eurozone CPI, OECD CLI)",
    )
    p.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: DEBUG)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    logger.info("=== Workflow Debug Runner ===")

    # Load settings (.env)
    from app.core.settings import get_settings

    try:
        settings = get_settings()
        logger.debug("Settings loaded from .env")
    except Exception as exc:
        logger.error("Failed to load settings: %s", exc)
        sys.exit(1)

    # Build thesis
    thesis = _load_thesis(args)

    # Build all available clients up front
    from app.integrations.anthropic_client import AnthropicClient
    from app.integrations.fred_client import FREDClient
    from app.integrations.oecd_client import OECDClient
    from app.integrations.web_search_client import WebSearchClient

    stub_db = _StubDB()
    clients: dict = {
        "anthropic_client": AnthropicClient(
            api_key=settings.anthropic_api_key, db=stub_db
        ),
        "fred_client": FREDClient(api_key=settings.fred_api_key),
        "web_search_client": WebSearchClient(api_key=settings.anthropic_api_key),
    }
    if args.with_oecd:
        clients["oecd_client"] = OECDClient()
        logger.debug("OECDClient ready")
    logger.debug("All clients ready")

    # Determine which workflows to run (all predecessors + target)
    steps = _pipeline_slice(args.workflow)
    target = steps[-1]
    predecessors = steps[:-1]

    if predecessors:
        logger.info(
            "Pipeline: %s",
            " → ".join(steps),
        )
    else:
        logger.info("Running: %s", target)

    # Build context once; prior_results accumulates across steps
    from app.workflows.base import WorkflowContext

    context = WorkflowContext(thesis=thesis, db=stub_db)

    # Run predecessor workflows, accumulating results into context
    for step_name in predecessors:
        logger.info("--- Predecessor: %s ---", step_name)
        workflow = _instantiate_workflow(step_name, clients)
        t0 = time.perf_counter()
        try:
            result = workflow.execute(thesis, context)
        except Exception as exc:
            logger.exception("Predecessor %s raised an exception: %s", step_name, exc)
            sys.exit(1)
        elapsed = time.perf_counter() - t0
        context.prior_results.append(result)
        out_path = _save_result(result, step_name, thesis, elapsed)
        logger.info(
            "  %s  status=%s  elapsed=%.2fs  saved=%s",
            step_name,
            result.status,
            elapsed,
            out_path.name,
        )

    # Run the target workflow and print full output
    logger.info("--- Target: %s ---", target)
    workflow = _instantiate_workflow(target, clients)
    t0 = time.perf_counter()
    try:
        result = workflow.execute(thesis, context)
    except Exception as exc:
        logger.exception("Workflow raised an exception: %s", exc)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    _print_result(result, elapsed)
    out_path = _save_result(result, target, thesis, elapsed)
    print(f"\n  Saved to: {out_path}\n")


if __name__ == "__main__":
    main()

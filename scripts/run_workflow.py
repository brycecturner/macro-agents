"""Debug runner — execute a single workflow against a thesis and dump full output.

Designed for prompt tuning, structured output inspection, and iterating on
workflow logic before the frontend exists. Uses real API calls (FRED, Anthropic).
No database required — LLM usage is logged to stdout instead.

Usage examples
--------------
Run MacroContextWorkflow with a thesis file:
    uv run python scripts/run_workflow.py --thesis examples/thesis_yield_curve.toml

Run with inline args:
    uv run python scripts/run_workflow.py \\
        --title "Yield Curve Steepener" \\
        --direction long \\
        --horizon "6 months" \\
        --notes "Long TLT as yield curve steepens."

Include OECD data:
    uv run python scripts/run_workflow.py \\
        --thesis examples/thesis_yield_curve.toml --with-oecd

Run a different workflow (once implemented):
    uv run python scripts/run_workflow.py --workflow HistoricalAnalogWorkflow \\
        --thesis examples/thesis_yield_curve.toml

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
# Workflow discovery
# ---------------------------------------------------------------------------

_WORKFLOW_REGISTRY: dict[str, str] = {
    "MacroContextWorkflow": "app.workflows.macro_context",
}


def _load_workflow_class(name: str):
    if name not in _WORKFLOW_REGISTRY:
        logger.error(
            "Unknown workflow: %r. Available: %s",
            name,
            list(_WORKFLOW_REGISTRY.keys()),
        )
        sys.exit(1)
    module = importlib.import_module(_WORKFLOW_REGISTRY[name])
    return getattr(module, name)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


_DIVIDER = "─" * 72


def _print_section(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def _print_result(result, elapsed: float) -> None:
    _print_section(f"RESULT  status={result.status}  elapsed={elapsed:.2f}s")

    _print_section("STRUCTURED OUTPUT")
    print(json.dumps(result.structured_output, indent=2, default=str))

    _print_section(f"CITATIONS  ({len(result.citations)})")
    for i, c in enumerate(result.citations, 1):
        print(f"  [{i}] {c.source_type}  {c.label}")

    _print_section(f"AGENT INFERENCES  ({len(result.agent_inferences)})")
    for inf in result.agent_inferences:
        print(textwrap.fill(f"  • {inf}", width=80, subsequent_indent="    "))

    _print_section("RAW LLM OUTPUT")
    print(result.raw_output)

    print(f"\n{_DIVIDER}")


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
    logger.info("Workflow: %s", args.workflow)

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

    # Build clients
    from app.integrations.anthropic_client import AnthropicClient
    from app.integrations.fred_client import FREDClient
    from app.integrations.oecd_client import OECDClient

    fred = FREDClient(api_key=settings.fred_api_key)
    logger.debug("FREDClient ready")

    oecd = OECDClient() if args.with_oecd else None
    if oecd:
        logger.debug("OECDClient ready")

    stub_db = _StubDB()
    anthropic = AnthropicClient(api_key=settings.anthropic_api_key, db=stub_db)
    logger.debug("AnthropicClient ready")

    # Build workflow
    workflow_cls = _load_workflow_class(args.workflow)

    # Wire up clients — MacroContextWorkflow accepts these kwargs; other
    # workflows may have a different signature as they're implemented.
    init_kwargs: dict = {"anthropic_client": anthropic}
    if hasattr(workflow_cls.__init__, "__code__"):
        params = workflow_cls.__init__.__code__.co_varnames
        if "fred_client" in params:
            init_kwargs["fred_client"] = fred
        if "oecd_client" in params and oecd is not None:
            init_kwargs["oecd_client"] = oecd

    workflow = workflow_cls(**init_kwargs)
    logger.info("Running %s...", args.workflow)

    # Build context (no DB needed — WorkflowRunner would normally provide this)
    from app.workflows.base import WorkflowContext

    context = WorkflowContext(thesis=thesis, db=stub_db)

    # Execute
    t0 = time.perf_counter()
    try:
        result = workflow.execute(thesis, context)
    except Exception as exc:
        logger.exception("Workflow raised an exception: %s", exc)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    # Print full result
    _print_result(result, elapsed)


if __name__ == "__main__":
    main()

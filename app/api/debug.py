"""Debug endpoints — not part of the production API surface.

These routes exist for operational verification only. They make real outbound
calls to external APIs so problems can be confirmed or ruled out quickly
without running a full workflow.

Mount: included in main.py under the /debug prefix.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter

from app.integrations.oecd_client import OECDClient, OECDClientError, OECDSeriesSpec
from app.workflows.macro_context import _OECD_CORE_SERIES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


def _probe_series(client: OECDClient, spec: OECDSeriesSpec) -> dict:
    """Fetch one series and return a status dict. Never raises."""
    try:
        result = client.get_series(spec)
        series = result.data.dropna()
        if series.empty:
            return {
                "label": spec.label,
                "agency": spec.agency,
                "dataflow": spec.dataflow,
                "version": spec.version,
                "dimension_key": spec.dimension_key,
                "status": "empty",
                "error": None,
                "observation_count": 0,
                "latest_date": None,
                "latest_value": None,
                "retrieved_at": result.retrieved_at.isoformat(),
            }
        return {
            "label": spec.label,
            "agency": spec.agency,
            "dataflow": spec.dataflow,
            "version": spec.version,
            "dimension_key": spec.dimension_key,
            "status": "ok",
            "error": None,
            "observation_count": len(series),
            "latest_date": series.index[-1].date().isoformat(),
            "latest_value": round(float(series.iloc[-1]), 6),
            "retrieved_at": result.retrieved_at.isoformat(),
        }
    except OECDClientError as exc:
        logger.warning("OECD probe failed for %r: %s", spec.label, exc)
        return {
            "label": spec.label,
            "agency": spec.agency,
            "dataflow": spec.dataflow,
            "version": spec.version,
            "dimension_key": spec.dimension_key,
            "status": "error",
            "error": str(exc),
            "observation_count": None,
            "latest_date": None,
            "latest_value": None,
            "retrieved_at": datetime.now(tz=UTC).isoformat(),
        }


@router.get("/oecd-probe")
def oecd_probe() -> dict:
    """Probe each OECD core series and report live fetch status.

    Makes real outbound calls to sdmx.oecd.org — one per core series.
    Returns per-series status, latest observation, and any error message.
    Use this to verify dimension keys are correct after any API migration.

    Response shape::

        {
            "probed_at": "<ISO timestamp>",
            "series": [
                {
                    "label": "ECB Policy Rate (Short-term Call Rate)",
                    "status": "ok" | "error" | "empty",
                    "observation_count": 360,
                    "latest_date": "2026-01",
                    "latest_value": 1.931671,
                    "error": null,
                    ...
                },
                ...
            ],
            "all_ok": true | false
        }
    """
    probed_at = datetime.now(tz=UTC).isoformat()
    client = OECDClient()
    try:
        results = [_probe_series(client, spec) for spec in _OECD_CORE_SERIES]
    finally:
        client.close()

    all_ok = all(r["status"] == "ok" for r in results)
    return {
        "probed_at": probed_at,
        "series": results,
        "all_ok": all_ok,
    }

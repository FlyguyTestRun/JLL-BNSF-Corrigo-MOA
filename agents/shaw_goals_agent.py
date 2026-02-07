"""Shaw Goals Integration Agent for JLL-BNSF Corrigo MAO.

National-level integration with the Shaw Goals application. Tracks goal
alignment between individual maintenance technician goals, team PM completion
targets, and campus-level KPIs at the BNSF Railway campus (Fort Worth, TX).

Designed for the Multi-Agent Orchestration (MAO) system with strict token
conservation: all prompts, responses, and inter-agent messages use concise
structured JSON and avoid verbose free-text wherever possible.

Python 3.11+
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Any

import urllib.request
import urllib.error
import urllib.parse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("shaw_goals_agent")
logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
)
logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GoalStatus(str, Enum):
    """Status of a goal relative to its target."""

    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    BEHIND = "behind"


class EffortLevel(str, Enum):
    """Estimated effort for an integration opportunity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PMStatus(str, Enum):
    """Status of a preventive-maintenance task."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class PMTask:
    """A single preventive-maintenance task sourced from Corrigo."""

    id: str
    asset_id: str
    trade: str
    building: str
    floor: str
    frequency: str  # e.g. "weekly", "monthly", "quarterly"
    last_completed: datetime | None
    next_due: datetime
    estimated_minutes: int
    actual_minutes: int | None
    technician_id: str
    status: PMStatus

    def is_overdue(self) -> bool:
        """Return True when the task has passed its due date without completion."""
        return (
            self.status not in (PMStatus.COMPLETED, PMStatus.CANCELLED)
            and datetime.now() > self.next_due
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        data = asdict(self)
        data["last_completed"] = (
            self.last_completed.isoformat() if self.last_completed else None
        )
        data["next_due"] = self.next_due.isoformat()
        data["status"] = self.status.value
        return data


@dataclass
class GoalAlignment:
    """Maps a Shaw Goals objective to a Corrigo-derived metric."""

    goal_id: str
    goal_name: str
    corrigo_metric: str
    target_value: float
    current_value: float
    variance_pct: float = field(init=False)
    status: GoalStatus = field(init=False)

    def __post_init__(self) -> None:
        if self.target_value == 0:
            self.variance_pct = 0.0
        else:
            self.variance_pct = round(
                ((self.current_value - self.target_value) / self.target_value) * 100, 2
            )
        self.status = self._derive_status()

    def _derive_status(self) -> GoalStatus:
        if self.variance_pct >= 0:
            return GoalStatus.ON_TRACK
        if self.variance_pct >= -10:
            return GoalStatus.AT_RISK
        return GoalStatus.BEHIND

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class IntegrationOpportunity:
    """A discovered automation or integration opportunity."""

    id: str
    category: str
    description: str
    estimated_impact: str
    effort_level: EffortLevel
    corrigo_endpoint: str
    priority_score: float  # 0.0 - 10.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["effort_level"] = self.effort_level.value
        return data


# ---------------------------------------------------------------------------
# Token Budget
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Track and enforce token usage across agent calls.

    Enforces a hard ceiling of *max_tokens_per_call* on every individual
    invocation and logs cumulative consumption so the orchestrator can
    monitor cost.
    """

    max_tokens_per_call: int = 500
    cumulative_tokens_used: int = 0
    call_count: int = 0
    _call_log: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def consume(self, tokens: int, call_label: str = "") -> int:
        """Record *tokens* consumed for a single call.

        Returns the clamped token count actually recorded (never exceeds
        *max_tokens_per_call*).

        Raises
        ------
        ValueError
            If *tokens* is negative.
        """
        if tokens < 0:
            raise ValueError("Token count cannot be negative.")

        clamped = min(tokens, self.max_tokens_per_call)
        self.cumulative_tokens_used += clamped
        self.call_count += 1
        entry = {
            "call": self.call_count,
            "label": call_label or f"call_{self.call_count}",
            "requested": tokens,
            "recorded": clamped,
            "cumulative": self.cumulative_tokens_used,
            "ts": datetime.now().isoformat(),
        }
        self._call_log.append(entry)
        logger.debug("TokenBudget | %s", json.dumps(entry))
        return clamped

    @property
    def remaining_per_call(self) -> int:
        """Return the per-call budget (constant)."""
        return self.max_tokens_per_call

    def summary(self) -> dict[str, Any]:
        """Return a concise summary for inter-agent messaging."""
        return {
            "max_per_call": self.max_tokens_per_call,
            "calls": self.call_count,
            "total_tokens": self.cumulative_tokens_used,
        }


# ---------------------------------------------------------------------------
# Corrigo API Client (lightweight wrapper)
# ---------------------------------------------------------------------------

class CorrigoAPIClient:
    """Minimal REST client for the Corrigo Enterprise API.

    Uses OAuth 2.0 bearer-token authentication. In production the *token*
    is obtained from the ``/OAuth/Token`` endpoint; here it is passed in
    directly or sourced from environment configuration.
    """

    DEFAULT_BASE_URL = "https://am-api.corrigo.com/api/v1"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        client_id: str = "PLACEHOLDER_CLIENT_ID",
        client_secret: str = "PLACEHOLDER_CLIENT_SECRET",
        company_name: str = "JLL-BNSF",
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.company_name = company_name
        self._token: str | None = token
        self._token_expiry: datetime | None = None

    # -- auth ---------------------------------------------------------------

    def _ensure_token(self) -> str:
        """Return a valid bearer token, refreshing if necessary."""
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._token

        logger.info("Requesting new Corrigo OAuth token for %s", self.company_name)
        payload = json.dumps({
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "company": self.company_name,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/OAuth/Token",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
                self._token = body["access_token"]
                self._token_expiry = datetime.now() + timedelta(
                    seconds=body.get("expires_in", 3600)
                )
        except (urllib.error.URLError, KeyError) as exc:
            logger.error("OAuth token request failed: %s", exc)
            raise ConnectionError(f"Corrigo auth failed: {exc}") from exc

        return self._token  # type: ignore[return-value]

    # -- generic request ----------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = self._ensure_token()
        url = f"{self.base_url}{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "CompanyName": self.company_name,
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            logger.error("Corrigo API %s %s failed: %s", method, endpoint, exc)
            raise

    # -- convenience methods ------------------------------------------------

    def get_work_orders(
        self,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """GET /WorkOrders with optional OData-style filters."""
        resp = self._request("GET", "/WorkOrders", params=filters)
        return resp.get("value", [])

    def get_assets(
        self,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """GET /Assets."""
        resp = self._request("GET", "/Assets", params=filters)
        return resp.get("value", [])

    def get_schedules(
        self,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """GET /Schedules (PM schedules)."""
        resp = self._request("GET", "/Schedules", params=filters)
        return resp.get("value", [])

    def get_employees(
        self,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """GET /Employees (technician roster)."""
        resp = self._request("GET", "/Employees", params=filters)
        return resp.get("value", [])


# ---------------------------------------------------------------------------
# Shaw Goals Agent
# ---------------------------------------------------------------------------

class ShawGoalsAgent:
    """National-level Shaw Goals integration agent.

    Bridges Corrigo CMMS data at the BNSF Fort Worth campus with the Shaw
    Goals application. Every public method returns compact JSON-serializable
    structures and respects the configured ``TokenBudget``.
    """

    CAMPUS = "BNSF Railway - Fort Worth, TX"

    def __init__(
        self,
        corrigo_client: CorrigoAPIClient | None = None,
        token_budget: TokenBudget | None = None,
    ) -> None:
        self.corrigo = corrigo_client or CorrigoAPIClient()
        self.budget = token_budget or TokenBudget()
        self._pm_cache: list[PMTask] = []
        logger.info(
            "ShawGoalsAgent initialized | campus=%s | budget=%d tok/call",
            self.CAMPUS,
            self.budget.max_tokens_per_call,
        )

    # ------------------------------------------------------------------
    # sync_corrigo_goals
    # ------------------------------------------------------------------

    def sync_corrigo_goals(
        self,
        pm_tasks: list[PMTask] | None = None,
        goals: list[dict[str, Any]] | None = None,
    ) -> list[GoalAlignment]:
        """Pull PM completion data and map to Shaw Goals objectives.

        Parameters
        ----------
        pm_tasks:
            Pre-fetched PM tasks. When *None*, the agent attempts to pull
            directly from Corrigo (will fail with placeholder credentials).
        goals:
            List of goal definitions, each containing at minimum
            ``goal_id``, ``goal_name``, ``corrigo_metric``, and ``target_value``.
            When *None*, a sensible set of BNSF campus defaults is used.

        Returns
        -------
        list[GoalAlignment]
            One alignment record per goal, scored against current Corrigo data.
        """
        self.budget.consume(120, "sync_corrigo_goals")
        logger.info("sync_corrigo_goals | start")

        tasks = pm_tasks if pm_tasks is not None else self._fetch_pm_tasks()
        self._pm_cache = tasks

        if goals is None:
            goals = self._default_goals()

        alignments: list[GoalAlignment] = []
        for g in goals:
            current = self._compute_metric(g["corrigo_metric"], tasks)
            alignment = GoalAlignment(
                goal_id=g["goal_id"],
                goal_name=g["goal_name"],
                corrigo_metric=g["corrigo_metric"],
                target_value=g["target_value"],
                current_value=current,
            )
            alignments.append(alignment)

        logger.info(
            "sync_corrigo_goals | aligned %d goals | %s",
            len(alignments),
            json.dumps([a.status.value for a in alignments]),
        )
        return alignments

    # ------------------------------------------------------------------
    # evaluate_pm_completion
    # ------------------------------------------------------------------

    def evaluate_pm_completion(
        self,
        pm_tasks: list[PMTask] | None = None,
        group_by: str = "technician",
    ) -> dict[str, Any]:
        """Score PM completion rate grouped by dimension.

        Parameters
        ----------
        pm_tasks:
            Pre-fetched tasks (uses internal cache when *None*).
        group_by:
            One of ``"technician"``, ``"trade"``, ``"building"``, ``"period"``.

        Returns
        -------
        dict
            ``{"group_by": ..., "results": [...], "overall_pct": ...}``
        """
        self.budget.consume(100, "evaluate_pm_completion")
        tasks = pm_tasks if pm_tasks is not None else self._pm_cache

        if not tasks:
            logger.warning("evaluate_pm_completion | no tasks available")
            return {"group_by": group_by, "results": [], "overall_pct": 0.0}

        valid_groups = ("technician", "trade", "building", "period")
        if group_by not in valid_groups:
            raise ValueError(f"group_by must be one of {valid_groups}")

        buckets: dict[str, list[PMTask]] = {}
        for t in tasks:
            key = self._bucket_key(t, group_by)
            buckets.setdefault(key, []).append(t)

        results: list[dict[str, Any]] = []
        for key, group_tasks in sorted(buckets.items()):
            total = len(group_tasks)
            completed = sum(
                1 for t in group_tasks if t.status == PMStatus.COMPLETED
            )
            pct = round((completed / total) * 100, 1) if total else 0.0
            results.append({
                "key": key,
                "total": total,
                "completed": completed,
                "completion_pct": pct,
            })

        total_all = len(tasks)
        completed_all = sum(1 for t in tasks if t.status == PMStatus.COMPLETED)
        overall = round((completed_all / total_all) * 100, 1) if total_all else 0.0

        payload = {
            "group_by": group_by,
            "results": results,
            "overall_pct": overall,
        }
        logger.info(
            "evaluate_pm_completion | group_by=%s | overall=%.1f%%",
            group_by,
            overall,
        )
        return payload

    # ------------------------------------------------------------------
    # identify_integration_opportunities
    # ------------------------------------------------------------------

    def identify_integration_opportunities(
        self,
        pm_tasks: list[PMTask] | None = None,
    ) -> list[IntegrationOpportunity]:
        """Analyze Corrigo work-order patterns for automation opportunities.

        Scans PM tasks for patterns such as chronic overdue work, large
        variance between estimated and actual labour-minutes, and trades
        with low completion rates, then generates prioritised integration
        opportunities for the national platform.

        Returns
        -------
        list[IntegrationOpportunity]
            Sorted descending by ``priority_score``.
        """
        self.budget.consume(150, "identify_integration_opportunities")
        tasks = pm_tasks if pm_tasks is not None else self._pm_cache
        opportunities: list[IntegrationOpportunity] = []

        # --- Pattern 1: chronic overdue tasks ----------------------------------
        overdue = [t for t in tasks if t.is_overdue()]
        if overdue:
            opportunities.append(IntegrationOpportunity(
                id=str(uuid.uuid4()),
                category="auto_reschedule",
                description=(
                    f"{len(overdue)} PM tasks chronically overdue. "
                    "Automate escalation and reschedule via Corrigo Schedules API."
                ),
                estimated_impact=f"{len(overdue)} tasks/month recovered",
                effort_level=EffortLevel.MEDIUM,
                corrigo_endpoint="/Schedules",
                priority_score=round(min(10.0, len(overdue) * 1.5), 1),
            ))

        # --- Pattern 2: labour-time variance -----------------------------------
        variance_tasks = [
            t for t in tasks
            if t.actual_minutes is not None
            and t.estimated_minutes > 0
            and abs(t.actual_minutes - t.estimated_minutes) / t.estimated_minutes > 0.3
        ]
        if variance_tasks:
            avg_var = round(
                sum(
                    abs(t.actual_minutes - t.estimated_minutes) / t.estimated_minutes  # type: ignore[operator]
                    for t in variance_tasks
                )
                / len(variance_tasks)
                * 100,
                1,
            )
            opportunities.append(IntegrationOpportunity(
                id=str(uuid.uuid4()),
                category="labour_estimate_calibration",
                description=(
                    f"{len(variance_tasks)} tasks with >{30}% labour variance "
                    f"(avg {avg_var}%). Sync actuals back to Shaw Goals for "
                    "estimate refinement."
                ),
                estimated_impact=f"Reduce estimate error by ~{avg_var // 2}%",
                effort_level=EffortLevel.LOW,
                corrigo_endpoint="/WorkOrders",
                priority_score=round(min(10.0, avg_var / 10), 1),
            ))

        # --- Pattern 3: low-completion trades ----------------------------------
        trade_stats: dict[str, dict[str, int]] = {}
        for t in tasks:
            entry = trade_stats.setdefault(t.trade, {"total": 0, "done": 0})
            entry["total"] += 1
            if t.status == PMStatus.COMPLETED:
                entry["done"] += 1

        for trade, stats in trade_stats.items():
            pct = (stats["done"] / stats["total"] * 100) if stats["total"] else 100
            if pct < 70:
                opportunities.append(IntegrationOpportunity(
                    id=str(uuid.uuid4()),
                    category="trade_pm_gap",
                    description=(
                        f"Trade '{trade}' at {pct:.0f}% PM completion. "
                        "Automate WO creation and technician assignment."
                    ),
                    estimated_impact=f"Lift {trade} completion to >90%",
                    effort_level=EffortLevel.HIGH,
                    corrigo_endpoint="/WorkOrders",
                    priority_score=round(min(10.0, (100 - pct) / 5), 1),
                ))

        opportunities.sort(key=lambda o: o.priority_score, reverse=True)
        logger.info(
            "identify_integration_opportunities | found %d opportunities",
            len(opportunities),
        )
        return opportunities

    # ------------------------------------------------------------------
    # generate_goal_report
    # ------------------------------------------------------------------

    def generate_goal_report(
        self,
        alignments: list[GoalAlignment] | None = None,
        pm_tasks: list[PMTask] | None = None,
    ) -> dict[str, Any]:
        """Produce a structured goal-progress report.

        Parameters
        ----------
        alignments:
            Pre-computed alignments; if *None* the agent runs
            ``sync_corrigo_goals`` internally.
        pm_tasks:
            PM task data (passed through to sync if needed).

        Returns
        -------
        dict
            Compact report payload suitable for inter-agent messaging and
            Shaw Goals API ingestion.
        """
        self.budget.consume(130, "generate_goal_report")

        if alignments is None:
            alignments = self.sync_corrigo_goals(pm_tasks=pm_tasks)

        on_track = sum(1 for a in alignments if a.status == GoalStatus.ON_TRACK)
        at_risk = sum(1 for a in alignments if a.status == GoalStatus.AT_RISK)
        behind = sum(1 for a in alignments if a.status == GoalStatus.BEHIND)

        report: dict[str, Any] = {
            "campus": self.CAMPUS,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_goals": len(alignments),
                "on_track": on_track,
                "at_risk": at_risk,
                "behind": behind,
            },
            "goals": [a.to_dict() for a in alignments],
            "token_usage": self.budget.summary(),
        }
        logger.info(
            "generate_goal_report | goals=%d | on_track=%d at_risk=%d behind=%d",
            len(alignments),
            on_track,
            at_risk,
            behind,
        )
        return report

    # ------------------------------------------------------------------
    # calculate_ie_metrics
    # ------------------------------------------------------------------

    def calculate_ie_metrics(
        self,
        pm_tasks: list[PMTask] | None = None,
        period_days: int = 30,
    ) -> dict[str, Any]:
        """Compute industrial-engineering KPIs from Corrigo PM data.

        Metrics
        -------
        * **MTBF** - Mean Time Between Failures (hours).  Approximated from
          the interval between consecutive completed PMs per asset.
        * **MTTR** - Mean Time To Repair (minutes).  Average ``actual_minutes``
          of completed tasks.
        * **OEE** - Overall Equipment Effectiveness (%).  Simplified as
          ``availability * performance * quality`` where availability is PM
          compliance, performance is estimated-vs-actual ratio, and quality
          is assumed at 99 % for preventive work.
        * **PM Compliance** - Percentage of tasks completed on or before
          their due date within *period_days*.

        Returns
        -------
        dict
            ``{"mtbf_hours", "mttr_minutes", "oee_pct", "pm_compliance_pct", ...}``
        """
        self.budget.consume(140, "calculate_ie_metrics")
        tasks = pm_tasks if pm_tasks is not None else self._pm_cache
        cutoff = datetime.now() - timedelta(days=period_days)

        period_tasks = [
            t for t in tasks
            if t.last_completed is not None and t.last_completed >= cutoff
            or t.next_due >= cutoff
        ]

        # -- MTTR ---------------------------------------------------------------
        completed = [
            t for t in period_tasks
            if t.status == PMStatus.COMPLETED and t.actual_minutes is not None
        ]
        mttr = (
            round(sum(t.actual_minutes for t in completed) / len(completed), 1)  # type: ignore[arg-type]
            if completed
            else 0.0
        )

        # -- MTBF (per-asset intervals) ----------------------------------------
        asset_completions: dict[str, list[datetime]] = {}
        for t in tasks:
            if t.status == PMStatus.COMPLETED and t.last_completed:
                asset_completions.setdefault(t.asset_id, []).append(t.last_completed)

        intervals_hours: list[float] = []
        for dates in asset_completions.values():
            sorted_dates = sorted(dates)
            for i in range(1, len(sorted_dates)):
                delta = sorted_dates[i] - sorted_dates[i - 1]
                intervals_hours.append(delta.total_seconds() / 3600)

        mtbf = round(sum(intervals_hours) / len(intervals_hours), 1) if intervals_hours else 0.0

        # -- PM compliance ------------------------------------------------------
        due_in_period = [
            t for t in period_tasks
            if cutoff <= t.next_due <= datetime.now()
        ]
        on_time = [
            t for t in due_in_period
            if t.status == PMStatus.COMPLETED
            and t.last_completed is not None
            and t.last_completed <= t.next_due
        ]
        compliance = (
            round(len(on_time) / len(due_in_period) * 100, 1)
            if due_in_period
            else 100.0
        )

        # -- OEE (simplified) ---------------------------------------------------
        availability = compliance / 100
        performance = 1.0
        if completed:
            est_total = sum(t.estimated_minutes for t in completed)
            act_total = sum(t.actual_minutes for t in completed)  # type: ignore[arg-type]
            performance = min((est_total / act_total) if act_total else 1.0, 1.0)
        quality = 0.99  # assumed for preventive maintenance
        oee = round(availability * performance * quality * 100, 1)

        metrics: dict[str, Any] = {
            "period_days": period_days,
            "tasks_in_period": len(period_tasks),
            "mtbf_hours": mtbf,
            "mttr_minutes": mttr,
            "oee_pct": oee,
            "pm_compliance_pct": compliance,
            "availability": round(availability * 100, 1),
            "performance": round(performance * 100, 1),
            "quality": round(quality * 100, 1),
        }
        logger.info(
            "calculate_ie_metrics | MTBF=%.1fh MTTR=%.1fm OEE=%.1f%% compliance=%.1f%%",
            mtbf,
            mttr,
            oee,
            compliance,
        )
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_pm_tasks(self) -> list[PMTask]:
        """Fetch PM work orders from Corrigo and convert to ``PMTask`` list.

        In production, this queries ``/WorkOrders`` with a PM type filter.
        With placeholder credentials, it returns an empty list.
        """
        try:
            raw = self.corrigo.get_work_orders(
                filters={"$filter": "TypeCategory eq 'PM'"}
            )
        except Exception:
            logger.warning("Corrigo fetch failed; returning empty task list")
            return []

        tasks: list[PMTask] = []
        for wo in raw:
            tasks.append(PMTask(
                id=str(wo.get("Id", "")),
                asset_id=str(wo.get("AssetId", "")),
                trade=wo.get("Trade", "General"),
                building=wo.get("BuildingName", "Unknown"),
                floor=wo.get("Floor", ""),
                frequency=wo.get("ScheduleFrequency", "monthly"),
                last_completed=_parse_dt(wo.get("CompletedDate")),
                next_due=_parse_dt(wo.get("DueDate")) or datetime.now(),
                estimated_minutes=int(wo.get("EstimatedMinutes", 0)),
                actual_minutes=_safe_int(wo.get("ActualMinutes")),
                technician_id=str(wo.get("TechnicianId", "")),
                status=PMStatus(wo.get("Status", "open")),
            ))
        return tasks

    @staticmethod
    def _bucket_key(task: PMTask, group_by: str) -> str:
        if group_by == "technician":
            return task.technician_id
        if group_by == "trade":
            return task.trade
        if group_by == "building":
            return task.building
        # period: month of next_due
        return task.next_due.strftime("%Y-%m")

    def _compute_metric(self, metric: str, tasks: list[PMTask]) -> float:
        """Derive a numeric value for a named Corrigo metric."""
        if metric == "pm_completion_pct":
            total = len(tasks)
            done = sum(1 for t in tasks if t.status == PMStatus.COMPLETED)
            return round((done / total) * 100, 1) if total else 0.0

        if metric == "overdue_count":
            return float(sum(1 for t in tasks if t.is_overdue()))

        if metric == "avg_response_minutes":
            completed = [
                t for t in tasks
                if t.status == PMStatus.COMPLETED and t.actual_minutes is not None
            ]
            if not completed:
                return 0.0
            return round(
                sum(t.actual_minutes for t in completed) / len(completed), 1  # type: ignore[arg-type]
            )

        if metric == "pm_on_time_pct":
            due_tasks = [
                t for t in tasks
                if t.status == PMStatus.COMPLETED and t.last_completed is not None
            ]
            if not due_tasks:
                return 100.0
            on_time = sum(
                1 for t in due_tasks if t.last_completed <= t.next_due  # type: ignore[operator]
            )
            return round((on_time / len(due_tasks)) * 100, 1)

        logger.warning("Unknown metric '%s'; returning 0.0", metric)
        return 0.0

    @staticmethod
    def _default_goals() -> list[dict[str, Any]]:
        """BNSF campus default Shaw Goals."""
        return [
            {
                "goal_id": "SG-001",
                "goal_name": "PM Completion Rate",
                "corrigo_metric": "pm_completion_pct",
                "target_value": 95.0,
            },
            {
                "goal_id": "SG-002",
                "goal_name": "Zero Overdue PMs",
                "corrigo_metric": "overdue_count",
                "target_value": 0.0,
            },
            {
                "goal_id": "SG-003",
                "goal_name": "Avg Response Time",
                "corrigo_metric": "avg_response_minutes",
                "target_value": 45.0,
            },
            {
                "goal_id": "SG-004",
                "goal_name": "On-Time PM Delivery",
                "corrigo_metric": "pm_on_time_pct",
                "target_value": 90.0,
            },
        ]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_dt(val: Any) -> datetime | None:
    """Best-effort ISO datetime parse."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Sample data factory (for demos and tests)
# ---------------------------------------------------------------------------

def _build_sample_tasks() -> list[PMTask]:
    """Create a realistic set of PM tasks for the BNSF campus."""
    now = datetime.now()
    trades = ["HVAC", "Electrical", "Plumbing", "Fire/Life Safety", "Elevator"]
    buildings = [
        "BNSF HQ Tower",
        "South Operations Center",
        "North Maintenance Facility",
        "Parking Garage A",
    ]
    techs = ["T-101", "T-102", "T-103", "T-104", "T-105"]

    tasks: list[PMTask] = []
    for i in range(25):
        trade = trades[i % len(trades)]
        bldg = buildings[i % len(buildings)]
        tech = techs[i % len(techs)]
        freq = ["weekly", "monthly", "quarterly"][i % 3]
        est = [30, 45, 60, 90, 120][i % 5]
        status = [
            PMStatus.COMPLETED,
            PMStatus.COMPLETED,
            PMStatus.COMPLETED,
            PMStatus.IN_PROGRESS,
            PMStatus.OVERDUE,
        ][i % 5]
        last = now - timedelta(days=(i * 3 + 1)) if status == PMStatus.COMPLETED else None
        due = now - timedelta(days=(i % 4 - 2))
        actual = (
            est + ((-1) ** i) * (i * 2)
            if status == PMStatus.COMPLETED
            else None
        )

        tasks.append(PMTask(
            id=f"WO-{10000 + i}",
            asset_id=f"AST-{2000 + i}",
            trade=trade,
            building=bldg,
            floor=f"Floor {(i % 5) + 1}",
            frequency=freq,
            last_completed=last,
            next_due=due,
            estimated_minutes=est,
            actual_minutes=actual,
            technician_id=tech,
            status=status,
        ))
    return tasks


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a self-contained demonstration of the Shaw Goals Agent."""
    logger.setLevel(logging.INFO)
    logger.info("=" * 60)
    logger.info("Shaw Goals Agent - Demo Run")
    logger.info("Campus: BNSF Railway - Fort Worth, TX")
    logger.info("=" * 60)

    agent = ShawGoalsAgent()
    sample_tasks = _build_sample_tasks()

    # 1 - Sync goals
    print("\n--- 1. Sync Corrigo Goals ---")
    alignments = agent.sync_corrigo_goals(pm_tasks=sample_tasks)
    for a in alignments:
        print(json.dumps(a.to_dict(), indent=2))

    # 2 - Evaluate PM completion
    print("\n--- 2. PM Completion by Trade ---")
    trade_report = agent.evaluate_pm_completion(pm_tasks=sample_tasks, group_by="trade")
    print(json.dumps(trade_report, indent=2))

    print("\n--- 2b. PM Completion by Technician ---")
    tech_report = agent.evaluate_pm_completion(pm_tasks=sample_tasks, group_by="technician")
    print(json.dumps(tech_report, indent=2))

    # 3 - Integration opportunities
    print("\n--- 3. Integration Opportunities ---")
    opportunities = agent.identify_integration_opportunities(pm_tasks=sample_tasks)
    for opp in opportunities:
        print(json.dumps(opp.to_dict(), indent=2))

    # 4 - Goal report
    print("\n--- 4. Goal Report ---")
    report = agent.generate_goal_report(alignments=alignments, pm_tasks=sample_tasks)
    print(json.dumps(report, indent=2))

    # 5 - IE Metrics
    print("\n--- 5. IE Metrics ---")
    ie = agent.calculate_ie_metrics(pm_tasks=sample_tasks, period_days=30)
    print(json.dumps(ie, indent=2))

    # 6 - Token budget summary
    print("\n--- 6. Token Budget ---")
    print(json.dumps(agent.budget.summary(), indent=2))

    logger.info("Demo complete.")


if __name__ == "__main__":
    main()

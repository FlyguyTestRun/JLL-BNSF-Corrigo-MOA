"""
Morning Identification & TODO List Agent
=========================================

Multi-Agent Orchestration (MAO) module for the JLL-managed BNSF Railway campus
at 2400 Lou Menk Dr, Fort Worth, TX 76131.

Every morning this agent:
  1. Pulls all open/pending work orders and PM tasks from Corrigo.
  2. Prioritizes them using a weighted scoring model.
  3. Generates an optimized, time-blocked technician schedule with campus
     routing maps and estimated completion times.
  4. Provides a management dashboard (Tony Vita, Juan Guerra) and individual
     technician views.
  5. Tracks actuals throughout the day and produces an end-of-day variance
     report that feeds back into the estimation model.

Token budget: every LLM call is capped at 500 tokens to conserve costs.

Python 3.11+ | dataclasses | logging
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import textwrap
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("mao.morning_todo_agent")
logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
if not logger.handlers:
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_LLM_TOKENS = 500
WORKDAY_START = time(7, 0)
WORKDAY_END = time(17, 0)
WORKDAY_HOURS = 10.0

SITE_ADDRESS = "2400 Lou Menk Dr, Fort Worth, TX 76131"
MANAGEMENT_CONTACTS: dict[str, str] = {
    "Tony Vita": "Facilities Director",
    "Juan Guerra": "Operations Manager",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class PriorityLevel(Enum):
    """Work-order priority tiers with base weights and SLA windows."""

    P1_EMERGENCY = ("P1", 100, timedelta(hours=0))
    P2_URGENT = ("P2", 80, timedelta(hours=4))
    P3_HIGH = ("P3", 60, timedelta(hours=8))
    P4_MEDIUM = ("P4", 40, timedelta(hours=24))
    P5_LOW = ("P5", 20, timedelta(hours=72))

    def __init__(self, code: str, weight: int, sla_window: timedelta) -> None:
        self.code = code
        self.weight = weight
        self.sla_window = sla_window


class TaskStatus(Enum):
    """Lifecycle states for a daily task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class Confidence(Enum):
    """Estimation confidence levels."""

    HIGH = "high"
    MEDIUM = "med"
    LOW = "low"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class CompletionEstimate:
    """Time estimate envelope for a single task."""

    task_id: str
    estimated_start: datetime
    estimated_duration_min: int
    estimated_end: datetime
    confidence: Confidence = Confidence.MEDIUM
    basis: str = "standard"  # historical | standard | manual

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "estimated_start": self.estimated_start.isoformat(),
            "estimated_duration_min": self.estimated_duration_min,
            "estimated_end": self.estimated_end.isoformat(),
            "confidence": self.confidence.value,
            "basis": self.basis,
        }


@dataclass
class DailyTask:
    """A single actionable task for the day, sourced from a Corrigo WO or PM."""

    task_id: str
    wo_number: str
    title: str
    description: str
    building: str
    floor: int
    zone: str
    trade: str
    priority_score: float = 0.0
    sla_deadline: datetime | None = None
    estimated_minutes: int = 30
    assigned_tech: str | None = None
    sequence_order: int = 0
    status: TaskStatus = TaskStatus.PENDING
    parts_needed: list[str] = field(default_factory=list)
    tools_needed: list[str] = field(default_factory=list)
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    estimate: CompletionEstimate | None = None

    # -- priority weighting components (populated by prioritize_tasks) ------
    sla_urgency_score: float = 0.0
    safety_impact_score: float = 0.0
    occupant_impact_score: float = 0.0
    efficiency_group_score: float = 0.0
    asset_criticality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "wo_number": self.wo_number,
            "title": self.title,
            "description": self.description,
            "building": self.building,
            "floor": self.floor,
            "zone": self.zone,
            "trade": self.trade,
            "priority_score": round(self.priority_score, 2),
            "sla_deadline": self.sla_deadline.isoformat() if self.sla_deadline else None,
            "estimated_minutes": self.estimated_minutes,
            "assigned_tech": self.assigned_tech,
            "sequence_order": self.sequence_order,
            "status": self.status.value,
            "parts_needed": self.parts_needed,
            "tools_needed": self.tools_needed,
            "estimate": self.estimate.to_dict() if self.estimate else None,
        }


@dataclass
class TechSchedule:
    """A technician's full daily schedule."""

    tech_id: str
    tech_name: str
    date: date
    tasks: list[DailyTask] = field(default_factory=list)
    total_estimated_hours: float = 0.0
    route_sequence: list[str] = field(default_factory=list)
    start_time: time = WORKDAY_START
    projected_end_time: time = WORKDAY_END

    def to_dict(self) -> dict[str, Any]:
        return {
            "tech_id": self.tech_id,
            "tech_name": self.tech_name,
            "date": self.date.isoformat(),
            "tasks": [t.to_dict() for t in self.tasks],
            "total_estimated_hours": round(self.total_estimated_hours, 2),
            "route_sequence": self.route_sequence,
            "start_time": self.start_time.isoformat(),
            "projected_end_time": self.projected_end_time.isoformat(),
        }


@dataclass
class ManagementDashboard:
    """High-level summary for Tony Vita and Juan Guerra."""

    date: date
    total_open_wo: int = 0
    total_pm_due: int = 0
    tasks_by_priority: dict[str, int] = field(default_factory=dict)
    tasks_by_trade: dict[str, int] = field(default_factory=dict)
    staffing: dict[str, int] = field(default_factory=dict)  # tech_id -> task_count
    risk_flags: list[str] = field(default_factory=list)
    projected_completion_pct: float = 0.0
    carryover_from_yesterday: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "total_open_wo": self.total_open_wo,
            "total_pm_due": self.total_pm_due,
            "tasks_by_priority": self.tasks_by_priority,
            "tasks_by_trade": self.tasks_by_trade,
            "staffing": self.staffing,
            "risk_flags": self.risk_flags,
            "projected_completion_pct": round(self.projected_completion_pct, 1),
            "carryover_from_yesterday": self.carryover_from_yesterday,
        }


@dataclass
class CampusZone:
    """A logical zone of the BNSF Fort Worth campus."""

    zone_id: str
    zone_name: str
    buildings: list[str] = field(default_factory=list)
    avg_travel_minutes_between: dict[str, float] = field(default_factory=dict)
    floor_count: int = 1
    typical_trades: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Campus Model -- BNSF 2400 Lou Menk Dr
# ---------------------------------------------------------------------------
def _build_campus_zones() -> dict[str, CampusZone]:
    """Return the canonical campus-zone map with inter-zone travel times."""

    zones: dict[str, CampusZone] = {
        "A": CampusZone(
            zone_id="A",
            zone_name="Main HQ Building",
            buildings=["HQ-Main", "HQ-East Wing", "HQ-West Wing"],
            floor_count=4,
            typical_trades=["HVAC", "electrical", "plumbing", "general maintenance"],
        ),
        "B": CampusZone(
            zone_id="B",
            zone_name="Operations Center",
            buildings=["OPS-Center", "OPS-Annex"],
            floor_count=2,
            typical_trades=["electrical", "HVAC", "critical systems"],
        ),
        "C": CampusZone(
            zone_id="C",
            zone_name="Data Center / Server Rooms",
            buildings=["DC-Primary", "DC-Secondary"],
            floor_count=1,
            typical_trades=["HVAC", "electrical", "fire suppression", "UPS"],
        ),
        "D": CampusZone(
            zone_id="D",
            zone_name="Maintenance & Shops",
            buildings=["Shop-Main", "Shop-Storage"],
            floor_count=1,
            typical_trades=["general", "welding", "equipment"],
        ),
        "E": CampusZone(
            zone_id="E",
            zone_name="Parking Structures",
            buildings=["Parking-North", "Parking-South"],
            floor_count=5,
            typical_trades=["lighting", "elevator", "general"],
        ),
        "F": CampusZone(
            zone_id="F",
            zone_name="Grounds & Exterior",
            buildings=["Grounds-North", "Grounds-South", "Grounds-Perimeter"],
            floor_count=1,
            typical_trades=["landscaping", "paving", "fencing", "signage"],
        ),
        "G": CampusZone(
            zone_id="G",
            zone_name="Cafeteria & Common Areas",
            buildings=["Cafeteria", "Breakroom-East"],
            floor_count=1,
            typical_trades=["plumbing", "HVAC", "kitchen equipment"],
        ),
    }

    # Symmetric travel-time matrix (minutes, walking)
    _travel: dict[tuple[str, str], float] = {
        ("A", "B"): 3,
        ("A", "C"): 5,
        ("A", "D"): 6,
        ("A", "E"): 4,
        ("A", "F"): 7,
        ("A", "G"): 2,
        ("B", "C"): 3,
        ("B", "D"): 5,
        ("B", "E"): 5,
        ("B", "F"): 8,
        ("B", "G"): 4,
        ("C", "D"): 4,
        ("C", "E"): 6,
        ("C", "F"): 8,
        ("C", "G"): 5,
        ("D", "E"): 3,
        ("D", "F"): 4,
        ("D", "G"): 6,
        ("E", "F"): 3,
        ("E", "G"): 5,
        ("F", "G"): 7,
    }

    for (z1, z2), minutes in _travel.items():
        zones[z1].avg_travel_minutes_between[z2] = minutes
        zones[z2].avg_travel_minutes_between[z1] = minutes

    # Self-travel (within zone) defaults to 0
    for zid in zones:
        zones[zid].avg_travel_minutes_between[zid] = 0.0

    return zones


CAMPUS_ZONES: dict[str, CampusZone] = _build_campus_zones()


def travel_minutes(zone_a: str, zone_b: str) -> float:
    """Look up walking travel time between two zone IDs."""
    if zone_a == zone_b:
        return 0.0
    zone_obj = CAMPUS_ZONES.get(zone_a)
    if zone_obj is None:
        return 5.0  # fallback average
    return zone_obj.avg_travel_minutes_between.get(zone_b, 5.0)


# ---------------------------------------------------------------------------
# Priority Weighting Helpers
# ---------------------------------------------------------------------------
# Weight factors (must sum to 1.0)
W_SLA_URGENCY = 0.40
W_SAFETY_IMPACT = 0.25
W_OCCUPANT_IMPACT = 0.15
W_EFFICIENCY_GROUP = 0.10
W_ASSET_CRITICALITY = 0.10

# Keywords for safety-impact heuristic
_SAFETY_KEYWORDS: set[str] = {
    "fire", "flood", "gas", "leak", "smoke", "sparking", "arc", "electrical hazard",
    "trip hazard", "mold", "asbestos", "chemical", "emergency", "injury",
}

# Keywords for high occupant-impact
_OCCUPANT_KEYWORDS: set[str] = {
    "no heat", "no cooling", "no ac", "no power", "elevator stuck",
    "restroom", "water outage", "odor", "noise", "temperature",
}

# Critical-asset zones/trades
_CRITICAL_ZONES: set[str] = {"B", "C"}
_CRITICAL_TRADES: set[str] = {"electrical", "HVAC", "fire suppression", "UPS", "critical systems"}


def _score_sla_urgency(task: DailyTask, now: datetime) -> float:
    """0-100 score based on how close the SLA deadline is."""
    if task.sla_deadline is None:
        return 30.0
    remaining = (task.sla_deadline - now).total_seconds() / 3600.0
    if remaining <= 0:
        return 100.0
    if remaining <= 1:
        return 95.0
    if remaining <= 4:
        return 80.0
    if remaining <= 8:
        return 60.0
    if remaining <= 24:
        return 40.0
    return 20.0


def _score_safety_impact(task: DailyTask) -> float:
    """0-100 score based on presence of safety-related keywords."""
    text = f"{task.title} {task.description}".lower()
    hits = sum(1 for kw in _SAFETY_KEYWORDS if kw in text)
    return min(100.0, hits * 35.0) if hits else 0.0


def _score_occupant_impact(task: DailyTask) -> float:
    """0-100 score for occupant comfort / usability impact."""
    text = f"{task.title} {task.description}".lower()
    hits = sum(1 for kw in _OCCUPANT_KEYWORDS if kw in text)
    return min(100.0, hits * 30.0) if hits else 0.0


def _score_efficiency_group(task: DailyTask, all_tasks: list[DailyTask]) -> float:
    """0-100: bonus when other tasks share the same building or zone."""
    same_zone = sum(1 for t in all_tasks if t.zone == task.zone and t.task_id != task.task_id)
    return min(100.0, same_zone * 20.0)


def _score_asset_criticality(task: DailyTask) -> float:
    """0-100 score for critical infrastructure zones / trades."""
    score = 0.0
    if task.zone in _CRITICAL_ZONES:
        score += 50.0
    if task.trade.lower() in {t.lower() for t in _CRITICAL_TRADES}:
        score += 50.0
    return min(100.0, score)


def _pm_priority_score(overdue_days: int) -> float:
    """PM tasks: base 30 + 5 per day overdue."""
    return 30.0 + 5.0 * max(overdue_days, 0)


# ---------------------------------------------------------------------------
# Routing: nearest-neighbour + zone clustering
# ---------------------------------------------------------------------------
def _nearest_neighbour_route(zones_to_visit: list[str], start_zone: str = "D") -> list[str]:
    """
    Greedy nearest-neighbour route through campus zones.

    Technicians typically start at Zone D (Maintenance & Shops), so we default
    the origin there.
    """
    if not zones_to_visit:
        return []

    remaining = list(set(zones_to_visit))
    route: list[str] = []
    current = start_zone if start_zone in remaining else remaining[0]

    while remaining:
        if current in remaining:
            remaining.remove(current)
        route.append(current)
        if not remaining:
            break
        next_zone = min(remaining, key=lambda z: travel_minutes(current, z))
        current = next_zone

    return route


def _cluster_by_zone(tasks: list[DailyTask]) -> dict[str, list[DailyTask]]:
    """Group tasks by zone for routing efficiency."""
    clusters: dict[str, list[DailyTask]] = {}
    for t in tasks:
        clusters.setdefault(t.zone, []).append(t)
    return clusters


# ---------------------------------------------------------------------------
# Main Agent Class
# ---------------------------------------------------------------------------
class MorningTodoAgent:
    """
    Morning Identification & TODO List Agent.

    Orchestrates the daily planning cycle for the BNSF campus:
    pull -> prioritize -> schedule -> route -> estimate -> present.

    All LLM interactions are budgeted at ``MAX_LLM_TOKENS`` (500) tokens.
    """

    def __init__(
        self,
        corrigo_client: Any | None = None,
        llm_client: Any | None = None,
        target_date: date | None = None,
    ) -> None:
        self.corrigo = corrigo_client
        self.llm = llm_client
        self.target_date: date = target_date or date.today()
        self.tasks: list[DailyTask] = []
        self.tech_schedules: dict[str, TechSchedule] = {}
        self.dashboard: ManagementDashboard | None = None
        self._now: datetime = datetime.combine(self.target_date, WORKDAY_START)
        logger.info(
            "MorningTodoAgent initialised for %s at %s",
            self.target_date.isoformat(),
            SITE_ADDRESS,
        )

    # ------------------------------------------------------------------
    # 1. Pull daily workload
    # ------------------------------------------------------------------
    def pull_daily_workload(
        self,
        raw_work_orders: list[dict[str, Any]] | None = None,
    ) -> list[DailyTask]:
        """
        Fetch all open WOs, scheduled PMs, and carryover tasks from the
        Corrigo API for *target_date*.

        If *raw_work_orders* is supplied (list of dicts), use those instead
        of calling the Corrigo client -- useful for testing and offline mode.

        Returns the list of ``DailyTask`` objects stored in ``self.tasks``.
        """
        logger.info("Pulling daily workload for %s ...", self.target_date)

        if raw_work_orders is not None:
            work_orders = raw_work_orders
        elif self.corrigo is not None:
            # Real integration placeholder -- call Corrigo REST API
            work_orders = self.corrigo.get_open_work_orders(self.target_date)  # type: ignore[union-attr]
        else:
            logger.warning("No Corrigo client and no raw data provided; returning empty task list.")
            return []

        self.tasks = []
        for wo in work_orders:
            sla_deadline_raw = wo.get("sla_deadline")
            if isinstance(sla_deadline_raw, str):
                sla_deadline = datetime.fromisoformat(sla_deadline_raw)
            elif isinstance(sla_deadline_raw, datetime):
                sla_deadline = sla_deadline_raw
            else:
                sla_deadline = None

            task = DailyTask(
                task_id=str(wo.get("task_id", wo.get("wo_number", ""))),
                wo_number=str(wo.get("wo_number", "")),
                title=str(wo.get("title", "")),
                description=str(wo.get("description", "")),
                building=str(wo.get("building", "")),
                floor=int(wo.get("floor", 1)),
                zone=str(wo.get("zone", "A")),
                trade=str(wo.get("trade", "general maintenance")),
                sla_deadline=sla_deadline,
                estimated_minutes=int(wo.get("estimated_minutes", 30)),
                parts_needed=wo.get("parts_needed", []),
                tools_needed=wo.get("tools_needed", []),
                status=TaskStatus(wo.get("status", "pending")),
            )
            self.tasks.append(task)

        logger.info("Loaded %d tasks.", len(self.tasks))
        return self.tasks

    # ------------------------------------------------------------------
    # 2. Prioritize tasks
    # ------------------------------------------------------------------
    def prioritize_tasks(self) -> list[DailyTask]:
        """
        Score and rank every task using the weighted-criteria model:

        - SLA urgency          40 %
        - Safety impact         25 %
        - Occupant impact       15 %
        - Efficiency grouping   10 %
        - Asset criticality     10 %

        PM tasks that are overdue receive a boosted base score of
        ``30 + 5 * overdue_days``.

        Updates ``priority_score`` on each task **in place** and sorts
        ``self.tasks`` descending by score.
        """
        logger.info("Prioritizing %d tasks ...", len(self.tasks))

        for task in self.tasks:
            task.sla_urgency_score = _score_sla_urgency(task, self._now)
            task.safety_impact_score = _score_safety_impact(task)
            task.occupant_impact_score = _score_occupant_impact(task)
            task.efficiency_group_score = _score_efficiency_group(task, self.tasks)
            task.asset_criticality_score = _score_asset_criticality(task)

            task.priority_score = (
                W_SLA_URGENCY * task.sla_urgency_score
                + W_SAFETY_IMPACT * task.safety_impact_score
                + W_OCCUPANT_IMPACT * task.occupant_impact_score
                + W_EFFICIENCY_GROUP * task.efficiency_group_score
                + W_ASSET_CRITICALITY * task.asset_criticality_score
            )

        self.tasks.sort(key=lambda t: t.priority_score, reverse=True)

        for idx, task in enumerate(self.tasks, start=1):
            task.sequence_order = idx
            logger.debug(
                "  #%d  score=%.1f  %s  [%s / Zone %s]",
                idx,
                task.priority_score,
                task.title,
                task.trade,
                task.zone,
            )

        return self.tasks

    # ------------------------------------------------------------------
    # 3. Generate technician schedules
    # ------------------------------------------------------------------
    def generate_technician_schedule(
        self,
        technicians: list[dict[str, Any]],
    ) -> dict[str, TechSchedule]:
        """
        For each technician, create a time-blocked daily schedule.

        *technicians* is a list of dicts with at least:
          ``{"tech_id": str, "tech_name": str, "trades": list[str]}``

        Assignment strategy:
        - Walk the priority-sorted task list.
        - Assign each task to the first tech whose trade list overlaps and
          who has remaining capacity.
        - After assignment, build the route and time blocks.

        Returns ``self.tech_schedules``.
        """
        logger.info("Generating schedules for %d technicians ...", len(technicians))

        # Initialise empty schedules
        self.tech_schedules = {}
        tech_meta: dict[str, dict[str, Any]] = {}
        for t in technicians:
            tid = t["tech_id"]
            sched = TechSchedule(
                tech_id=tid,
                tech_name=t["tech_name"],
                date=self.target_date,
                start_time=WORKDAY_START,
            )
            self.tech_schedules[tid] = sched
            tech_meta[tid] = {
                "trades": {tr.lower() for tr in t.get("trades", [])},
                "remaining_min": WORKDAY_HOURS * 60,
            }

        # Assign tasks in priority order
        for task in self.tasks:
            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                continue

            best_tech: str | None = None
            best_remaining: float = -1.0

            for tid, meta in tech_meta.items():
                trade_match = task.trade.lower() in meta["trades"] or "general maintenance" in meta["trades"]
                if trade_match and meta["remaining_min"] >= task.estimated_minutes:
                    if meta["remaining_min"] > best_remaining:
                        best_remaining = meta["remaining_min"]
                        best_tech = tid

            if best_tech is not None:
                task.assigned_tech = best_tech
                task.status = TaskStatus.ASSIGNED
                self.tech_schedules[best_tech].tasks.append(task)
                tech_meta[best_tech]["remaining_min"] -= task.estimated_minutes
            else:
                logger.warning(
                    "No eligible technician for task %s (%s / %s). Leaving unassigned.",
                    task.task_id,
                    task.trade,
                    task.title,
                )

        # Build routes and time estimates per tech
        for tid, sched in self.tech_schedules.items():
            sched.route_sequence = self.map_campus_route(sched.tasks)
            self._time_block_schedule(sched)

        return self.tech_schedules

    def _time_block_schedule(self, sched: TechSchedule) -> None:
        """Fill in start/end estimates for every task in a schedule."""
        cursor = datetime.combine(sched.date, sched.start_time)
        total_min = 0.0
        ordered_tasks = self._order_tasks_by_route(sched.tasks, sched.route_sequence)
        prev_zone: str | None = "D"  # techs start at Maintenance & Shops

        for task in ordered_tasks:
            # Add travel time
            if prev_zone is not None:
                travel = travel_minutes(prev_zone, task.zone)
                cursor += timedelta(minutes=travel)
                total_min += travel

            est_start = cursor
            est_end = cursor + timedelta(minutes=task.estimated_minutes)
            task.estimate = CompletionEstimate(
                task_id=task.task_id,
                estimated_start=est_start,
                estimated_duration_min=task.estimated_minutes,
                estimated_end=est_end,
                confidence=self._estimate_confidence(task),
                basis="standard",
            )
            cursor = est_end
            total_min += task.estimated_minutes
            prev_zone = task.zone

        sched.total_estimated_hours = round(total_min / 60.0, 2)
        sched.projected_end_time = cursor.time()
        sched.tasks = ordered_tasks

    @staticmethod
    def _order_tasks_by_route(
        tasks: list[DailyTask],
        route: list[str],
    ) -> list[DailyTask]:
        """Re-order tasks to match the optimised route sequence."""
        zone_order = {z: i for i, z in enumerate(route)}
        return sorted(
            tasks,
            key=lambda t: (zone_order.get(t.zone, 999), -t.priority_score),
        )

    @staticmethod
    def _estimate_confidence(task: DailyTask) -> Confidence:
        """Heuristic confidence level for a time estimate."""
        if task.estimated_minutes <= 15:
            return Confidence.HIGH
        if task.estimated_minutes <= 60:
            return Confidence.MEDIUM
        return Confidence.LOW

    # ------------------------------------------------------------------
    # 4. Campus routing
    # ------------------------------------------------------------------
    def map_campus_route(self, tasks: list[DailyTask]) -> list[str]:
        """
        Generate an optimal walking/driving route through campus buildings.

        Uses **nearest-neighbour with zone clustering**: first cluster tasks
        by zone, then order zones via nearest-neighbour from Zone D (shops).

        Returns an ordered list of zone IDs.
        """
        if not tasks:
            return []

        clusters = _cluster_by_zone(tasks)
        zones_needed = list(clusters.keys())
        route = _nearest_neighbour_route(zones_needed, start_zone="D")

        total_travel = 0.0
        for i in range(1, len(route)):
            total_travel += travel_minutes(route[i - 1], route[i])

        logger.info(
            "Route for %d tasks across %d zones: %s (%.0f min travel)",
            len(tasks),
            len(route),
            " -> ".join(route),
            total_travel,
        )
        return route

    # ------------------------------------------------------------------
    # 5. Completion-time estimates
    # ------------------------------------------------------------------
    def estimate_completion_times(self) -> list[CompletionEstimate]:
        """
        For every task, provide estimated start, duration, end, confidence,
        and basis.  (Already computed during schedule generation -- this
        method is a convenience accessor.)

        Returns a flat list of ``CompletionEstimate`` across all techs.
        """
        estimates: list[CompletionEstimate] = []
        for sched in self.tech_schedules.values():
            for task in sched.tasks:
                if task.estimate is not None:
                    estimates.append(task.estimate)
        return estimates

    # ------------------------------------------------------------------
    # 6. Management view
    # ------------------------------------------------------------------
    def generate_management_view(
        self,
        carryover_count: int = 0,
    ) -> ManagementDashboard:
        """
        Summary view for Tony Vita & Juan Guerra.

        Includes: total tasks, by-priority breakdown, staffing coverage,
        risk flags, expected completion percentage.
        """
        logger.info("Building management dashboard ...")

        # Aggregate by priority tier
        by_priority: dict[str, int] = {}
        by_trade: dict[str, int] = {}
        pm_count = 0
        wo_count = 0

        for task in self.tasks:
            # Derive a human-readable priority bucket
            if task.priority_score >= 80:
                bucket = "P1-Emergency"
            elif task.priority_score >= 60:
                bucket = "P2-Urgent"
            elif task.priority_score >= 40:
                bucket = "P3-High"
            elif task.priority_score >= 25:
                bucket = "P4-Medium"
            else:
                bucket = "P5-Low"
            by_priority[bucket] = by_priority.get(bucket, 0) + 1
            by_trade[task.trade] = by_trade.get(task.trade, 0) + 1

            if task.wo_number.upper().startswith("PM"):
                pm_count += 1
            else:
                wo_count += 1

        # Staffing map
        staffing: dict[str, int] = {}
        for tid, sched in self.tech_schedules.items():
            staffing[tid] = len(sched.tasks)

        # Risk flags
        risk_flags: list[str] = []
        unassigned = [t for t in self.tasks if t.assigned_tech is None and t.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)]
        if unassigned:
            risk_flags.append(f"{len(unassigned)} task(s) unassigned -- insufficient trade coverage or capacity.")
        overloaded = [
            tid for tid, sched in self.tech_schedules.items()
            if sched.total_estimated_hours > WORKDAY_HOURS * 0.9
        ]
        if overloaded:
            risk_flags.append(f"Technician(s) {', '.join(overloaded)} at >90% capacity.")
        p1_tasks = [t for t in self.tasks if t.priority_score >= 80]
        if p1_tasks:
            risk_flags.append(f"{len(p1_tasks)} emergency/urgent task(s) require immediate attention.")

        # Projected completion %
        total = len(self.tasks)
        assigned = sum(1 for t in self.tasks if t.assigned_tech is not None)
        projected_pct = (assigned / total * 100.0) if total else 100.0

        self.dashboard = ManagementDashboard(
            date=self.target_date,
            total_open_wo=wo_count,
            total_pm_due=pm_count,
            tasks_by_priority=by_priority,
            tasks_by_trade=by_trade,
            staffing=staffing,
            risk_flags=risk_flags,
            projected_completion_pct=projected_pct,
            carryover_from_yesterday=carryover_count,
        )

        logger.info(
            "Dashboard: %d WOs, %d PMs, projected %.0f%% completion",
            wo_count,
            pm_count,
            projected_pct,
        )
        return self.dashboard

    # ------------------------------------------------------------------
    # 7. Technician view
    # ------------------------------------------------------------------
    def generate_tech_view(self, tech_id: str) -> dict[str, Any]:
        """
        Individual technician view: task list in route order, map, time
        estimates, parts/tools needed.
        """
        sched = self.tech_schedules.get(tech_id)
        if sched is None:
            logger.error("No schedule found for tech %s", tech_id)
            return {"error": f"No schedule found for tech_id={tech_id}"}

        view: dict[str, Any] = {
            "tech_id": sched.tech_id,
            "tech_name": sched.tech_name,
            "date": sched.date.isoformat(),
            "start_time": sched.start_time.isoformat(),
            "projected_end_time": sched.projected_end_time.isoformat(),
            "total_estimated_hours": sched.total_estimated_hours,
            "route_sequence": sched.route_sequence,
            "route_description": self._describe_route(sched.route_sequence),
            "tasks": [],
        }

        for task in sched.tasks:
            task_view: dict[str, Any] = {
                "seq": task.sequence_order,
                "wo_number": task.wo_number,
                "title": task.title,
                "building": task.building,
                "floor": task.floor,
                "zone": task.zone,
                "zone_name": CAMPUS_ZONES[task.zone].zone_name if task.zone in CAMPUS_ZONES else task.zone,
                "trade": task.trade,
                "estimated_minutes": task.estimated_minutes,
                "parts_needed": task.parts_needed,
                "tools_needed": task.tools_needed,
                "status": task.status.value,
            }
            if task.estimate:
                task_view["estimated_start"] = task.estimate.estimated_start.strftime("%H:%M")
                task_view["estimated_end"] = task.estimate.estimated_end.strftime("%H:%M")
                task_view["confidence"] = task.estimate.confidence.value
            view["tasks"].append(task_view)

        return view

    @staticmethod
    def _describe_route(route: list[str]) -> str:
        """Human-readable route description."""
        if not route:
            return "No tasks assigned."
        names = []
        for zid in route:
            zone = CAMPUS_ZONES.get(zid)
            names.append(f"Zone {zid} ({zone.zone_name})" if zone else f"Zone {zid}")
        return " -> ".join(names)

    # ------------------------------------------------------------------
    # 8. Track actuals
    # ------------------------------------------------------------------
    def track_actuals(
        self,
        task_id: str,
        actual_start: datetime | None = None,
        actual_end: datetime | None = None,
    ) -> None:
        """
        As tasks complete during the day, update estimates and re-optimise
        the remaining schedule for the assigned technician.
        """
        target: DailyTask | None = None
        for task in self.tasks:
            if task.task_id == task_id:
                target = task
                break

        if target is None:
            logger.error("track_actuals: task_id=%s not found.", task_id)
            return

        if actual_start:
            target.actual_start = actual_start
            target.status = TaskStatus.IN_PROGRESS
            logger.info("Task %s started at %s", task_id, actual_start)

        if actual_end:
            target.actual_end = actual_end
            target.status = TaskStatus.COMPLETED
            logger.info("Task %s completed at %s", task_id, actual_end)

        # Re-optimise the tech's remaining schedule
        if target.assigned_tech and target.assigned_tech in self.tech_schedules:
            sched = self.tech_schedules[target.assigned_tech]
            remaining = [t for t in sched.tasks if t.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)]
            if remaining:
                cursor_time = actual_end if actual_end else datetime.now()
                self._reoptimise_remaining(sched, remaining, cursor_time)

    def _reoptimise_remaining(
        self,
        sched: TechSchedule,
        remaining: list[DailyTask],
        cursor: datetime,
    ) -> None:
        """Recalculate times for the remaining tasks after actuals update."""
        route = _nearest_neighbour_route(
            [t.zone for t in remaining],
            start_zone=remaining[0].zone,
        )
        ordered = self._order_tasks_by_route(remaining, route)

        prev_zone = ordered[0].zone if ordered else "D"
        total_min = 0.0
        for task in ordered:
            travel = travel_minutes(prev_zone, task.zone)
            cursor += timedelta(minutes=travel)
            total_min += travel

            task.estimate = CompletionEstimate(
                task_id=task.task_id,
                estimated_start=cursor,
                estimated_duration_min=task.estimated_minutes,
                estimated_end=cursor + timedelta(minutes=task.estimated_minutes),
                confidence=self._estimate_confidence(task),
                basis="standard",
            )
            cursor += timedelta(minutes=task.estimated_minutes)
            total_min += task.estimated_minutes
            prev_zone = task.zone

        sched.projected_end_time = cursor.time()
        logger.info(
            "Re-optimised %s: %d remaining tasks, new projected end %s",
            sched.tech_name,
            len(ordered),
            sched.projected_end_time,
        )

    # ------------------------------------------------------------------
    # 9. End-of-day summary
    # ------------------------------------------------------------------
    def end_of_day_summary(self) -> dict[str, Any]:
        """
        Compare planned vs actual, calculate accuracy metrics, and produce
        a feedback payload for the estimation model.
        """
        logger.info("Generating end-of-day summary ...")

        completed: list[DailyTask] = []
        deferred: list[DailyTask] = []
        variance_minutes: list[float] = []

        for task in self.tasks:
            if task.status == TaskStatus.COMPLETED:
                completed.append(task)
                if task.actual_start and task.actual_end and task.estimate:
                    actual_dur = (task.actual_end - task.actual_start).total_seconds() / 60.0
                    planned_dur = task.estimate.estimated_duration_min
                    variance_minutes.append(actual_dur - planned_dur)
            elif task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                deferred.append(task)

        avg_variance = (sum(variance_minutes) / len(variance_minutes)) if variance_minutes else 0.0
        abs_variances = [abs(v) for v in variance_minutes]
        mae = (sum(abs_variances) / len(abs_variances)) if abs_variances else 0.0
        accuracy_pct = (
            (len(completed) / len(self.tasks) * 100.0) if self.tasks else 100.0
        )

        summary: dict[str, Any] = {
            "date": self.target_date.isoformat(),
            "total_planned": len(self.tasks),
            "total_completed": len(completed),
            "total_deferred": len(deferred),
            "completion_rate_pct": round(accuracy_pct, 1),
            "avg_variance_minutes": round(avg_variance, 1),
            "mean_absolute_error_minutes": round(mae, 1),
            "deferred_tasks": [
                {"task_id": t.task_id, "wo_number": t.wo_number, "title": t.title}
                for t in deferred
            ],
            "estimation_feedback": {
                "sample_size": len(variance_minutes),
                "avg_over_under_min": round(avg_variance, 1),
                "mae_min": round(mae, 1),
                "note": (
                    "Positive variance = tasks took longer than estimated. "
                    "Negative = finished early."
                ),
            },
        }

        logger.info(
            "EOD: %d/%d completed (%.0f%%), avg variance %.1f min",
            len(completed),
            len(self.tasks),
            accuracy_pct,
            avg_variance,
        )
        return summary

    # ------------------------------------------------------------------
    # Output Formatters
    # ------------------------------------------------------------------
    def to_plain_text(self) -> str:
        """Render the full morning plan as plain text (email / SMS friendly)."""
        lines: list[str] = [
            f"=== BNSF Campus Morning Plan -- {self.target_date.isoformat()} ===",
            f"Site: {SITE_ADDRESS}",
            "",
        ]

        if self.dashboard:
            db = self.dashboard
            lines.append("--- Management Summary ---")
            lines.append(f"Open WOs: {db.total_open_wo}  |  PMs Due: {db.total_pm_due}")
            lines.append(f"Projected completion: {db.projected_completion_pct:.0f}%")
            lines.append(f"Carryover from yesterday: {db.carryover_from_yesterday}")
            if db.risk_flags:
                lines.append("Risk flags:")
                for flag in db.risk_flags:
                    lines.append(f"  ! {flag}")
            lines.append("")

        for tid, sched in self.tech_schedules.items():
            lines.append(f"--- {sched.tech_name} ({tid}) ---")
            lines.append(f"Route: {self._describe_route(sched.route_sequence)}")
            lines.append(f"Estimated hours: {sched.total_estimated_hours}")
            lines.append(f"Projected end: {sched.projected_end_time.isoformat()}")
            for task in sched.tasks:
                est_str = ""
                if task.estimate:
                    est_str = (
                        f"  [{task.estimate.estimated_start.strftime('%H:%M')}"
                        f"-{task.estimate.estimated_end.strftime('%H:%M')}"
                        f" {task.estimate.confidence.value}]"
                    )
                parts_str = f"  Parts: {', '.join(task.parts_needed)}" if task.parts_needed else ""
                tools_str = f"  Tools: {', '.join(task.tools_needed)}" if task.tools_needed else ""
                lines.append(
                    f"  {task.sequence_order}. [{task.wo_number}] {task.title} "
                    f"(Zone {task.zone}, {task.building} F{task.floor}) "
                    f"~{task.estimated_minutes}min{est_str}{parts_str}{tools_str}"
                )
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Render the full morning plan as a structured dict (JSON API)."""
        return {
            "date": self.target_date.isoformat(),
            "site": SITE_ADDRESS,
            "dashboard": self.dashboard.to_dict() if self.dashboard else None,
            "tech_schedules": {
                tid: sched.to_dict() for tid, sched in self.tech_schedules.items()
            },
            "all_tasks": [t.to_dict() for t in self.tasks],
            "completion_estimates": [
                e.to_dict() for e in self.estimate_completion_times()
            ],
        }

    def to_html(self) -> str:
        """Render the morning plan as an HTML table for dashboard embedding."""
        html_parts: list[str] = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'>",
            "<style>",
            "  body { font-family: Arial, sans-serif; margin: 20px; }",
            "  h1, h2, h3 { color: #1a3c5e; }",
            "  table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "  th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }",
            "  th { background: #1a3c5e; color: #fff; }",
            "  tr:nth-child(even) { background: #f4f4f4; }",
            "  .risk { color: #c0392b; font-weight: bold; }",
            "  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;"
            "           font-size: 0.85em; color: #fff; }",
            "  .badge-high { background: #c0392b; }",
            "  .badge-med  { background: #e67e22; }",
            "  .badge-low  { background: #27ae60; }",
            "</style>",
            f"<title>BNSF Morning Plan {self.target_date.isoformat()}</title>",
            "</head><body>",
            f"<h1>BNSF Campus Morning Plan &mdash; {self.target_date.isoformat()}</h1>",
            f"<p>Site: {SITE_ADDRESS}</p>",
        ]

        # Dashboard summary
        if self.dashboard:
            db = self.dashboard
            html_parts.append("<h2>Management Summary</h2>")
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>Open WOs</th><th>PMs Due</th><th>Projected Completion</th>"
                "<th>Carryover</th></tr>"
            )
            html_parts.append(
                f"<tr><td>{db.total_open_wo}</td><td>{db.total_pm_due}</td>"
                f"<td>{db.projected_completion_pct:.0f}%</td>"
                f"<td>{db.carryover_from_yesterday}</td></tr>"
            )
            html_parts.append("</table>")

            if db.risk_flags:
                html_parts.append("<h3>Risk Flags</h3><ul>")
                for flag in db.risk_flags:
                    html_parts.append(f"<li class='risk'>{flag}</li>")
                html_parts.append("</ul>")

            # Priority breakdown
            html_parts.append("<h3>Tasks by Priority</h3><table>")
            html_parts.append("<tr><th>Priority</th><th>Count</th></tr>")
            for prio, cnt in sorted(db.tasks_by_priority.items()):
                html_parts.append(f"<tr><td>{prio}</td><td>{cnt}</td></tr>")
            html_parts.append("</table>")

            # Trade breakdown
            html_parts.append("<h3>Tasks by Trade</h3><table>")
            html_parts.append("<tr><th>Trade</th><th>Count</th></tr>")
            for trade, cnt in sorted(db.tasks_by_trade.items()):
                html_parts.append(f"<tr><td>{trade}</td><td>{cnt}</td></tr>")
            html_parts.append("</table>")

        # Technician schedules
        for tid, sched in self.tech_schedules.items():
            html_parts.append(f"<h2>{sched.tech_name} ({tid})</h2>")
            html_parts.append(
                f"<p>Route: {self._describe_route(sched.route_sequence)}<br>"
                f"Estimated hours: {sched.total_estimated_hours} | "
                f"Projected end: {sched.projected_end_time.isoformat()}</p>"
            )
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>#</th><th>WO</th><th>Title</th><th>Location</th>"
                "<th>Trade</th><th>Est. Min</th><th>Window</th>"
                "<th>Confidence</th><th>Parts</th><th>Tools</th></tr>"
            )
            for task in sched.tasks:
                window = ""
                conf_badge = ""
                if task.estimate:
                    window = (
                        f"{task.estimate.estimated_start.strftime('%H:%M')}"
                        f" - {task.estimate.estimated_end.strftime('%H:%M')}"
                    )
                    clevel = task.estimate.confidence.value
                    badge_cls = {
                        "high": "badge-low",   # green = good
                        "med": "badge-med",
                        "low": "badge-high",   # red = uncertain
                    }.get(clevel, "badge-med")
                    conf_badge = f"<span class='badge {badge_cls}'>{clevel}</span>"

                parts = ", ".join(task.parts_needed) if task.parts_needed else "&mdash;"
                tools = ", ".join(task.tools_needed) if task.tools_needed else "&mdash;"
                html_parts.append(
                    f"<tr>"
                    f"<td>{task.sequence_order}</td>"
                    f"<td>{task.wo_number}</td>"
                    f"<td>{task.title}</td>"
                    f"<td>Zone {task.zone} / {task.building} F{task.floor}</td>"
                    f"<td>{task.trade}</td>"
                    f"<td>{task.estimated_minutes}</td>"
                    f"<td>{window}</td>"
                    f"<td>{conf_badge}</td>"
                    f"<td>{parts}</td>"
                    f"<td>{tools}</td>"
                    f"</tr>"
                )
            html_parts.append("</table>")

        html_parts.append("</body></html>")
        return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
def _demo() -> None:
    """Full morning-generation cycle using sample data."""

    today = date.today()
    morning = datetime.combine(today, WORKDAY_START)

    # -- Sample work orders -------------------------------------------------
    sample_work_orders: list[dict[str, Any]] = [
        {
            "task_id": "T001",
            "wo_number": "WO-10451",
            "title": "AHU-3 no cooling -- 2nd floor HQ East Wing",
            "description": "Air handling unit 3 not providing cooling. Occupants reporting 82F. No AC on entire east wing.",
            "building": "HQ-East Wing",
            "floor": 2,
            "zone": "A",
            "trade": "HVAC",
            "sla_deadline": (morning + timedelta(hours=4)).isoformat(),
            "estimated_minutes": 90,
            "parts_needed": ["refrigerant R-410A", "contactor"],
            "tools_needed": ["manifold gauges", "multimeter"],
        },
        {
            "task_id": "T002",
            "wo_number": "WO-10452",
            "title": "Emergency -- electrical sparking Panel 7B Ops Center",
            "description": "Sparking observed at main distribution panel 7B. Fire risk. Isolated breaker.",
            "building": "OPS-Center",
            "floor": 1,
            "zone": "B",
            "trade": "electrical",
            "sla_deadline": (morning + timedelta(hours=0, minutes=30)).isoformat(),
            "estimated_minutes": 60,
            "parts_needed": ["breaker 200A", "bus bar"],
            "tools_needed": ["arc flash PPE", "torque wrench", "multimeter"],
        },
        {
            "task_id": "T003",
            "wo_number": "WO-10453",
            "title": "Restroom fixture leak -- Cafeteria men's room",
            "description": "Toilet supply line leaking. Water on floor, slip hazard.",
            "building": "Cafeteria",
            "floor": 1,
            "zone": "G",
            "trade": "plumbing",
            "sla_deadline": (morning + timedelta(hours=8)).isoformat(),
            "estimated_minutes": 45,
            "parts_needed": ["1/2in supply line", "shut-off valve"],
            "tools_needed": ["basin wrench", "pipe cutter"],
        },
        {
            "task_id": "T004",
            "wo_number": "PM-2201",
            "title": "PM -- Monthly CRAC unit inspection (DC-Primary)",
            "description": "Scheduled PM for computer room AC units in primary data center. 3 days overdue.",
            "building": "DC-Primary",
            "floor": 1,
            "zone": "C",
            "trade": "HVAC",
            "sla_deadline": (morning + timedelta(hours=24)).isoformat(),
            "estimated_minutes": 120,
            "parts_needed": ["air filters (4)", "belt set"],
            "tools_needed": ["filter wrench", "belt tension gauge"],
        },
        {
            "task_id": "T005",
            "wo_number": "WO-10454",
            "title": "Parking garage level 3 lights out -- North structure",
            "description": "Multiple LED fixtures dark on level 3. Security concern.",
            "building": "Parking-North",
            "floor": 3,
            "zone": "E",
            "trade": "electrical",
            "sla_deadline": (morning + timedelta(hours=24)).isoformat(),
            "estimated_minutes": 40,
            "parts_needed": ["LED troffer (4)"],
            "tools_needed": ["lift", "wire nuts"],
        },
        {
            "task_id": "T006",
            "wo_number": "WO-10455",
            "title": "Fence section damaged -- south perimeter",
            "description": "8ft section of chain-link fence knocked down by vehicle. Perimeter security gap.",
            "building": "Grounds-South",
            "floor": 1,
            "zone": "F",
            "trade": "general maintenance",
            "sla_deadline": (morning + timedelta(hours=24)).isoformat(),
            "estimated_minutes": 90,
            "parts_needed": ["chain-link panel 8ft", "tension wire", "post caps (2)"],
            "tools_needed": ["fence stretcher", "come-along", "post driver"],
        },
        {
            "task_id": "T007",
            "wo_number": "WO-10456",
            "title": "Conference room B-204 projector mount loose",
            "description": "Ceiling projector mount wobbling. Cosmetic / low risk.",
            "building": "HQ-Main",
            "floor": 2,
            "zone": "A",
            "trade": "general maintenance",
            "sla_deadline": (morning + timedelta(hours=72)).isoformat(),
            "estimated_minutes": 20,
            "parts_needed": ["toggle bolts (4)"],
            "tools_needed": ["drill", "stud finder"],
        },
        {
            "task_id": "T008",
            "wo_number": "PM-2202",
            "title": "PM -- Quarterly elevator inspection (Parking-South)",
            "description": "State-mandated quarterly elevator inspection. Due today.",
            "building": "Parking-South",
            "floor": 1,
            "zone": "E",
            "trade": "elevator",
            "sla_deadline": (morning + timedelta(hours=8)).isoformat(),
            "estimated_minutes": 60,
            "parts_needed": [],
            "tools_needed": ["elevator key", "inspection checklist"],
        },
        {
            "task_id": "T009",
            "wo_number": "WO-10457",
            "title": "Kitchen hood exhaust fan vibration -- Cafeteria",
            "description": "Exhaust fan over main cook line vibrating loudly. Bearings likely worn.",
            "building": "Cafeteria",
            "floor": 1,
            "zone": "G",
            "trade": "kitchen equipment",
            "sla_deadline": (morning + timedelta(hours=8)).isoformat(),
            "estimated_minutes": 75,
            "parts_needed": ["fan bearings (2)", "v-belt"],
            "tools_needed": ["bearing puller", "alignment tool"],
        },
        {
            "task_id": "T010",
            "wo_number": "WO-10458",
            "title": "UPS battery alarm -- DC-Secondary",
            "description": "UPS string 2 showing battery fault. Redundancy compromised.",
            "building": "DC-Secondary",
            "floor": 1,
            "zone": "C",
            "trade": "electrical",
            "sla_deadline": (morning + timedelta(hours=4)).isoformat(),
            "estimated_minutes": 45,
            "parts_needed": ["UPS battery module"],
            "tools_needed": ["insulated gloves", "battery tester", "torque wrench"],
        },
    ]

    # -- Sample technicians -------------------------------------------------
    technicians: list[dict[str, Any]] = [
        {
            "tech_id": "TECH-01",
            "tech_name": "Carlos Mendoza",
            "trades": ["HVAC", "general maintenance"],
        },
        {
            "tech_id": "TECH-02",
            "tech_name": "David Park",
            "trades": ["electrical", "general maintenance"],
        },
        {
            "tech_id": "TECH-03",
            "tech_name": "Maria Santos",
            "trades": ["plumbing", "kitchen equipment", "general maintenance"],
        },
        {
            "tech_id": "TECH-04",
            "tech_name": "James Walker",
            "trades": ["elevator", "electrical", "general maintenance"],
        },
    ]

    # -- Run full morning cycle ---------------------------------------------
    agent = MorningTodoAgent(target_date=today)

    print("=" * 72)
    print(" BNSF Morning TODO Agent -- Full Generation Cycle")
    print("=" * 72)

    # Step 1: Pull
    print("\n[Step 1] Pulling daily workload ...")
    agent.pull_daily_workload(raw_work_orders=sample_work_orders)
    print(f"  Loaded {len(agent.tasks)} tasks.")

    # Step 2: Prioritize
    print("\n[Step 2] Prioritizing tasks ...")
    agent.prioritize_tasks()
    print("  Ranked tasks (highest priority first):")
    for t in agent.tasks:
        print(f"    {t.sequence_order:>2}. [{t.wo_number}] score={t.priority_score:6.1f}  {t.title}")

    # Step 3: Generate schedules
    print("\n[Step 3] Generating technician schedules ...")
    agent.generate_technician_schedule(technicians)
    for tid, sched in agent.tech_schedules.items():
        print(f"  {sched.tech_name}: {len(sched.tasks)} tasks, "
              f"{sched.total_estimated_hours:.1f}h, end ~{sched.projected_end_time}")

    # Step 4: Completion estimates
    print("\n[Step 4] Completion estimates:")
    estimates = agent.estimate_completion_times()
    for est in estimates:
        print(
            f"  {est.task_id}: {est.estimated_start.strftime('%H:%M')}"
            f" - {est.estimated_end.strftime('%H:%M')}"
            f"  ({est.estimated_duration_min}min, {est.confidence.value} confidence)"
        )

    # Step 5: Management view
    print("\n[Step 5] Management dashboard:")
    dashboard = agent.generate_management_view(carryover_count=2)
    print(f"  Open WOs: {dashboard.total_open_wo}")
    print(f"  PMs Due: {dashboard.total_pm_due}")
    print(f"  By priority: {dashboard.tasks_by_priority}")
    print(f"  Projected completion: {dashboard.projected_completion_pct:.0f}%")
    if dashboard.risk_flags:
        print("  Risk flags:")
        for flag in dashboard.risk_flags:
            print(f"    ! {flag}")

    # Step 6: Tech views (show one example)
    print("\n[Step 6] Technician view (Carlos Mendoza):")
    tech_view = agent.generate_tech_view("TECH-01")
    print(f"  Route: {tech_view.get('route_description', 'N/A')}")
    for tv in tech_view.get("tasks", []):
        print(
            f"    {tv['wo_number']}  {tv['title'][:50]}"
            f"  [{tv.get('estimated_start', '?')}-{tv.get('estimated_end', '?')}]"
            f"  conf={tv.get('confidence', '?')}"
        )

    # Step 7: Simulate tracking actuals
    print("\n[Step 7] Simulating actual tracking ...")
    first_task = agent.tech_schedules["TECH-01"].tasks[0] if agent.tech_schedules["TECH-01"].tasks else None
    if first_task:
        sim_start = datetime.combine(today, time(7, 10))
        sim_end = sim_start + timedelta(minutes=first_task.estimated_minutes + 12)
        agent.track_actuals(first_task.task_id, actual_start=sim_start)
        agent.track_actuals(first_task.task_id, actual_end=sim_end)
        print(f"  Tracked {first_task.wo_number}: started {sim_start.strftime('%H:%M')}, "
              f"ended {sim_end.strftime('%H:%M')} (+12 min over estimate)")

    # Step 8: End-of-day summary
    print("\n[Step 8] End-of-day summary:")
    eod = agent.end_of_day_summary()
    print(f"  Completed: {eod['total_completed']}/{eod['total_planned']}")
    print(f"  Completion rate: {eod['completion_rate_pct']}%")
    print(f"  Avg variance: {eod['avg_variance_minutes']} min")
    print(f"  Deferred: {len(eod['deferred_tasks'])} task(s)")

    # Step 9: Output formats
    print("\n[Step 9] Plain text output (first 40 lines):")
    plain = agent.to_plain_text()
    for line in plain.split("\n")[:40]:
        print(f"  {line}")

    print("\n[Step 9b] Dict output keys:")
    d = agent.to_dict()
    print(f"  Top-level keys: {list(d.keys())}")
    print(f"  Tech schedules: {list(d['tech_schedules'].keys())}")

    print("\n[Step 9c] HTML output generated ({} chars)".format(len(agent.to_html())))
    print("\nDone.")


if __name__ == "__main__":
    _demo()

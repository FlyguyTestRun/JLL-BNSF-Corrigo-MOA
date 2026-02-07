"""PM Systematic Completion Agent -- Industrial Engineering Methodology.

Applies time studies, process optimization, and standard work procedures
to preventive maintenance task completion at the JLL-managed BNSF Railway
campus (2400 Lou Menk Dr, Fort Worth, TX).

Part of the JLL-BNSF Corrigo Multi-Agent Orchestration (MAO) system.

Design principles:
    - Strict token conservation: all LLM prompts < 500 tokens, JSON in/out.
    - Nearest-neighbor routing across campus buildings/floors.
    - Continuous improvement via ADRs (Action/Decision Records).
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("mao.pm_systematic")

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BottleneckCategory(str, Enum):
    """Categories of systematic PM bottlenecks."""

    PARTS = "parts"
    ACCESS = "access"
    SKILL = "skill"
    TOOL = "tool"
    SCHEDULING = "scheduling"


class ADRStatus(str, Enum):
    """Lifecycle states for an Action/Decision Record."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    IMPLEMENTED = "implemented"
    DEPRECATED = "deprecated"


class Trade(str, Enum):
    """Maintenance trade classifications used on the BNSF campus."""

    HVAC = "hvac"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    GENERAL = "general"
    FIRE_LIFE_SAFETY = "fire_life_safety"
    ELEVATOR = "elevator"
    ROOFING = "roofing"
    GROUNDS = "grounds"


# ---------------------------------------------------------------------------
# Industrial Engineering dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TimeStudy:
    """Observed time-study record for a single PM task execution.

    All durations are in minutes.  ``efficiency_rating`` is the ratio of
    wrench_minutes to total_minutes (higher is better, 1.0 = perfect).
    """

    task_id: str
    task_type: str
    setup_minutes: float
    travel_minutes: float
    wrench_minutes: float
    doc_minutes: float
    total_minutes: float
    efficiency_rating: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StandardWork:
    """Standard Operating Procedure for a PM task type.

    Each step dict has keys:
        step_num (int), description (str), est_minutes (float),
        tools_needed (list[str]), safety_notes (str).
    """

    procedure_id: str
    task_type: str
    trade: str
    steps: list[dict[str, Any]]
    total_est_minutes: float
    last_updated: str  # ISO-8601 date

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BottleneckAnalysis:
    """A identified systemic bottleneck in PM execution."""

    bottleneck_id: str
    category: BottleneckCategory
    description: str
    frequency: int  # occurrences in analysis window
    avg_delay_minutes: float
    affected_tasks: list[str]
    root_cause: str
    recommended_fix: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        return data


@dataclass
class ADR:
    """Action/Decision Record for tracking improvement decisions.

    Follows the ADR pattern: Context -> Decision -> Consequences.
    """

    adr_id: str
    date: str  # ISO-8601 date
    title: str
    context: str
    decision: str
    consequences: str
    status: ADRStatus
    related_wo_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


# ---------------------------------------------------------------------------
# Campus layout -- simplified graph model
# ---------------------------------------------------------------------------

# Nodes: building/zone id -> metadata
CAMPUS_BUILDINGS: dict[str, dict[str, Any]] = {
    "main_hq": {
        "name": "Main HQ",
        "floors": 3,
        "coords": (0.0, 0.0),
        "description": "Corporate headquarters, offices, and conference rooms",
    },
    "maintenance_bldg": {
        "name": "Maintenance Building",
        "floors": 1,
        "coords": (0.3, 0.1),
        "description": "Central maintenance shop, parts storage, tool cribs",
    },
    "data_center": {
        "name": "Data Center",
        "floors": 2,
        "coords": (0.2, -0.15),
        "description": "Primary IT/data center with precision cooling",
    },
    "parking_garage": {
        "name": "Parking Garage",
        "floors": 4,
        "coords": (-0.2, 0.25),
        "description": "Multi-level employee parking structure",
    },
    "grounds": {
        "name": "Grounds",
        "floors": 0,
        "coords": (0.0, 0.35),
        "description": "Exterior landscaping, irrigation, and lot areas",
    },
    "training_center": {
        "name": "Training Center",
        "floors": 2,
        "coords": (-0.15, -0.1),
        "description": "Employee training classrooms and simulation labs",
    },
    "operations_center": {
        "name": "Operations Center",
        "floors": 2,
        "coords": (0.35, -0.05),
        "description": "Rail operations monitoring and dispatch",
    },
    "warehouse": {
        "name": "Warehouse",
        "floors": 1,
        "coords": (0.45, 0.2),
        "description": "Bulk parts and equipment storage",
    },
    "cafe_wellness": {
        "name": "Cafe / Wellness Center",
        "floors": 1,
        "coords": (-0.1, 0.15),
        "description": "Employee cafeteria and fitness facility",
    },
}

# Vertical travel cost in minutes per floor transition
_FLOOR_TRAVEL_MINUTES = 2.0


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two campus coordinate points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _travel_minutes(
    from_bldg: str,
    from_floor: int,
    to_bldg: str,
    to_floor: int,
    *,
    walk_speed_factor: float = 30.0,
) -> float:
    """Estimate walking travel time in minutes between two campus locations.

    Parameters
    ----------
    from_bldg, to_bldg:
        Building identifiers (keys of ``CAMPUS_BUILDINGS``).
    from_floor, to_floor:
        Floor numbers (0-indexed).
    walk_speed_factor:
        Minutes per unit of Euclidean distance.  Calibrated so that the
        farthest walk on campus is roughly 15 minutes.
    """
    if from_bldg == to_bldg:
        # Same building -- only vertical travel.
        return abs(from_floor - to_floor) * _FLOOR_TRAVEL_MINUTES

    a = CAMPUS_BUILDINGS[from_bldg]["coords"]
    b = CAMPUS_BUILDINGS[to_bldg]["coords"]
    horiz = _euclidean(a, b) * walk_speed_factor
    vert = abs(from_floor - to_floor) * _FLOOR_TRAVEL_MINUTES
    return round(horiz + vert, 1)


# ---------------------------------------------------------------------------
# Standard time baselines (IE reference data)
# ---------------------------------------------------------------------------

# Average observed times by task_type (minutes).  Keys match Corrigo
# PM task types.  Values are (setup, wrench, doc).
_STANDARD_TIMES: dict[str, tuple[float, float, float]] = {
    "hvac_filter_change": (5.0, 15.0, 5.0),
    "hvac_belt_inspection": (5.0, 20.0, 5.0),
    "hvac_coil_cleaning": (10.0, 45.0, 10.0),
    "electrical_panel_inspection": (5.0, 30.0, 10.0),
    "electrical_lighting_check": (3.0, 10.0, 5.0),
    "plumbing_fixture_inspection": (5.0, 15.0, 5.0),
    "plumbing_backflow_test": (10.0, 30.0, 10.0),
    "fire_extinguisher_inspection": (3.0, 10.0, 5.0),
    "fire_alarm_test": (10.0, 25.0, 10.0),
    "elevator_monthly_inspection": (10.0, 45.0, 15.0),
    "roof_inspection": (10.0, 40.0, 10.0),
    "grounds_irrigation_check": (5.0, 20.0, 5.0),
    "general_door_hardware": (5.0, 15.0, 5.0),
    "general_ceiling_tile": (3.0, 10.0, 3.0),
}


# ---------------------------------------------------------------------------
# Token-conserving LLM prompt helpers
# ---------------------------------------------------------------------------


def _build_llm_prompt(intent: str, payload: dict[str, Any]) -> str:
    """Build a structured JSON prompt under 500 tokens for LLM interaction.

    Parameters
    ----------
    intent:
        Short verb phrase describing the request (e.g. ``"classify_bottleneck"``).
    payload:
        Compact data dict to include in the prompt.

    Returns
    -------
    str
        JSON string suitable for LLM input, kept under 500 tokens.
    """
    prompt_obj = {
        "system": "PM-IE-Agent",
        "intent": intent,
        "data": payload,
        "response_format": "json",
    }
    raw = json.dumps(prompt_obj, separators=(",", ":"))
    # Hard guard: truncate data values if total exceeds ~2000 chars (~500 tok)
    if len(raw) > 2000:
        logger.warning(
            "Prompt exceeds 2000-char budget (%d chars); truncating payload",
            len(raw),
        )
        truncated = {k: str(v)[:120] for k, v in payload.items()}
        prompt_obj["data"] = truncated
        raw = json.dumps(prompt_obj, separators=(",", ":"))
    return raw


def _parse_llm_response(raw_text: str) -> dict[str, Any]:
    """Parse an LLM response as JSON, falling back to a wrapped string.

    Parameters
    ----------
    raw_text:
        Raw text returned by the LLM.

    Returns
    -------
    dict
        Parsed JSON object, or ``{"raw": raw_text}`` on parse failure.
    """
    text = raw_text.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("LLM response was not valid JSON; wrapping as raw string")
        return {"raw": text}


# ---------------------------------------------------------------------------
# PMSystematicAgent
# ---------------------------------------------------------------------------


class PMSystematicAgent:
    """Industrial-engineering-driven PM completion agent.

    Provides nearest-neighbor route optimization, time study analysis,
    standard work generation, bottleneck detection, and ADR creation for
    the BNSF Fort Worth campus managed by JLL.

    All LLM interactions (when an ``llm_callable`` is provided) use
    structured JSON prompts capped at 500 tokens, with responses parsed
    strictly as JSON.

    Parameters
    ----------
    llm_callable:
        Optional async or sync callable ``(str) -> str`` that sends a prompt
        to the backing LLM and returns the response text.  When *None*,
        the agent operates in offline/analytical mode.
    """

    def __init__(
        self,
        llm_callable: Any | None = None,
    ) -> None:
        self._llm = llm_callable
        self._time_studies: list[TimeStudy] = []
        self._standard_works: dict[str, StandardWork] = {}
        self._bottlenecks: list[BottleneckAnalysis] = []
        self._adrs: list[ADR] = []
        logger.info("PMSystematicAgent initialized (llm=%s)", "yes" if llm_callable else "offline")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        """Generate a short unique id with the given prefix."""
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def _call_llm(self, intent: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a token-conserving prompt to the LLM and parse the reply.

        If no LLM callable is configured, returns an empty dict and logs
        a debug message.
        """
        if self._llm is None:
            logger.debug("No LLM configured; skipping call for intent=%s", intent)
            return {}
        prompt = _build_llm_prompt(intent, payload)
        logger.debug("LLM prompt (%d chars): %s", len(prompt), prompt[:200])
        raw_response: str = self._llm(prompt)
        result = _parse_llm_response(raw_response)
        logger.debug("LLM response parsed: %s", list(result.keys()))
        return result

    # ------------------------------------------------------------------
    # 1. analyze_pm_sequence -- nearest-neighbor routing
    # ------------------------------------------------------------------

    def analyze_pm_sequence(
        self,
        tasks: list[dict[str, Any]],
        *,
        start_building: str = "maintenance_bldg",
        start_floor: int = 0,
    ) -> list[dict[str, Any]]:
        """Determine optimal PM task execution order via nearest-neighbor routing.

        Each task dict must contain at minimum::

            {
                "task_id": str,
                "task_type": str,
                "building": str,   # key in CAMPUS_BUILDINGS
                "floor": int,
            }

        Parameters
        ----------
        tasks:
            Unordered list of PM task dicts.
        start_building:
            Building where the technician begins (default: maintenance shop).
        start_floor:
            Floor where the technician begins.

        Returns
        -------
        list[dict]
            Tasks in optimised order, each augmented with:
            ``sequence_num``, ``travel_from_prev_min``, ``cumulative_travel_min``.
        """
        if not tasks:
            logger.warning("analyze_pm_sequence called with empty task list")
            return []

        remaining = list(tasks)
        ordered: list[dict[str, Any]] = []
        cur_bldg = start_building
        cur_floor = start_floor
        cumulative = 0.0

        while remaining:
            # Find nearest unvisited task.
            best_idx = 0
            best_travel = float("inf")
            for idx, t in enumerate(remaining):
                t_bldg = t.get("building", "main_hq")
                t_floor = t.get("floor", 0)
                if t_bldg not in CAMPUS_BUILDINGS:
                    logger.warning(
                        "Unknown building '%s' for task %s; defaulting to main_hq",
                        t_bldg,
                        t.get("task_id"),
                    )
                    t_bldg = "main_hq"
                travel = _travel_minutes(cur_bldg, cur_floor, t_bldg, t_floor)
                if travel < best_travel:
                    best_travel = travel
                    best_idx = idx

            chosen = remaining.pop(best_idx)
            cumulative += best_travel
            enriched = {
                **chosen,
                "sequence_num": len(ordered) + 1,
                "travel_from_prev_min": round(best_travel, 1),
                "cumulative_travel_min": round(cumulative, 1),
            }
            ordered.append(enriched)
            cur_bldg = chosen.get("building", "main_hq")
            cur_floor = chosen.get("floor", 0)

        total_travel = ordered[-1]["cumulative_travel_min"] if ordered else 0.0
        logger.info(
            "PM sequence optimized: %d tasks, total travel %.1f min",
            len(ordered),
            total_travel,
        )
        return ordered

    # ------------------------------------------------------------------
    # 2. estimate_task_duration -- IE standard times + history
    # ------------------------------------------------------------------

    def estimate_task_duration(
        self,
        task_type: str,
        building: str = "main_hq",
        floor: int = 0,
        *,
        from_building: str = "maintenance_bldg",
        from_floor: int = 0,
    ) -> TimeStudy:
        """Estimate duration for a PM task using IE standard times and history.

        Breaks the estimate into setup, travel, wrench (actual work), and
        documentation time.  If historical ``TimeStudy`` records exist for
        this ``task_type``, actual averages are blended with the standard.

        Parameters
        ----------
        task_type:
            Corrigo PM task type key (e.g. ``"hvac_filter_change"``).
        building:
            Target building key.
        floor:
            Target floor.
        from_building, from_floor:
            Technician's current location (default: maintenance shop, ground).

        Returns
        -------
        TimeStudy
            Estimated time breakdown with computed efficiency rating.
        """
        # Base standard times (setup, wrench, doc).
        std = _STANDARD_TIMES.get(task_type, (5.0, 20.0, 5.0))
        setup_std, wrench_std, doc_std = std

        # Historical blending: if we have recorded studies, weight 60/40.
        historicals = [
            ts for ts in self._time_studies if ts.task_type == task_type
        ]
        if historicals:
            avg_setup = sum(t.setup_minutes for t in historicals) / len(historicals)
            avg_wrench = sum(t.wrench_minutes for t in historicals) / len(historicals)
            avg_doc = sum(t.doc_minutes for t in historicals) / len(historicals)
            setup = round(0.4 * setup_std + 0.6 * avg_setup, 1)
            wrench = round(0.4 * wrench_std + 0.6 * avg_wrench, 1)
            doc = round(0.4 * doc_std + 0.6 * avg_doc, 1)
            logger.debug(
                "Blended estimate for %s from %d historical records",
                task_type,
                len(historicals),
            )
        else:
            setup = setup_std
            wrench = wrench_std
            doc = doc_std

        travel = _travel_minutes(from_building, from_floor, building, floor)
        total = round(setup + travel + wrench + doc, 1)
        efficiency = round(wrench / total, 3) if total > 0 else 0.0

        study = TimeStudy(
            task_id=self._next_id("TS"),
            task_type=task_type,
            setup_minutes=setup,
            travel_minutes=travel,
            wrench_minutes=wrench,
            doc_minutes=doc,
            total_minutes=total,
            efficiency_rating=efficiency,
        )
        logger.info(
            "Duration estimate for %s: %.1f min (eff=%.1f%%)",
            task_type,
            total,
            efficiency * 100,
        )
        return study

    # ------------------------------------------------------------------
    # 3. create_standard_work_procedure
    # ------------------------------------------------------------------

    def create_standard_work_procedure(
        self,
        task_type: str,
        trade: str,
        *,
        custom_steps: list[dict[str, Any]] | None = None,
    ) -> StandardWork:
        """Generate a Standard Work (SOP) for a PM task type.

        If ``custom_steps`` are provided they are used directly; otherwise
        a default procedure is generated from IE baselines and, when an LLM
        is configured, enriched via a token-conserving LLM call.

        Parameters
        ----------
        task_type:
            PM task type key.
        trade:
            Trade classification string (see :class:`Trade`).
        custom_steps:
            Optional pre-defined step list to use instead of generation.

        Returns
        -------
        StandardWork
            The generated or provided standard work procedure.
        """
        if custom_steps is not None:
            steps = custom_steps
        else:
            steps = self._generate_default_steps(task_type, trade)

        total_est = round(sum(s.get("est_minutes", 0.0) for s in steps), 1)

        sw = StandardWork(
            procedure_id=self._next_id("SWP"),
            task_type=task_type,
            trade=trade,
            steps=steps,
            total_est_minutes=total_est,
            last_updated=date.today().isoformat(),
        )
        self._standard_works[task_type] = sw
        logger.info(
            "Standard work created: %s (%d steps, %.1f min)",
            sw.procedure_id,
            len(steps),
            total_est,
        )
        return sw

    def _generate_default_steps(
        self, task_type: str, trade: str
    ) -> list[dict[str, Any]]:
        """Build a generic step list from IE baselines + optional LLM enrichment."""
        std = _STANDARD_TIMES.get(task_type, (5.0, 20.0, 5.0))
        setup_min, wrench_min, doc_min = std

        base_steps: list[dict[str, Any]] = [
            {
                "step_num": 1,
                "description": "Gather tools, PPE, and required parts from tool crib",
                "est_minutes": setup_min * 0.6,
                "tools_needed": ["tool bag", "PPE kit"],
                "safety_notes": "Verify PPE condition before departing",
            },
            {
                "step_num": 2,
                "description": "Travel to work location; verify safe access",
                "est_minutes": setup_min * 0.4,
                "tools_needed": [],
                "safety_notes": "Check for overhead hazards upon arrival",
            },
            {
                "step_num": 3,
                "description": f"Perform {task_type.replace('_', ' ')} per manufacturer specs",
                "est_minutes": wrench_min,
                "tools_needed": self._tools_for_trade(trade),
                "safety_notes": "Lock-out/tag-out if energy isolation required",
            },
            {
                "step_num": 4,
                "description": "Inspect completed work; verify operation and clean area",
                "est_minutes": doc_min * 0.4,
                "tools_needed": [],
                "safety_notes": "Ensure all panels/covers replaced",
            },
            {
                "step_num": 5,
                "description": "Complete Corrigo work order documentation with photos",
                "est_minutes": doc_min * 0.6,
                "tools_needed": ["mobile device"],
                "safety_notes": "",
            },
        ]

        # Attempt LLM enrichment if available.
        llm_result = self._call_llm(
            "enrich_standard_work",
            {
                "task_type": task_type,
                "trade": trade,
                "step_count": len(base_steps),
            },
        )
        if llm_result and "steps" in llm_result:
            logger.info("LLM enrichment applied to standard work for %s", task_type)
            # Merge LLM-suggested safety notes into base steps.
            for llm_step in llm_result["steps"]:
                idx = llm_step.get("step_num", 0) - 1
                if 0 <= idx < len(base_steps) and "safety_notes" in llm_step:
                    base_steps[idx]["safety_notes"] = llm_step["safety_notes"]

        return base_steps

    @staticmethod
    def _tools_for_trade(trade: str) -> list[str]:
        """Return default tool list for a given trade."""
        trade_tools: dict[str, list[str]] = {
            "hvac": ["refrigerant gauges", "multimeter", "filter wrench", "fin comb"],
            "electrical": ["multimeter", "voltage tester", "wire strippers", "torque screwdriver"],
            "plumbing": ["pipe wrench", "basin wrench", "plunger", "thread seal tape"],
            "general": ["drill/driver", "level", "tape measure", "utility knife"],
            "fire_life_safety": ["extinguisher gauge tester", "smoke detector tester", "flashlight"],
            "elevator": ["elevator key set", "multimeter", "inspection mirror"],
            "roofing": ["moisture meter", "caulk gun", "pry bar", "safety harness"],
            "grounds": ["irrigation controller tool", "soil probe", "pruning shears"],
        }
        return trade_tools.get(trade, ["general tool set"])

    # ------------------------------------------------------------------
    # 4. calculate_completion_metrics
    # ------------------------------------------------------------------

    def calculate_completion_metrics(
        self,
        completed_tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute PM completion KPIs from a list of finished tasks.

        Each task dict should contain::

            {
                "task_id": str,
                "planned_minutes": float,
                "actual_minutes": float,
                "completed_first_time": bool,
                "required_rework": bool,
                "parts_available": bool,
            }

        Returns
        -------
        dict
            Metrics dictionary with keys: ``task_count``, ``avg_planned_min``,
            ``avg_actual_min``, ``plan_vs_actual_ratio``,
            ``first_time_completion_rate``, ``rework_rate``,
            ``parts_availability_rate``, ``tasks_over_estimate``.
        """
        if not completed_tasks:
            logger.warning("calculate_completion_metrics called with no tasks")
            return {
                "task_count": 0,
                "avg_planned_min": 0.0,
                "avg_actual_min": 0.0,
                "plan_vs_actual_ratio": 0.0,
                "first_time_completion_rate": 0.0,
                "rework_rate": 0.0,
                "parts_availability_rate": 0.0,
                "tasks_over_estimate": 0,
            }

        n = len(completed_tasks)
        total_planned = sum(t.get("planned_minutes", 0.0) for t in completed_tasks)
        total_actual = sum(t.get("actual_minutes", 0.0) for t in completed_tasks)
        first_time = sum(1 for t in completed_tasks if t.get("completed_first_time", False))
        reworks = sum(1 for t in completed_tasks if t.get("required_rework", False))
        parts_ok = sum(1 for t in completed_tasks if t.get("parts_available", True))
        over_est = sum(
            1
            for t in completed_tasks
            if t.get("actual_minutes", 0) > t.get("planned_minutes", 0)
        )

        avg_planned = round(total_planned / n, 1)
        avg_actual = round(total_actual / n, 1)
        ratio = round(total_planned / total_actual, 3) if total_actual > 0 else 0.0

        metrics = {
            "task_count": n,
            "avg_planned_min": avg_planned,
            "avg_actual_min": avg_actual,
            "plan_vs_actual_ratio": ratio,
            "first_time_completion_rate": round(first_time / n, 3),
            "rework_rate": round(reworks / n, 3),
            "parts_availability_rate": round(parts_ok / n, 3),
            "tasks_over_estimate": over_est,
        }
        logger.info(
            "Completion metrics: %d tasks, plan/actual=%.2f, FTC=%.0f%%",
            n,
            ratio,
            metrics["first_time_completion_rate"] * 100,
        )
        return metrics

    # ------------------------------------------------------------------
    # 5. identify_bottlenecks
    # ------------------------------------------------------------------

    def identify_bottlenecks(
        self,
        execution_data: list[dict[str, Any]],
        *,
        delay_threshold_minutes: float = 15.0,
    ) -> list[BottleneckAnalysis]:
        """Analyse PM execution data to find systematic delays.

        Each record in ``execution_data`` should contain::

            {
                "task_id": str,
                "delay_minutes": float,
                "delay_reason": str,   # free-text or category keyword
            }

        Parameters
        ----------
        execution_data:
            Historical execution records that include delay information.
        delay_threshold_minutes:
            Minimum average delay to flag as a bottleneck.

        Returns
        -------
        list[BottleneckAnalysis]
            Detected bottlenecks sorted by average delay (descending).
        """
        # Bucket delays by normalised reason keyword.
        keyword_map: dict[str, BottleneckCategory] = {
            "parts": BottleneckCategory.PARTS,
            "material": BottleneckCategory.PARTS,
            "supply": BottleneckCategory.PARTS,
            "access": BottleneckCategory.ACCESS,
            "locked": BottleneckCategory.ACCESS,
            "key": BottleneckCategory.ACCESS,
            "badge": BottleneckCategory.ACCESS,
            "skill": BottleneckCategory.SKILL,
            "training": BottleneckCategory.SKILL,
            "certification": BottleneckCategory.SKILL,
            "tool": BottleneckCategory.TOOL,
            "equipment": BottleneckCategory.TOOL,
            "schedule": BottleneckCategory.SCHEDULING,
            "conflict": BottleneckCategory.SCHEDULING,
            "tenant": BottleneckCategory.SCHEDULING,
            "occupied": BottleneckCategory.SCHEDULING,
        }

        # Group by category.
        buckets: dict[BottleneckCategory, list[dict[str, Any]]] = {}
        for rec in execution_data:
            reason = rec.get("delay_reason", "").lower()
            cat = BottleneckCategory.SCHEDULING  # default
            for kw, c in keyword_map.items():
                if kw in reason:
                    cat = c
                    break
            buckets.setdefault(cat, []).append(rec)

        results: list[BottleneckAnalysis] = []
        for cat, records in buckets.items():
            delays = [r.get("delay_minutes", 0.0) for r in records]
            avg_delay = round(sum(delays) / len(delays), 1) if delays else 0.0
            if avg_delay < delay_threshold_minutes:
                continue

            affected = [r.get("task_id", "unknown") for r in records]
            root_cause = self._infer_root_cause(cat, records)
            fix = self._suggest_fix(cat, avg_delay)

            bn = BottleneckAnalysis(
                bottleneck_id=self._next_id("BN"),
                category=cat,
                description=f"{cat.value.title()} delays across {len(records)} task(s)",
                frequency=len(records),
                avg_delay_minutes=avg_delay,
                affected_tasks=affected,
                root_cause=root_cause,
                recommended_fix=fix,
            )
            results.append(bn)

        # Sort by severity.
        results.sort(key=lambda b: b.avg_delay_minutes, reverse=True)
        self._bottlenecks.extend(results)
        logger.info("Bottleneck analysis complete: %d bottleneck(s) found", len(results))
        return results

    @staticmethod
    def _infer_root_cause(
        cat: BottleneckCategory, records: list[dict[str, Any]]
    ) -> str:
        """Derive a root-cause summary string from delay category + data."""
        reasons = [r.get("delay_reason", "") for r in records]
        common = max(set(reasons), key=reasons.count) if reasons else "unknown"
        templates: dict[BottleneckCategory, str] = {
            BottleneckCategory.PARTS: (
                f"Recurring parts unavailability; most common reason: '{common}'"
            ),
            BottleneckCategory.ACCESS: (
                f"Repeated access-control delays; most common reason: '{common}'"
            ),
            BottleneckCategory.SKILL: (
                f"Skill/certification gap identified; most common reason: '{common}'"
            ),
            BottleneckCategory.TOOL: (
                f"Tool/equipment shortage; most common reason: '{common}'"
            ),
            BottleneckCategory.SCHEDULING: (
                f"Scheduling conflicts with occupied spaces; most common reason: '{common}'"
            ),
        }
        return templates.get(cat, f"Unclassified delay pattern: '{common}'")

    @staticmethod
    def _suggest_fix(cat: BottleneckCategory, avg_delay: float) -> str:
        """Return a recommended fix based on bottleneck category and severity."""
        fixes: dict[BottleneckCategory, str] = {
            BottleneckCategory.PARTS: (
                "Implement min/max inventory levels for top-consumed PM parts; "
                "establish vendor consignment for critical filters and belts"
            ),
            BottleneckCategory.ACCESS: (
                "Pre-coordinate access with security 48 hours before PM window; "
                "issue standing maintenance access badges for recurring routes"
            ),
            BottleneckCategory.SKILL: (
                "Schedule cross-training sessions for identified skill gaps; "
                "pair junior techs with seniors on complex PM tasks"
            ),
            BottleneckCategory.TOOL: (
                "Audit tool crib inventory against PM schedule demand; "
                "acquire duplicate specialty tools for high-frequency tasks"
            ),
            BottleneckCategory.SCHEDULING: (
                "Negotiate dedicated PM windows with building occupants; "
                "shift non-critical PMs to off-hours or weekends"
            ),
        }
        base = fixes.get(cat, "Conduct detailed root-cause analysis")
        if avg_delay > 60:
            base += "; ESCALATE -- average delay exceeds 1 hour"
        return base

    # ------------------------------------------------------------------
    # 6. generate_improvement_recommendations (ADRs)
    # ------------------------------------------------------------------

    def generate_improvement_recommendations(
        self,
        issues: list[dict[str, Any]],
    ) -> list[ADR]:
        """Produce Action/Decision Records for failures and improvement opportunities.

        Each issue dict should contain::

            {
                "title": str,
                "context": str,
                "related_wo_ids": list[str],  # optional
            }

        Parameters
        ----------
        issues:
            Problems or opportunities to address.

        Returns
        -------
        list[ADR]
            Generated ADR records in ``proposed`` status.
        """
        adrs: list[ADR] = []
        for issue in issues:
            title = issue.get("title", "Untitled issue")
            context = issue.get("context", "No context provided")
            wo_ids = issue.get("related_wo_ids", [])

            # Attempt LLM-assisted decision drafting.
            llm_result = self._call_llm(
                "draft_adr",
                {"title": title, "context": context[:300]},
            )
            decision = llm_result.get(
                "decision",
                f"Investigate and resolve: {title}",
            )
            consequences = llm_result.get(
                "consequences",
                "Improved PM reliability and reduced rework if implemented",
            )

            adr = ADR(
                adr_id=self._next_id("ADR"),
                date=date.today().isoformat(),
                title=title,
                context=context,
                decision=decision,
                consequences=consequences,
                status=ADRStatus.PROPOSED,
                related_wo_ids=wo_ids,
            )
            adrs.append(adr)

        self._adrs.extend(adrs)
        logger.info("Generated %d ADR(s)", len(adrs))
        return adrs

    # ------------------------------------------------------------------
    # Record keeping
    # ------------------------------------------------------------------

    def record_time_study(self, study: TimeStudy) -> None:
        """Persist a completed time study for future estimation blending."""
        self._time_studies.append(study)
        logger.debug("Recorded time study %s", study.task_id)

    def get_all_time_studies(self) -> list[dict[str, Any]]:
        """Return all recorded time studies as dicts."""
        return [ts.to_dict() for ts in self._time_studies]

    def get_all_adrs(self) -> list[dict[str, Any]]:
        """Return all ADRs as dicts."""
        return [a.to_dict() for a in self._adrs]

    def get_all_bottlenecks(self) -> list[dict[str, Any]]:
        """Return all bottleneck analyses as dicts."""
        return [b.to_dict() for b in self._bottlenecks]

    def get_standard_work(self, task_type: str) -> dict[str, Any] | None:
        """Retrieve a standard work procedure by task type, or None."""
        sw = self._standard_works.get(task_type)
        return sw.to_dict() if sw else None


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Demonstrate PMSystematicAgent capabilities with sample data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    agent = PMSystematicAgent()

    # --- 1. Route optimization ---
    print("=" * 70)
    print("1. PM SEQUENCE OPTIMIZATION (Nearest-Neighbor Routing)")
    print("=" * 70)
    sample_tasks = [
        {"task_id": "PM-1001", "task_type": "hvac_filter_change", "building": "main_hq", "floor": 2},
        {"task_id": "PM-1002", "task_type": "electrical_panel_inspection", "building": "data_center", "floor": 1},
        {"task_id": "PM-1003", "task_type": "fire_extinguisher_inspection", "building": "parking_garage", "floor": 3},
        {"task_id": "PM-1004", "task_type": "plumbing_fixture_inspection", "building": "cafe_wellness", "floor": 0},
        {"task_id": "PM-1005", "task_type": "hvac_coil_cleaning", "building": "operations_center", "floor": 1},
        {"task_id": "PM-1006", "task_type": "general_door_hardware", "building": "training_center", "floor": 0},
    ]
    sequence = agent.analyze_pm_sequence(sample_tasks)
    for t in sequence:
        bldg_name = CAMPUS_BUILDINGS.get(t["building"], {}).get("name", t["building"])
        print(
            f"  #{t['sequence_num']:>2}  {t['task_id']}  {bldg_name} F{t['floor']}"
            f"  travel={t['travel_from_prev_min']:.1f}min"
            f"  cumul={t['cumulative_travel_min']:.1f}min"
        )

    # --- 2. Task duration estimation ---
    print("\n" + "=" * 70)
    print("2. TASK DURATION ESTIMATION (IE Standard Times)")
    print("=" * 70)
    for tt in ["hvac_filter_change", "electrical_panel_inspection", "elevator_monthly_inspection"]:
        est = agent.estimate_task_duration(tt, "main_hq", 1)
        print(
            f"  {tt:40s}  total={est.total_minutes:5.1f}min"
            f"  wrench={est.wrench_minutes:5.1f}min"
            f"  eff={est.efficiency_rating:.0%}"
        )

    # --- 3. Standard work procedure ---
    print("\n" + "=" * 70)
    print("3. STANDARD WORK PROCEDURE")
    print("=" * 70)
    sw = agent.create_standard_work_procedure("hvac_filter_change", "hvac")
    print(f"  Procedure: {sw.procedure_id}  ({sw.total_est_minutes:.1f} min)")
    for step in sw.steps:
        print(f"    Step {step['step_num']}: {step['description']}")
        print(f"           Est: {step['est_minutes']:.1f}min | Tools: {step['tools_needed']}")
        if step.get("safety_notes"):
            print(f"           Safety: {step['safety_notes']}")

    # --- 4. Completion metrics ---
    print("\n" + "=" * 70)
    print("4. COMPLETION METRICS")
    print("=" * 70)
    completed = [
        {"task_id": "PM-901", "planned_minutes": 30, "actual_minutes": 28, "completed_first_time": True, "required_rework": False, "parts_available": True},
        {"task_id": "PM-902", "planned_minutes": 45, "actual_minutes": 62, "completed_first_time": False, "required_rework": True, "parts_available": True},
        {"task_id": "PM-903", "planned_minutes": 20, "actual_minutes": 18, "completed_first_time": True, "required_rework": False, "parts_available": True},
        {"task_id": "PM-904", "planned_minutes": 60, "actual_minutes": 55, "completed_first_time": True, "required_rework": False, "parts_available": False},
        {"task_id": "PM-905", "planned_minutes": 25, "actual_minutes": 40, "completed_first_time": False, "required_rework": True, "parts_available": False},
    ]
    metrics = agent.calculate_completion_metrics(completed)
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")

    # --- 5. Bottleneck identification ---
    print("\n" + "=" * 70)
    print("5. BOTTLENECK ANALYSIS")
    print("=" * 70)
    delays = [
        {"task_id": "PM-801", "delay_minutes": 45, "delay_reason": "Parts not in stock - filter 20x20x1"},
        {"task_id": "PM-802", "delay_minutes": 30, "delay_reason": "Parts backordered - belt A68"},
        {"task_id": "PM-803", "delay_minutes": 25, "delay_reason": "Locked mechanical room - no badge access"},
        {"task_id": "PM-804", "delay_minutes": 20, "delay_reason": "Room occupied by tenant - schedule conflict"},
        {"task_id": "PM-805", "delay_minutes": 60, "delay_reason": "Parts unavailable - compressor contactor"},
        {"task_id": "PM-806", "delay_minutes": 18, "delay_reason": "Key not available for roof access hatch"},
    ]
    bottlenecks = agent.identify_bottlenecks(delays)
    for bn in bottlenecks:
        print(f"  [{bn.category.value.upper():12s}]  avg delay={bn.avg_delay_minutes:.1f}min  freq={bn.frequency}")
        print(f"    Root cause : {bn.root_cause}")
        print(f"    Fix        : {bn.recommended_fix}")

    # --- 6. Improvement ADRs ---
    print("\n" + "=" * 70)
    print("6. ACTION/DECISION RECORDS (ADRs)")
    print("=" * 70)
    issues = [
        {
            "title": "Recurring HVAC filter stockout",
            "context": "Filters 20x20x1 have been out of stock 3 times this quarter, "
            "causing PM deferrals and rework on WO-4401, WO-4455, WO-4512.",
            "related_wo_ids": ["WO-4401", "WO-4455", "WO-4512"],
        },
        {
            "title": "Mechanical room access delays",
            "context": "Technicians consistently report 20-30 minute delays waiting "
            "for security escort to unlock mechanical rooms in Main HQ.",
            "related_wo_ids": ["WO-4320", "WO-4388"],
        },
    ]
    adrs = agent.generate_improvement_recommendations(issues)
    for adr in adrs:
        print(f"  {adr.adr_id}  [{adr.status.value}]  {adr.title}")
        print(f"    Context : {adr.context[:100]}...")
        print(f"    Decision: {adr.decision}")
        print(f"    WOs     : {adr.related_wo_ids}")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    _demo()

"""
MAO Orchestrator -- Central coordinator for JLL-BNSF Corrigo Multi-Agent Orchestration.

Routes tasks between Shaw Goals Agent, PM Systematic Agent, and Morning TODO Agent.
Manages shared state, enforces per-cycle token budgets (default 5 000 tokens), and
produces unified management / technician outputs.

Designed for Python 3.11+.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CycleType(str, Enum):
    """Which daily cycle is running."""
    MORNING = "morning"
    MIDDAY = "midday"
    END_OF_DAY = "eod"


class MessageType(str, Enum):
    """Inter-agent message types."""
    REQUEST = "request"
    RESPONSE = "response"
    ALERT = "alert"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TokenLedger:
    """Single entry in the token-usage ledger.

    Tracks how many tokens were *allocated* vs. *used* for one agent call.
    """
    agent_name: str
    call_type: str
    tokens_allocated: int
    tokens_used: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def overage(self) -> int:
        """Positive value means the agent exceeded its allocation."""
        return max(0, self.tokens_used - self.tokens_allocated)


@dataclass
class AgentMessage:
    """Structured message passed between agents.

    All inter-agent communication flows through this format so the
    orchestrator can meter, log, and audit every exchange.
    """
    from_agent: str
    to_agent: str
    message_type: MessageType
    payload: dict[str, Any]
    token_cost: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class OrchestrationState:
    """Snapshot of one orchestration cycle."""
    cycle_id: str
    date: date
    cycle_type: CycleType
    agents_called: list[str] = field(default_factory=list)
    token_budget_remaining: int = 5000
    tasks_processed: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def elapsed_seconds(self) -> float | None:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# ---------------------------------------------------------------------------
# Lightweight agent stubs
# ---------------------------------------------------------------------------
# These stubs simulate the three sub-agents that will be imported once their
# own modules are implemented.  Each method returns structured JSON-ready
# dicts to minimise token use during inter-agent communication.

class _MorningTodoAgentStub:
    """Stub for the Morning TODO Agent."""

    name: str = "MorningTodoAgent"

    def pull_daily_workload(self, target_date: date) -> dict[str, Any]:
        """Fetch today's open WOs, scheduled PMs, and carryovers from Corrigo."""
        logger.info("[%s] Pulling daily workload for %s", self.name, target_date)
        return {
            "date": target_date.isoformat(),
            "open_wos": [],
            "scheduled_pms": [],
            "carryovers": [],
            "token_estimate": 120,
        }

    def prioritize_tasks(
        self,
        tasks: list[dict[str, Any]],
        pm_sequence: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank tasks by SLA urgency, Shaw Goals weight, and travel distance."""
        logger.info("[%s] Prioritising %d tasks", self.name, len(tasks))
        return sorted(tasks, key=lambda t: t.get("priority", 999))

    def map_campus_route(
        self, technician_id: str, tasks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Produce an optimised campus route for a single technician."""
        logger.info(
            "[%s] Mapping campus route for tech %s (%d tasks)",
            self.name,
            technician_id,
            len(tasks),
        )
        return {
            "technician_id": technician_id,
            "ordered_tasks": [t.get("wo_id") for t in tasks],
            "estimated_travel_minutes": 0,
            "token_estimate": 80,
        }


class _PMSystematicAgentStub:
    """Stub for the PM Systematic Agent."""

    name: str = "PMSystematicAgent"

    def analyze_pm_sequence(
        self, pms: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return PMs in optimised execution order (trade grouping, location clustering)."""
        logger.info("[%s] Analysing PM sequence for %d PMs", self.name, len(pms))
        return pms  # no-op until real implementation


class _ShawGoalsAgentStub:
    """Stub for the Shaw Goals Agent."""

    name: str = "ShawGoalsAgent"

    def evaluate_pm_completion(
        self, completed_pms: list[dict[str, Any]], target_date: date
    ) -> dict[str, Any]:
        """Score today's PM completion against Shaw Goals targets."""
        logger.info(
            "[%s] Evaluating PM completion (%d PMs) for %s",
            self.name,
            len(completed_pms),
            target_date,
        )
        return {
            "date": target_date.isoformat(),
            "pm_target": 0,
            "pm_actual": len(completed_pms),
            "on_track": True,
            "token_estimate": 90,
        }


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------

class _QueryCache:
    """Simple in-memory cache to avoid re-fetching the same WO data twice."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[datetime, Any]] = {}
        self._ttl = timedelta(minutes=30)

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if datetime.now() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def put(self, key: str, value: Any) -> None:
        self._store[key] = (datetime.now(), value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# MAO Orchestrator
# ---------------------------------------------------------------------------

class MAOOrchestrator:
    """Central Multi-Agent Orchestrator.

    Coordinates Shaw Goals Agent, PM Systematic Agent, and Morning TODO
    Agent.  Every orchestration cycle is budgeted at *max_tokens_per_cycle*
    tokens; agents that exceed their allocation are logged and truncated.

    Attributes
    ----------
    max_tokens_per_cycle : int
        Global token ceiling for one cycle (default 5 000).
    """

    # Default per-agent token allocations as fractions of the cycle budget.
    _AGENT_BUDGET_SHARES: dict[str, float] = {
        "MorningTodoAgent": 0.35,
        "PMSystematicAgent": 0.25,
        "ShawGoalsAgent": 0.20,
        "ReportGeneration": 0.15,
        "Overhead": 0.05,
    }

    def __init__(self, max_tokens_per_cycle: int = 5000) -> None:
        self.max_tokens_per_cycle = max_tokens_per_cycle

        # Sub-agents
        self.morning_todo = _MorningTodoAgentStub()
        self.pm_systematic = _PMSystematicAgentStub()
        self.shaw_goals = _ShawGoalsAgentStub()

        # Shared state
        self._current_state: OrchestrationState | None = None
        self._history: list[OrchestrationState] = []
        self._ledger: list[TokenLedger] = []
        self._messages: list[AgentMessage] = []
        self._cache = _QueryCache()

        logger.info(
            "MAOOrchestrator initialised  (token budget = %d per cycle)",
            self.max_tokens_per_cycle,
        )

    # ----- helpers ----------------------------------------------------------

    def _new_state(self, cycle_type: CycleType) -> OrchestrationState:
        """Create and register a fresh OrchestrationState."""
        state = OrchestrationState(
            cycle_id=uuid.uuid4().hex[:10],
            date=date.today(),
            cycle_type=cycle_type,
            token_budget_remaining=self.max_tokens_per_cycle,
            start_time=datetime.now(),
        )
        self._current_state = state
        return state

    def _close_state(self, state: OrchestrationState) -> None:
        state.end_time = datetime.now()
        self._history.append(state)
        logger.info(
            "Cycle %s (%s) completed in %.2f s  |  tokens remaining: %d  |  tasks: %d  |  errors: %d",
            state.cycle_id,
            state.cycle_type.value,
            state.elapsed_seconds or 0,
            state.token_budget_remaining,
            state.tasks_processed,
            len(state.errors),
        )

    def _allocate_tokens(self, agent_name: str) -> int:
        """Return the token allocation for *agent_name* within the current budget."""
        share = self._AGENT_BUDGET_SHARES.get(agent_name, 0.10)
        allocation = int(self.max_tokens_per_cycle * share)
        if self._current_state and allocation > self._current_state.token_budget_remaining:
            allocation = self._current_state.token_budget_remaining
        return allocation

    def _record_tokens(
        self,
        agent_name: str,
        call_type: str,
        allocated: int,
        used: int,
    ) -> None:
        """Record token usage and deduct from the cycle budget."""
        entry = TokenLedger(
            agent_name=agent_name,
            call_type=call_type,
            tokens_allocated=allocated,
            tokens_used=used,
        )
        self._ledger.append(entry)

        if entry.overage > 0:
            logger.warning(
                "Token overage: %s/%s used %d (allocated %d, overage +%d)",
                agent_name,
                call_type,
                used,
                allocated,
                entry.overage,
            )

        if self._current_state:
            self._current_state.token_budget_remaining -= min(used, allocated)
            self._current_state.agents_called.append(f"{agent_name}.{call_type}")

    def _send_message(
        self,
        from_agent: str,
        to_agent: str,
        msg_type: MessageType,
        payload: dict[str, Any],
        token_cost: int = 0,
    ) -> AgentMessage:
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=msg_type,
            payload=payload,
            token_cost=token_cost,
        )
        self._messages.append(msg)
        return msg

    # ----- daily cycles -----------------------------------------------------

    def morning_cycle(
        self, technicians: list[str] | None = None
    ) -> dict[str, Any]:
        """Run the full morning orchestration.

        Steps
        -----
        1. Pull today's workload (MorningTodoAgent).
        2. Optimise PM execution order (PMSystematicAgent).
        3. Prioritise merged task list (MorningTodoAgent).
        4. Map campus routes per technician (MorningTodoAgent).
        5. Evaluate PM completion forecast vs Shaw Goals.
        6. Generate management report (Tony & Juan).
        7. Generate individual tech schedules.
        8. Log total token usage.

        Returns a summary dict suitable for JSON serialisation.
        """
        state = self._new_state(CycleType.MORNING)
        technicians = technicians or []
        today = date.today()
        report: dict[str, Any] = {"cycle_id": state.cycle_id, "date": today.isoformat()}

        try:
            # 1 -- pull daily workload
            cache_key = f"workload:{today.isoformat()}"
            workload = self._cache.get(cache_key)
            if workload is None:
                alloc = self._allocate_tokens(self.morning_todo.name)
                workload = self.morning_todo.pull_daily_workload(today)
                used = workload.get("token_estimate", alloc)
                self._record_tokens(self.morning_todo.name, "pull_daily_workload", alloc, used)
                self._cache.put(cache_key, workload)
            report["workload"] = workload

            # 2 -- optimise PM sequence
            alloc = self._allocate_tokens(self.pm_systematic.name)
            pms = workload.get("scheduled_pms", [])
            optimised_pms = self.pm_systematic.analyze_pm_sequence(pms)
            self._record_tokens(self.pm_systematic.name, "analyze_pm_sequence", alloc, len(pms) * 5)
            report["optimised_pms"] = optimised_pms

            # 3 -- prioritise combined task list
            all_tasks: list[dict[str, Any]] = (
                workload.get("open_wos", [])
                + optimised_pms
                + workload.get("carryovers", [])
            )
            alloc = self._allocate_tokens(self.morning_todo.name)
            ranked = self.morning_todo.prioritize_tasks(all_tasks, optimised_pms)
            self._record_tokens(self.morning_todo.name, "prioritize_tasks", alloc, len(all_tasks) * 3)
            state.tasks_processed = len(ranked)
            report["ranked_tasks"] = ranked

            # 4 -- campus route per technician
            tech_routes: dict[str, Any] = {}
            for tech_id in technicians:
                tech_tasks = [t for t in ranked if t.get("assigned_to") == tech_id]
                alloc = self._allocate_tokens(self.morning_todo.name)
                route = self.morning_todo.map_campus_route(tech_id, tech_tasks)
                used = route.get("token_estimate", alloc)
                self._record_tokens(self.morning_todo.name, "map_campus_route", alloc, used)
                tech_routes[tech_id] = route
            report["tech_routes"] = tech_routes

            # 5 -- Shaw Goals evaluation
            alloc = self._allocate_tokens(self.shaw_goals.name)
            goals_eval = self.shaw_goals.evaluate_pm_completion(optimised_pms, today)
            used = goals_eval.get("token_estimate", alloc)
            self._record_tokens(self.shaw_goals.name, "evaluate_pm_completion", alloc, used)
            report["shaw_goals"] = goals_eval

            # 6 -- management report (Tony & Juan)
            mgmt_report = self._generate_management_report(state, report)
            report["management_report"] = mgmt_report

            # 7 -- individual tech schedules
            tech_schedules = self._generate_tech_schedules(tech_routes, ranked)
            report["tech_schedules"] = tech_schedules

        except Exception as exc:
            state.errors.append(str(exc))
            logger.exception("Morning cycle error: %s", exc)

        # 8 -- log token usage
        self._close_state(state)
        report["token_usage"] = self.get_token_usage_report(cycle_id=state.cycle_id)
        return report

    def midday_check(
        self,
        completed_wo_ids: list[str] | None = None,
        new_wos: list[dict[str, Any]] | None = None,
        technicians: list[str] | None = None,
    ) -> dict[str, Any]:
        """Re-evaluate progress at midday and re-optimise remaining work.

        Parameters
        ----------
        completed_wo_ids : list[str]
            WO IDs completed since morning.
        new_wos : list[dict]
            Any new reactive WOs that arrived.
        technicians : list[str]
            Active technician IDs.

        Returns
        -------
        dict  Summary of adjustments made.
        """
        state = self._new_state(CycleType.MIDDAY)
        completed_wo_ids = completed_wo_ids or []
        new_wos = new_wos or []
        technicians = technicians or []

        result: dict[str, Any] = {"cycle_id": state.cycle_id}

        try:
            # Re-fetch workload (uses cache if still valid)
            cache_key = f"workload:{date.today().isoformat()}"
            workload = self._cache.get(cache_key)
            remaining_tasks: list[dict[str, Any]] = []
            if workload:
                for t in workload.get("open_wos", []) + workload.get("scheduled_pms", []):
                    if t.get("wo_id") not in completed_wo_ids:
                        remaining_tasks.append(t)
            remaining_tasks.extend(new_wos)

            alloc = self._allocate_tokens(self.morning_todo.name)
            ranked = self.morning_todo.prioritize_tasks(remaining_tasks)
            self._record_tokens(self.morning_todo.name, "prioritize_tasks", alloc, len(remaining_tasks) * 3)
            state.tasks_processed = len(ranked)

            tech_routes: dict[str, Any] = {}
            for tech_id in technicians:
                tech_tasks = [t for t in ranked if t.get("assigned_to") == tech_id]
                alloc = self._allocate_tokens(self.morning_todo.name)
                route = self.morning_todo.map_campus_route(tech_id, tech_tasks)
                self._record_tokens(self.morning_todo.name, "map_campus_route", alloc, route.get("token_estimate", 80))
                tech_routes[tech_id] = route

            result["remaining_tasks"] = len(ranked)
            result["new_wos_absorbed"] = len(new_wos)
            result["tech_routes"] = tech_routes

        except Exception as exc:
            state.errors.append(str(exc))
            logger.exception("Midday check error: %s", exc)

        self._close_state(state)
        result["token_usage"] = self.get_token_usage_report(cycle_id=state.cycle_id)
        return result

    def end_of_day_cycle(
        self,
        completed_wos: list[dict[str, Any]] | None = None,
        failed_wos: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Collect actuals, generate ADRs for failures, update Shaw Goals metrics.

        Parameters
        ----------
        completed_wos : list[dict]
            Work orders completed today with actuals.
        failed_wos : list[dict]
            Work orders that failed / missed SLA.

        Returns
        -------
        dict  End-of-day summary.
        """
        state = self._new_state(CycleType.END_OF_DAY)
        completed_wos = completed_wos or []
        failed_wos = failed_wos or []
        today = date.today()

        summary: dict[str, Any] = {
            "cycle_id": state.cycle_id,
            "date": today.isoformat(),
            "completed_count": len(completed_wos),
            "failed_count": len(failed_wos),
        }

        try:
            # Shaw Goals feedback
            alloc = self._allocate_tokens(self.shaw_goals.name)
            goals_eval = self.shaw_goals.evaluate_pm_completion(completed_wos, today)
            self._record_tokens(
                self.shaw_goals.name,
                "evaluate_pm_completion",
                alloc,
                goals_eval.get("token_estimate", alloc),
            )
            summary["shaw_goals_eod"] = goals_eval

            # ADR generation stubs for failures
            adr_ids: list[str] = []
            for wo in failed_wos:
                adr_id = f"ADR-{today.year}-{len(adr_ids) + 1:03d}"
                adr_ids.append(adr_id)
                self._send_message(
                    from_agent="Orchestrator",
                    to_agent="ADREngine",
                    msg_type=MessageType.REQUEST,
                    payload={"action": "create_adr_from_failure", "wo": wo, "adr_id": adr_id},
                )
                logger.info("Requested ADR %s for failed WO %s", adr_id, wo.get("wo_id"))
            summary["adrs_generated"] = adr_ids

            state.tasks_processed = len(completed_wos) + len(failed_wos)

        except Exception as exc:
            state.errors.append(str(exc))
            logger.exception("End-of-day cycle error: %s", exc)

        self._close_state(state)
        summary["token_usage"] = self.get_token_usage_report(cycle_id=state.cycle_id)
        return summary

    def handle_emergency(
        self,
        emergency_wo: dict[str, Any],
        technicians: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle a P1 emergency that arrives mid-day.

        Re-prioritises affected technicians' schedules and returns the
        updated routing.

        Parameters
        ----------
        emergency_wo : dict
            The emergency work order payload.
        technicians : list[str]
            Technician IDs eligible to respond.

        Returns
        -------
        dict  Emergency response plan.
        """
        technicians = technicians or []
        logger.warning(
            "EMERGENCY: P1 WO %s received -- re-prioritising %d technicians",
            emergency_wo.get("wo_id", "UNKNOWN"),
            len(technicians),
        )

        # Inject the emergency into a midday check with highest priority
        emergency_wo.setdefault("priority", 0)  # 0 = highest
        result = self.midday_check(
            new_wos=[emergency_wo],
            technicians=technicians,
        )
        result["emergency"] = True
        result["emergency_wo_id"] = emergency_wo.get("wo_id")

        self._send_message(
            from_agent="Orchestrator",
            to_agent="ALL",
            msg_type=MessageType.ALERT,
            payload={"emergency_wo": emergency_wo},
        )
        return result

    # ----- reporting --------------------------------------------------------

    def _generate_management_report(
        self,
        state: OrchestrationState,
        cycle_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce a concise management-level summary for Tony & Juan."""
        return {
            "date": state.date.isoformat(),
            "cycle": state.cycle_type.value,
            "total_tasks": state.tasks_processed,
            "errors": state.errors,
            "shaw_goals": cycle_data.get("shaw_goals", {}),
            "token_budget_remaining": state.token_budget_remaining,
            "technician_count": len(cycle_data.get("tech_routes", {})),
        }

    @staticmethod
    def _generate_tech_schedules(
        routes: dict[str, Any],
        ranked_tasks: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Map each technician to their ordered task list for the day."""
        schedules: dict[str, list[dict[str, Any]]] = {}
        for tech_id, route in routes.items():
            ordered_ids = route.get("ordered_tasks", [])
            tasks_map = {t.get("wo_id"): t for t in ranked_tasks}
            schedules[tech_id] = [
                tasks_map[wid] for wid in ordered_ids if wid in tasks_map
            ]
        return schedules

    def get_token_usage_report(
        self,
        cycle_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a structured token-usage report.

        Parameters
        ----------
        cycle_id : str, optional
            If provided, filter the ledger to entries that occurred during
            the given cycle.  Otherwise report cumulative totals.

        Returns
        -------
        dict  Breakdown by agent, by call_type, and totals.
        """
        entries = self._ledger  # default: everything

        if cycle_id:
            # Find the cycle's time window
            cycle = next(
                (s for s in self._history if s.cycle_id == cycle_id), None
            )
            if cycle and cycle.start_time and cycle.end_time:
                entries = [
                    e
                    for e in self._ledger
                    if cycle.start_time <= e.timestamp <= cycle.end_time
                ]

        by_agent: dict[str, int] = {}
        by_call: dict[str, int] = {}
        total_allocated = 0
        total_used = 0

        for e in entries:
            by_agent[e.agent_name] = by_agent.get(e.agent_name, 0) + e.tokens_used
            by_call[f"{e.agent_name}.{e.call_type}"] = (
                by_call.get(f"{e.agent_name}.{e.call_type}", 0) + e.tokens_used
            )
            total_allocated += e.tokens_allocated
            total_used += e.tokens_used

        return {
            "by_agent": by_agent,
            "by_call": by_call,
            "total_allocated": total_allocated,
            "total_used": total_used,
            "budget_per_cycle": self.max_tokens_per_cycle,
            "entries": len(entries),
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 72)
    print("  MAO Orchestrator -- demo run")
    print("=" * 72)

    orch = MAOOrchestrator(max_tokens_per_cycle=5000)

    # Simulate technicians
    techs = ["TECH-001", "TECH-002", "TECH-003"]

    # --- Morning cycle ---
    print("\n--- MORNING CYCLE ---")
    morning = orch.morning_cycle(technicians=techs)
    print(json.dumps(morning, indent=2, default=str))

    # --- Midday check ---
    print("\n--- MIDDAY CHECK ---")
    midday = orch.midday_check(
        completed_wo_ids=["WO-100", "WO-101"],
        new_wos=[{"wo_id": "WO-200", "priority": 2, "assigned_to": "TECH-001"}],
        technicians=techs,
    )
    print(json.dumps(midday, indent=2, default=str))

    # --- Emergency ---
    print("\n--- EMERGENCY P1 ---")
    emergency = orch.handle_emergency(
        emergency_wo={"wo_id": "WO-911", "description": "Roof leak - water intrusion", "priority": 0},
        technicians=techs,
    )
    print(json.dumps(emergency, indent=2, default=str))

    # --- End-of-day ---
    print("\n--- END OF DAY ---")
    eod = orch.end_of_day_cycle(
        completed_wos=[
            {"wo_id": "WO-100", "trade": "HVAC"},
            {"wo_id": "WO-101", "trade": "Plumbing"},
        ],
        failed_wos=[
            {"wo_id": "WO-102", "trade": "Electrical", "failure_reason": "Parts unavailable"},
        ],
    )
    print(json.dumps(eod, indent=2, default=str))

    # --- Token usage report ---
    print("\n--- CUMULATIVE TOKEN USAGE ---")
    usage = orch.get_token_usage_report()
    print(json.dumps(usage, indent=2, default=str))

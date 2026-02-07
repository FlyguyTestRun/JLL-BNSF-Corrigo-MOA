"""
ADR (Action Decision Record) Engine for JLL-BNSF Corrigo MAO.

When something goes wrong -- SLA breach, PM miss, rework, parts delay, safety
concern -- this engine auto-generates an ADR documenting what happened, the
root-cause analysis (5-why), the decision, consequences, and corrective actions.

ADRs also capture *improvements*: when a better method is discovered on-site,
the engine records it so future work benefits.

Designed for Python 3.11+.
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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

class ADRStatus(str, Enum):
    """Lifecycle states of an ADR."""
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    IMPLEMENTED = "implemented"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"


class FailureCategory(str, Enum):
    """Broad categories for classifying root causes."""
    PARTS = "parts"
    ACCESS = "access"
    SKILL = "skill"
    TOOL = "tool"
    PROCEDURE = "procedure"
    DESIGN = "design"


class TriggerType(str, Enum):
    """Events that can auto-create an ADR."""
    SLA_BREACH = "sla_breach"
    PM_OVERTIME = "pm_overtime"
    CHRONIC_FAILURE = "chronic_failure"
    REWORK_CALLBACK = "rework_callback"
    PARTS_DELAY = "parts_delay"
    SAFETY_CONCERN = "safety_concern"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ActionItem:
    """Single corrective-action task inside an ADR."""
    description: str
    owner: str
    due_date: date
    status: str = "open"  # open | in_progress | done


@dataclass
class ADR:
    """Action Decision Record.

    Attributes
    ----------
    adr_id : str
        Auto-incremented identifier in format ``ADR-YYYY-NNN``.
    date_created : date
        When this ADR was generated.
    title : str
        Short descriptive title.
    status : ADRStatus
        Current lifecycle state.
    context : str
        What situation triggered this ADR.
    decision : str
        What was decided.
    consequences : dict[str, list[str]]
        ``{"positive": [...], "negative": [...]}``.
    related_wo_ids : list[str]
        Work-order IDs involved.
    trade : str
        Trade discipline (HVAC, Electrical, Plumbing, ...).
    building : str
        Building / location code.
    asset_id : str
        Corrigo asset identifier.
    failure_category : FailureCategory | None
        Root-cause category.
    root_cause_chain : list[str]
        Ordered list of "why" answers (5-why analysis).
    action_items : list[ActionItem]
        Corrective actions.
    lessons_learned : str
        Free-text takeaway.
    superseded_by : str | None
        If this ADR is superseded, the ID of the replacement.
    trigger : TriggerType | None
        What auto-generation rule fired.
    """
    adr_id: str
    date_created: date
    title: str
    status: ADRStatus = ADRStatus.PROPOSED
    context: str = ""
    decision: str = ""
    consequences: dict[str, list[str]] = field(
        default_factory=lambda: {"positive": [], "negative": []}
    )
    related_wo_ids: list[str] = field(default_factory=list)
    trade: str = ""
    building: str = ""
    asset_id: str = ""
    failure_category: FailureCategory | None = None
    root_cause_chain: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    lessons_learned: str = ""
    superseded_by: str | None = None
    trigger: TriggerType | None = None

    # -- serialisation helpers -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "adr_id": self.adr_id,
            "date_created": self.date_created.isoformat(),
            "title": self.title,
            "status": self.status.value,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "related_wo_ids": self.related_wo_ids,
            "trade": self.trade,
            "building": self.building,
            "asset_id": self.asset_id,
            "failure_category": self.failure_category.value if self.failure_category else None,
            "root_cause_chain": self.root_cause_chain,
            "action_items": [
                {
                    "description": ai.description,
                    "owner": ai.owner,
                    "due_date": ai.due_date.isoformat(),
                    "status": ai.status,
                }
                for ai in self.action_items
            ],
            "lessons_learned": self.lessons_learned,
            "superseded_by": self.superseded_by,
            "trigger": self.trigger.value if self.trigger else None,
        }

    def to_markdown(self) -> str:
        """Render the ADR as a Markdown document."""
        whys = "\n".join(
            f"  {i}. {w}" for i, w in enumerate(self.root_cause_chain, 1)
        ) or "  (none recorded)"

        actions = "\n".join(
            f"  - [{ai.status}] {ai.description} (owner: {ai.owner}, due: {ai.due_date})"
            for ai in self.action_items
        ) or "  (none)"

        pos = "\n".join(f"  - {c}" for c in self.consequences.get("positive", [])) or "  (none)"
        neg = "\n".join(f"  - {c}" for c in self.consequences.get("negative", [])) or "  (none)"

        return textwrap.dedent(f"""\
            # {self.adr_id}: {self.title}

            **Date:** {self.date_created}
            **Status:** {self.status.value}
            **Trade:** {self.trade}
            **Building:** {self.building}
            **Asset:** {self.asset_id}
            **Category:** {self.failure_category.value if self.failure_category else 'N/A'}
            **Related WOs:** {', '.join(self.related_wo_ids) or 'none'}

            ## Context
            {self.context}

            ## Root-Cause Analysis (5-Why)
            {whys}

            ## Decision
            {self.decision}

            ## Consequences
            **Positive:**
            {pos}

            **Negative:**
            {neg}

            ## Action Items
            {actions}

            ## Lessons Learned
            {self.lessons_learned}
        """)

    def to_html(self) -> str:
        """Render the ADR as a simple HTML fragment."""
        rows = "".join(
            f"<tr><td>{ai.description}</td><td>{ai.owner}</td>"
            f"<td>{ai.due_date}</td><td>{ai.status}</td></tr>"
            for ai in self.action_items
        )
        whys_html = "<ol>" + "".join(f"<li>{w}</li>" for w in self.root_cause_chain) + "</ol>"
        return textwrap.dedent(f"""\
            <div class="adr" id="{self.adr_id}">
              <h2>{self.adr_id}: {self.title}</h2>
              <p><strong>Date:</strong> {self.date_created} |
                 <strong>Status:</strong> {self.status.value} |
                 <strong>Trade:</strong> {self.trade} |
                 <strong>Building:</strong> {self.building}</p>
              <h3>Context</h3><p>{self.context}</p>
              <h3>Root-Cause Chain</h3>{whys_html}
              <h3>Decision</h3><p>{self.decision}</p>
              <h3>Action Items</h3>
              <table border="1"><tr><th>Description</th><th>Owner</th><th>Due</th><th>Status</th></tr>{rows}</table>
              <h3>Lessons Learned</h3><p>{self.lessons_learned}</p>
            </div>
        """)


@dataclass
class ADRStore:
    """In-memory ADR repository with secondary indices for fast look-ups."""
    adrs: list[ADR] = field(default_factory=list)
    index_by_trade: dict[str, list[str]] = field(default_factory=dict)
    index_by_building: dict[str, list[str]] = field(default_factory=dict)
    index_by_category: dict[str, list[str]] = field(default_factory=dict)

    def add(self, adr: ADR) -> None:
        """Add an ADR and update all indices."""
        self.adrs.append(adr)
        self.index_by_trade.setdefault(adr.trade, []).append(adr.adr_id)
        self.index_by_building.setdefault(adr.building, []).append(adr.adr_id)
        if adr.failure_category:
            self.index_by_category.setdefault(
                adr.failure_category.value, []
            ).append(adr.adr_id)

    def find_by_id(self, adr_id: str) -> ADR | None:
        return next((a for a in self.adrs if a.adr_id == adr_id), None)


# ---------------------------------------------------------------------------
# Auto-generation trigger definitions
# ---------------------------------------------------------------------------

@dataclass
class TriggerRule:
    """Definition of one auto-generation trigger."""
    trigger_type: TriggerType
    description: str
    evaluate: Any  # callable(wo_data) -> bool


def _sla_breach(wo: dict[str, Any]) -> bool:
    """SLA breach on any work order."""
    return wo.get("sla_breached", False)


def _pm_overtime(wo: dict[str, Any]) -> bool:
    """PM completed more than 20 % over estimated time."""
    estimated = wo.get("estimated_minutes", 0)
    actual = wo.get("actual_minutes", 0)
    if estimated <= 0:
        return False
    return actual > estimated * 1.2


def _chronic_failure(wo: dict[str, Any]) -> bool:
    """Same asset has 3+ WOs in 30 days (caller must pre-populate ``recent_wo_count``)."""
    return wo.get("recent_wo_count", 0) >= 3


def _rework_callback(wo: dict[str, Any]) -> bool:
    """Rework/callback on a completed WO within 7 days."""
    return wo.get("is_rework", False)


def _parts_delay(wo: dict[str, Any]) -> bool:
    """Parts not available causing delay > 2 hours."""
    delay_minutes = wo.get("parts_delay_minutes", 0)
    return delay_minutes > 120


def _safety_concern(wo: dict[str, Any]) -> bool:
    """Technician reports safety concern."""
    return wo.get("safety_concern", False)


DEFAULT_TRIGGER_RULES: list[TriggerRule] = [
    TriggerRule(TriggerType.SLA_BREACH, "SLA breach on any work order", _sla_breach),
    TriggerRule(TriggerType.PM_OVERTIME, "PM completed >20% over estimate", _pm_overtime),
    TriggerRule(TriggerType.CHRONIC_FAILURE, "Asset has 3+ WOs in 30 days", _chronic_failure),
    TriggerRule(TriggerType.REWORK_CALLBACK, "Rework/callback within 7 days", _rework_callback),
    TriggerRule(TriggerType.PARTS_DELAY, "Parts delay >2 hours", _parts_delay),
    TriggerRule(TriggerType.SAFETY_CONCERN, "Technician-reported safety concern", _safety_concern),
]


# ---------------------------------------------------------------------------
# ADR Engine
# ---------------------------------------------------------------------------

class ADREngine:
    """Engine that creates, stores, searches, and exports ADRs.

    Automatically generates ADRs when trigger conditions are met and
    maintains a searchable, indexed in-memory store.

    Parameters
    ----------
    trigger_rules : list[TriggerRule], optional
        Custom trigger rules.  Defaults to ``DEFAULT_TRIGGER_RULES``.
    """

    def __init__(
        self,
        trigger_rules: list[TriggerRule] | None = None,
    ) -> None:
        self.store = ADRStore()
        self._counter: int = 0
        self._trigger_rules = trigger_rules or list(DEFAULT_TRIGGER_RULES)
        logger.info(
            "ADREngine initialised with %d trigger rules", len(self._trigger_rules)
        )

    # ----- ID generation ----------------------------------------------------

    def _next_id(self) -> str:
        self._counter += 1
        return f"ADR-{date.today().year}-{self._counter:03d}"

    # ----- core creation methods --------------------------------------------

    def create_adr_from_failure(
        self,
        wo_data: dict[str, Any],
        root_causes: list[str] | None = None,
        decision: str = "",
        action_items: list[dict[str, Any]] | None = None,
    ) -> ADR:
        """Generate an ADR from a failed or problematic work order.

        Parameters
        ----------
        wo_data : dict
            Work-order payload.  Expected keys include ``wo_id``, ``trade``,
            ``building``, ``asset_id``, ``description``, ``failure_reason``,
            and optionally ``sla_breached``, ``actual_minutes``, etc.
        root_causes : list[str], optional
            Pre-populated 5-why chain.  If omitted a placeholder chain is
            generated from ``failure_reason``.
        decision : str, optional
            The corrective decision.  Auto-generated if blank.
        action_items : list[dict], optional
            Each dict should contain ``description``, ``owner``, ``due_date``
            (ISO string or ``date``), and optionally ``status``.

        Returns
        -------
        ADR  The newly created record (also stored internally).
        """
        wo_id = wo_data.get("wo_id", "UNKNOWN")
        trade = wo_data.get("trade", "General")
        building = wo_data.get("building", "")
        asset_id = wo_data.get("asset_id", "")
        failure_reason = wo_data.get("failure_reason", "Unspecified failure")
        description = wo_data.get("description", "")

        # Determine failure category heuristically
        category = self._infer_category(wo_data)

        # Build 5-why chain
        if root_causes:
            chain = root_causes
        else:
            chain = [
                f"Work order {wo_id} failed: {failure_reason}",
                f"Why? {failure_reason}",
                "Why? (root cause investigation pending)",
                "Why? (to be determined)",
                "Why? (to be determined)",
            ]

        # Determine trigger
        trigger = self._match_trigger(wo_data)

        # Auto-generate decision text if not provided
        if not decision:
            decision = (
                f"Investigate root cause of '{failure_reason}' on asset {asset_id} "
                f"in building {building}. Implement corrective action within 7 days."
            )

        # Build action items
        parsed_actions: list[ActionItem] = []
        for item in action_items or []:
            due = item.get("due_date", date.today() + timedelta(days=7))
            if isinstance(due, str):
                due = date.fromisoformat(due)
            parsed_actions.append(
                ActionItem(
                    description=item.get("description", ""),
                    owner=item.get("owner", "Unassigned"),
                    due_date=due,
                    status=item.get("status", "open"),
                )
            )
        if not parsed_actions:
            parsed_actions.append(
                ActionItem(
                    description=f"Investigate and resolve: {failure_reason}",
                    owner="Site Lead",
                    due_date=date.today() + timedelta(days=7),
                )
            )

        adr = ADR(
            adr_id=self._next_id(),
            date_created=date.today(),
            title=f"Failure on WO {wo_id}: {failure_reason[:80]}",
            status=ADRStatus.PROPOSED,
            context=(
                f"Work order {wo_id} ({description}) in building {building} "
                f"experienced a failure: {failure_reason}."
            ),
            decision=decision,
            consequences={
                "positive": ["Issue documented for future prevention"],
                "negative": [f"Service disruption on WO {wo_id}"],
            },
            related_wo_ids=[wo_id],
            trade=trade,
            building=building,
            asset_id=asset_id,
            failure_category=category,
            root_cause_chain=chain,
            action_items=parsed_actions,
            lessons_learned="",
            trigger=trigger,
        )

        self.store.add(adr)
        logger.info("Created failure ADR %s for WO %s [%s]", adr.adr_id, wo_id, trigger)
        return adr

    def create_adr_from_improvement(
        self,
        title: str,
        context: str,
        decision: str,
        trade: str = "",
        building: str = "",
        asset_id: str = "",
        lessons_learned: str = "",
        related_wo_ids: list[str] | None = None,
    ) -> ADR:
        """Record a discovered improvement as an ADR for future reference.

        Parameters
        ----------
        title : str
            Short title describing the improvement.
        context : str
            What prompted the discovery.
        decision : str
            The new/better approach.
        trade, building, asset_id : str
            Classification fields.
        lessons_learned : str
            Free-text takeaway.
        related_wo_ids : list[str]
            Any associated WOs.

        Returns
        -------
        ADR  The created record.
        """
        adr = ADR(
            adr_id=self._next_id(),
            date_created=date.today(),
            title=title,
            status=ADRStatus.ACCEPTED,
            context=context,
            decision=decision,
            consequences={
                "positive": ["Improved process documented"],
                "negative": [],
            },
            related_wo_ids=related_wo_ids or [],
            trade=trade,
            building=building,
            asset_id=asset_id,
            lessons_learned=lessons_learned,
        )

        self.store.add(adr)
        logger.info("Created improvement ADR %s: %s", adr.adr_id, title)
        return adr

    # ----- search & recommendations -----------------------------------------

    def search_adrs(
        self,
        trade: str | None = None,
        building: str | None = None,
        asset_id: str | None = None,
        failure_category: FailureCategory | str | None = None,
        status: ADRStatus | None = None,
        keyword: str | None = None,
    ) -> list[ADR]:
        """Find ADRs matching the given filters.

        All filters are combined with AND logic.  Omitted filters are ignored.

        Returns
        -------
        list[ADR]  Matching records, newest first.
        """
        candidates: list[ADR] = list(self.store.adrs)

        if trade is not None:
            ids = set(self.store.index_by_trade.get(trade, []))
            candidates = [a for a in candidates if a.adr_id in ids]

        if building is not None:
            ids = set(self.store.index_by_building.get(building, []))
            candidates = [a for a in candidates if a.adr_id in ids]

        if asset_id is not None:
            candidates = [a for a in candidates if a.asset_id == asset_id]

        if failure_category is not None:
            cat_val = (
                failure_category.value
                if isinstance(failure_category, FailureCategory)
                else failure_category
            )
            ids = set(self.store.index_by_category.get(cat_val, []))
            candidates = [a for a in candidates if a.adr_id in ids]

        if status is not None:
            candidates = [a for a in candidates if a.status == status]

        if keyword is not None:
            kw = keyword.lower()
            candidates = [
                a
                for a in candidates
                if kw in a.title.lower()
                or kw in a.context.lower()
                or kw in a.decision.lower()
                or kw in a.lessons_learned.lower()
            ]

        # Newest first
        candidates.sort(key=lambda a: a.date_created, reverse=True)
        return candidates

    def get_recommendations(
        self,
        task: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Given a new task, find ADRs with relevant lessons learned.

        Parameters
        ----------
        task : dict
            Incoming task / WO payload.

        Returns
        -------
        list[dict]  Each dict has ``adr_id``, ``title``, ``lesson``, ``relevance``.
        """
        trade = task.get("trade")
        building = task.get("building")
        asset_id = task.get("asset_id")

        hits: list[ADR] = []
        # Priority 1: same asset
        if asset_id:
            hits.extend(self.search_adrs(asset_id=asset_id))
        # Priority 2: same building + trade
        if building and trade:
            hits.extend(self.search_adrs(trade=trade, building=building))
        # Priority 3: same trade
        if trade:
            hits.extend(self.search_adrs(trade=trade))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[ADR] = []
        for adr in hits:
            if adr.adr_id not in seen:
                seen.add(adr.adr_id)
                unique.append(adr)

        recommendations: list[dict[str, Any]] = []
        for adr in unique[:10]:  # top 10
            relevance = "low"
            if adr.asset_id == asset_id and asset_id:
                relevance = "high"
            elif adr.building == building and adr.trade == trade:
                relevance = "medium"
            recommendations.append(
                {
                    "adr_id": adr.adr_id,
                    "title": adr.title,
                    "lesson": adr.lessons_learned or adr.decision,
                    "relevance": relevance,
                }
            )
        return recommendations

    # ----- export -----------------------------------------------------------

    def export_adrs(
        self,
        fmt: str = "json",
        adrs: list[ADR] | None = None,
    ) -> str:
        """Export ADRs in the requested format.

        Parameters
        ----------
        fmt : str
            One of ``"json"``, ``"markdown"``, ``"html"``.
        adrs : list[ADR], optional
            Subset to export.  Defaults to all ADRs in the store.

        Returns
        -------
        str  The rendered output.
        """
        records = adrs if adrs is not None else self.store.adrs

        if fmt == "json":
            return json.dumps(
                [a.to_dict() for a in records], indent=2, default=str
            )

        if fmt == "markdown":
            return "\n---\n\n".join(a.to_markdown() for a in records)

        if fmt == "html":
            body = "\n<hr>\n".join(a.to_html() for a in records)
            return (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>ADR Report</title></head><body>\n"
                f"{body}\n</body></html>"
            )

        raise ValueError(f"Unsupported export format: {fmt!r}. Use json, markdown, or html.")

    def generate_weekly_adr_summary(
        self,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        """Generate a weekly digest of new ADRs, patterns, and systemic issues.

        Parameters
        ----------
        as_of : date, optional
            End date of the reporting week.  Defaults to today.

        Returns
        -------
        dict  Structured summary.
        """
        as_of = as_of or date.today()
        week_start = as_of - timedelta(days=7)

        weekly = [
            a
            for a in self.store.adrs
            if week_start <= a.date_created <= as_of
        ]

        # Detect patterns
        trade_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        building_counts: dict[str, int] = {}
        for a in weekly:
            trade_counts[a.trade] = trade_counts.get(a.trade, 0) + 1
            if a.failure_category:
                cat = a.failure_category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1
            building_counts[a.building] = building_counts.get(a.building, 0) + 1

        # Flag systemic issues (any dimension with 3+ ADRs in the week)
        systemic: list[dict[str, Any]] = []
        for trade, count in trade_counts.items():
            if count >= 3:
                systemic.append({"dimension": "trade", "value": trade, "count": count})
        for cat, count in category_counts.items():
            if count >= 3:
                systemic.append({"dimension": "category", "value": cat, "count": count})
        for bldg, count in building_counts.items():
            if count >= 3:
                systemic.append({"dimension": "building", "value": bldg, "count": count})

        return {
            "week_ending": as_of.isoformat(),
            "total_new_adrs": len(weekly),
            "by_trade": trade_counts,
            "by_category": category_counts,
            "by_building": building_counts,
            "systemic_issues": systemic,
            "adr_ids": [a.adr_id for a in weekly],
        }

    # ----- trigger evaluation -----------------------------------------------

    def evaluate_triggers(
        self,
        wo_data: dict[str, Any],
        auto_create: bool = True,
    ) -> list[TriggerType]:
        """Evaluate all trigger rules against a work order.

        Parameters
        ----------
        wo_data : dict
            Work-order data.
        auto_create : bool
            If ``True``, automatically create an ADR for each matched trigger.

        Returns
        -------
        list[TriggerType]  Triggers that fired.
        """
        fired: list[TriggerType] = []
        for rule in self._trigger_rules:
            try:
                if rule.evaluate(wo_data):
                    fired.append(rule.trigger_type)
                    logger.info(
                        "Trigger fired: %s on WO %s",
                        rule.trigger_type.value,
                        wo_data.get("wo_id", "UNKNOWN"),
                    )
            except Exception as exc:
                logger.warning(
                    "Trigger %s evaluation error: %s", rule.trigger_type.value, exc
                )

        if auto_create and fired:
            self.create_adr_from_failure(wo_data)

        return fired

    # ----- corrective script generation -------------------------------------

    @staticmethod
    def generate_corrective_script(
        adr: ADR,
        script_type: str = "python",
    ) -> str:
        """Produce a small script snippet to automate the corrective action.

        The generated script is illustrative and intended to be reviewed by
        a human before execution.

        Parameters
        ----------
        adr : ADR
            The ADR whose corrective action should be scripted.
        script_type : str
            ``"python"`` or ``"shell"``.

        Returns
        -------
        str  Script source code.
        """
        if script_type == "shell":
            return _generate_shell_script(adr)
        return _generate_python_script(adr)

    # ----- private helpers --------------------------------------------------

    @staticmethod
    def _infer_category(wo_data: dict[str, Any]) -> FailureCategory | None:
        """Heuristically map work-order data to a FailureCategory."""
        reason = (wo_data.get("failure_reason") or "").lower()
        if any(kw in reason for kw in ("part", "supply", "material", "stock")):
            return FailureCategory.PARTS
        if any(kw in reason for kw in ("access", "locked", "key", "permission")):
            return FailureCategory.ACCESS
        if any(kw in reason for kw in ("skill", "training", "certif")):
            return FailureCategory.SKILL
        if any(kw in reason for kw in ("tool", "equipment", "instrument")):
            return FailureCategory.TOOL
        if any(kw in reason for kw in ("procedure", "process", "checklist", "step")):
            return FailureCategory.PROCEDURE
        if any(kw in reason for kw in ("design", "spec", "engineer", "drawing")):
            return FailureCategory.DESIGN
        return None

    def _match_trigger(self, wo_data: dict[str, Any]) -> TriggerType | None:
        """Return the first trigger that matches, or None."""
        for rule in self._trigger_rules:
            try:
                if rule.evaluate(wo_data):
                    return rule.trigger_type
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# Script generators (module-level helpers)
# ---------------------------------------------------------------------------

def _generate_python_script(adr: ADR) -> str:
    """Generate a Python corrective-action script for the given ADR."""
    action_desc = (
        adr.action_items[0].description if adr.action_items else "No action specified"
    )
    asset = adr.asset_id or "UNKNOWN_ASSET"
    building = adr.building or "UNKNOWN_BUILDING"
    trade = adr.trade or "General"
    wo_ids = ", ".join(f'"{w}"' for w in adr.related_wo_ids) or '"UNKNOWN"'

    lines: list[str] = [
        '"""',
        f"Auto-generated corrective script for {adr.adr_id}",
        f"Title : {adr.title}",
        f"Action: {action_desc}",
        f"Date  : {adr.date_created}",
        '"""',
        "",
        "import logging",
        "from datetime import date, timedelta",
        "",
        "logger = logging.getLogger(__name__)",
        "",
    ]

    # Category-specific logic
    if adr.failure_category == FailureCategory.PARTS:
        lines.extend([
            f'ASSET_ID = "{asset}"',
            f'BUILDING = "{building}"',
            f"RELATED_WOS = [{wo_ids}]",
            "",
            "def auto_reorder_part():",
            '    """Submit a parts reorder request to the procurement system."""',
            f'    part_description = "{action_desc}"',
            "    logger.info('Submitting reorder for %s at %s', part_description, BUILDING)",
            "    # TODO: integrate with procurement API",
            f'    print(f"[REORDER] Part request created for asset {{ASSET_ID}} in {{BUILDING}}")',
            "",
            "",
            'if __name__ == "__main__":',
            "    auto_reorder_part()",
        ])
    elif adr.failure_category in (FailureCategory.PROCEDURE, FailureCategory.DESIGN):
        lines.extend([
            f'ASSET_ID = "{asset}"',
            "",
            "def schedule_follow_up_inspection():",
            '    """Schedule a follow-up inspection within 14 days."""',
            "    due = date.today() + timedelta(days=14)",
            f'    logger.info("Scheduling inspection for asset %s by %s", ASSET_ID, due)',
            '    # TODO: create a scheduled WO in Corrigo',
            f'    print(f"[INSPECTION] Follow-up scheduled for asset {{ASSET_ID}} on {{due}}")',
            "",
            "",
            'if __name__ == "__main__":',
            "    schedule_follow_up_inspection()",
        ])
    else:
        # Generic: flag for replacement review / general follow-up
        lines.extend([
            f'ASSET_ID = "{asset}"',
            f'TRADE = "{trade}"',
            "",
            "def flag_for_review():",
            '    """Flag the asset for replacement or engineering review."""',
            f'    logger.info("Flagging asset %s (%s) for review", ASSET_ID, TRADE)',
            '    # TODO: update asset status in Corrigo',
            f'    print(f"[REVIEW] Asset {{ASSET_ID}} flagged for {{TRADE}} review")',
            "",
            "",
            'if __name__ == "__main__":',
            "    flag_for_review()",
        ])

    return "\n".join(lines) + "\n"


def _generate_shell_script(adr: ADR) -> str:
    """Generate a shell corrective-action script for the given ADR."""
    action_desc = (
        adr.action_items[0].description if adr.action_items else "No action specified"
    )
    asset = adr.asset_id or "UNKNOWN_ASSET"
    building = adr.building or "UNKNOWN_BUILDING"

    return textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # Auto-generated corrective script for {adr.adr_id}
        # Title : {adr.title}
        # Action: {action_desc}
        # Date  : {adr.date_created}

        set -euo pipefail

        ASSET_ID="{asset}"
        BUILDING="{building}"

        echo "[INFO] Running corrective action for $ASSET_ID in $BUILDING"

        # Example: create a follow-up work-order via Corrigo CLI/API
        # corrigo-cli create-wo \\
        #   --asset "$ASSET_ID" \\
        #   --building "$BUILDING" \\
        #   --priority 3 \\
        #   --description "Follow-up: {action_desc}"

        echo "[DONE] Corrective script completed for {adr.adr_id}"
    """)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  ADR Engine -- demo run")
    print("=" * 72)

    engine = ADREngine()

    # --- Create ADRs from failures ---
    print("\n--- Creating ADRs from failures ---")

    adr1 = engine.create_adr_from_failure(
        wo_data={
            "wo_id": "WO-5001",
            "trade": "HVAC",
            "building": "BLD-A12",
            "asset_id": "RTU-042",
            "description": "Quarterly filter replacement",
            "failure_reason": "Parts unavailable - filters out of stock",
            "sla_breached": True,
            "parts_delay_minutes": 180,
        },
        root_causes=[
            "PM WO-5001 could not be completed on schedule",
            "Required filters were not in stock at the facility",
            "Reorder point was not triggered in procurement system",
            "Inventory count was inaccurate after last audit",
            "Manual inventory process has no automated reconciliation",
        ],
        action_items=[
            {
                "description": "Implement automated reorder alerts for HVAC filters",
                "owner": "Procurement Lead",
                "due_date": (date.today() + timedelta(days=14)).isoformat(),
            },
            {
                "description": "Conduct inventory audit for BLD-A12 HVAC parts",
                "owner": "Site Lead",
                "due_date": (date.today() + timedelta(days=7)).isoformat(),
            },
        ],
    )

    adr2 = engine.create_adr_from_failure(
        wo_data={
            "wo_id": "WO-5023",
            "trade": "Electrical",
            "building": "BLD-C03",
            "asset_id": "PNL-117",
            "description": "Panel inspection - annual",
            "failure_reason": "Access denied - locked electrical room, no key available",
            "sla_breached": False,
            "estimated_minutes": 60,
            "actual_minutes": 180,
        },
    )

    adr3 = engine.create_adr_from_failure(
        wo_data={
            "wo_id": "WO-5044",
            "trade": "HVAC",
            "building": "BLD-A12",
            "asset_id": "RTU-042",
            "description": "Emergency repair - unit not cooling",
            "failure_reason": "Parts unavailable - compressor contactor",
            "is_rework": True,
            "recent_wo_count": 4,
        },
    )

    # --- Create improvement ADR ---
    print("\n--- Creating improvement ADR ---")
    adr_imp = engine.create_adr_from_improvement(
        title="Pre-stage HVAC filters at BLD-A12 quarterly",
        context="Repeated filter stock-outs at BLD-A12 causing PM delays.",
        decision=(
            "Procurement will pre-stage a quarter's worth of HVAC filters "
            "at BLD-A12 storage room 30 days before each quarterly PM cycle."
        ),
        trade="HVAC",
        building="BLD-A12",
        lessons_learned="Pre-staging eliminates same-day parts delays for predictable PMs.",
        related_wo_ids=["WO-5001", "WO-5044"],
    )

    # --- Search ---
    print("\n--- Searching ADRs (trade=HVAC) ---")
    results = engine.search_adrs(trade="HVAC")
    for r in results:
        print(f"  {r.adr_id}: {r.title}")

    # --- Recommendations ---
    print("\n--- Recommendations for new HVAC task in BLD-A12 ---")
    recs = engine.get_recommendations(
        {"trade": "HVAC", "building": "BLD-A12", "asset_id": "RTU-042"}
    )
    for rec in recs:
        print(f"  [{rec['relevance']}] {rec['adr_id']}: {rec['lesson'][:80]}")

    # --- Trigger evaluation ---
    print("\n--- Evaluating triggers on a new WO ---")
    triggers = engine.evaluate_triggers(
        {
            "wo_id": "WO-6000",
            "trade": "Plumbing",
            "building": "BLD-B07",
            "failure_reason": "Pipe leak",
            "sla_breached": True,
            "safety_concern": True,
        },
        auto_create=True,
    )
    print(f"  Triggers fired: {[t.value for t in triggers]}")

    # --- Weekly summary ---
    print("\n--- Weekly ADR Summary ---")
    summary = engine.generate_weekly_adr_summary()
    print(json.dumps(summary, indent=2, default=str))

    # --- Export (markdown) ---
    print("\n--- Markdown export (first ADR) ---")
    print(engine.export_adrs(fmt="markdown", adrs=[adr1]))

    # --- Corrective script ---
    print("\n--- Corrective Python script for ADR-1 ---")
    print(engine.generate_corrective_script(adr1, script_type="python"))

    print("\n--- Corrective shell script for ADR-2 ---")
    print(engine.generate_corrective_script(adr2, script_type="shell"))

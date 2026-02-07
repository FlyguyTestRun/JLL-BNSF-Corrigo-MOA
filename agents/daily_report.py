"""Daily Report Generator for JLL-BNSF Corrigo MAO.

Produces a concise, easily digestible daily pull of current work orders and
maintenance team activities at the BNSF Railway HQ campus in Fort Worth, TX.

Primary recipients:
    - Tony Vita, Facilities Manager
    - Juan Guerra, Maintenance Manager

This is the FIRST deliverable of the MAO system -- the report Tony and Juan
open every morning to understand the state of all maintenance operations.

Python 3.11+
"""

from __future__ import annotations

import json
import logging
import os
import random
import smtplib
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("daily_report")
logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
)
if not logger.handlers:
    logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    """Work order priority levels."""

    P1 = "P1"  # Emergency / Life Safety
    P2 = "P2"  # Urgent
    P3 = "P3"  # High
    P4 = "P4"  # Routine
    P5 = "P5"  # Low / Scheduled

    @property
    def label(self) -> str:
        labels = {
            "P1": "Emergency",
            "P2": "Urgent",
            "P3": "High",
            "P4": "Routine",
            "P5": "Low",
        }
        return labels[self.value]

    @property
    def sla_hours(self) -> int:
        """Default SLA response window in hours per priority."""
        windows = {"P1": 1, "P2": 4, "P3": 8, "P4": 24, "P5": 72}
        return windows[self.value]


class SLAStatus(str, Enum):
    """Traffic-light SLA status."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class WOStatus(str, Enum):
    """Work order lifecycle status."""

    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    ON_HOLD = "On Hold"
    PENDING_VENDOR = "Pending Vendor"
    COMPLETED = "Completed"
    CLOSED = "Closed"


class TechStatus(str, Enum):
    """Technician availability status."""

    ACTIVE = "active"
    BREAK = "break"
    OFF_SITE = "off-site"
    NOT_STARTED = "not-started"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """Configuration for a daily report run."""

    report_date: date
    recipients: list[str] = field(default_factory=lambda: [
        "tony.vita@jll.com",
        "juan.guerra@jll.com",
    ])
    campus_name: str = "BNSF HQ Campus"
    include_sections: list[str] = field(default_factory=lambda: [
        "executive_summary",
        "work_order_status",
        "pm_schedule",
        "technician_workload",
        "sla_watchlist",
        "yesterdays_completions",
        "vendor_activity",
        "weekly_trends",
        "action_items",
        "adr_highlights",
    ])
    sla_warning_hours: int = 4
    timezone: str = "America/Chicago"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["report_date"] = self.report_date.isoformat()
        return data


@dataclass
class WorkOrderSummary:
    """A single work order record for report display."""

    wo_number: str
    title: str
    building: str
    floor: str
    trade: str
    priority: Priority
    assigned_tech: str
    created_at: datetime
    sla_deadline: datetime
    sla_status: SLAStatus
    age_hours: float
    status: WOStatus

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["priority"] = self.priority.value
        data["sla_status"] = self.sla_status.value
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["sla_deadline"] = self.sla_deadline.isoformat()
        data["age_hours"] = round(self.age_hours, 1)
        return data


@dataclass
class TechnicianStatus:
    """Per-technician workload snapshot."""

    tech_id: str
    name: str
    trade: str
    tasks_today: int
    estimated_hours: float
    completed_today: int
    current_task: str
    status: TechStatus

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["estimated_hours"] = round(self.estimated_hours, 1)
        return data


@dataclass
class DailyMetrics:
    """Aggregate metrics for one day of operations."""

    date: date
    total_open: int
    total_closed_today: int
    total_created_today: int
    avg_cycle_hours: float
    pm_due: int
    pm_completed: int
    pm_compliance_pct: float
    sla_breaches: int
    callbacks: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["date"] = self.date.isoformat()
        data["avg_cycle_hours"] = round(self.avg_cycle_hours, 1)
        data["pm_compliance_pct"] = round(self.pm_compliance_pct, 1)
        return data


@dataclass
class PMTask:
    """A preventive-maintenance task for today's schedule."""

    pm_id: str
    title: str
    building: str
    floor: str
    trade: str
    assigned_tech: str
    due_date: date
    frequency: str
    is_overdue: bool
    is_completed: bool
    estimated_minutes: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["due_date"] = self.due_date.isoformat()
        return data


@dataclass
class VendorWorkOrder:
    """An externally-assigned vendor work order."""

    wo_number: str
    vendor_name: str
    description: str
    building: str
    status: str
    scheduled_date: date | None
    invoice_status: str
    estimated_cost: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scheduled_date"] = (
            self.scheduled_date.isoformat() if self.scheduled_date else None
        )
        return data


@dataclass
class ActionItem:
    """A flagged item needing management attention."""

    item_id: str
    description: str
    urgency: str  # "high", "medium", "low"
    related_wo: str | None
    owner: str  # "Tony Vita" or "Juan Guerra"
    due_by: str  # e.g. "EOD", "noon", "ASAP"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ADRHighlight:
    """An Architectural Decision Record highlight for management review."""

    adr_id: str
    title: str
    status: str  # "proposed", "accepted", "superseded"
    summary: str
    impact_area: str
    date_created: date

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["date_created"] = self.date_created.isoformat()
        return data


# ---------------------------------------------------------------------------
# Daily Report Generator
# ---------------------------------------------------------------------------

class DailyReportGenerator:
    """Assembles and formats the daily maintenance operations report.

    This is the first thing Tony Vita and Juan Guerra see each morning.
    Everything is designed for quick scanning: color-coded priorities,
    traffic-light SLA indicators, and clear action items at the top.
    """

    CAMPUS_ADDRESS = "2650 Lou Menk Dr, Fort Worth, TX 76131"

    def __init__(
        self,
        report_date: date | None = None,
        campus_name: str = "BNSF HQ Campus",
    ) -> None:
        self.report_date = report_date or date.today()
        self.campus_name = campus_name
        self.config = ReportConfig(
            report_date=self.report_date,
            campus_name=campus_name,
        )

        # Data containers -- populated by generate_full_report or individually
        self.work_orders: list[WorkOrderSummary] = []
        self.technicians: list[TechnicianStatus] = []
        self.pm_tasks: list[PMTask] = []
        self.vendor_orders: list[VendorWorkOrder] = []
        self.action_items: list[ActionItem] = []
        self.adr_highlights: list[ADRHighlight] = []
        self.daily_metrics: DailyMetrics | None = None
        self.weekly_metrics: list[DailyMetrics] = []

        # Report sections cache
        self._sections: dict[str, Any] = {}
        self._generated_at: datetime | None = None

        logger.info(
            "DailyReportGenerator initialized | date=%s | campus=%s",
            self.report_date.isoformat(),
            self.campus_name,
        )

    # ------------------------------------------------------------------
    # Master generator
    # ------------------------------------------------------------------

    def generate_full_report(self) -> dict[str, Any]:
        """Master method: call all sub-generators, assemble final report.

        Returns
        -------
        dict
            The complete report as a JSON-serializable dictionary.
        """
        logger.info("Generating full daily report for %s", self.report_date)
        self._generated_at = datetime.now()

        self._sections = {
            "executive_summary": self.section_executive_summary(),
            "work_order_status": self.section_work_order_status(),
            "pm_schedule": self.section_pm_schedule(),
            "technician_workload": self.section_technician_workload(),
            "sla_watchlist": self.section_sla_watchlist(),
            "yesterdays_completions": self.section_yesterdays_completions(),
            "vendor_activity": self.section_vendor_activity(),
            "weekly_trends": self.section_weekly_trends(),
            "action_items": self.section_action_items(),
            "adr_highlights": self.section_adr_highlights(),
        }

        report = {
            "report_date": self.report_date.isoformat(),
            "campus_name": self.campus_name,
            "generated_at": self._generated_at.isoformat(),
            "config": self.config.to_dict(),
            "sections": self._sections,
        }
        logger.info("Full report generated with %d sections", len(self._sections))
        return report

    # ------------------------------------------------------------------
    # Section generators
    # ------------------------------------------------------------------

    def section_executive_summary(self) -> dict[str, Any]:
        """3-4 bullet executive summary for quick morning scan.

        Covers: total open WOs, PMs due today, SLA risks, staffing status.
        """
        open_wos = [
            wo for wo in self.work_orders
            if wo.status in (WOStatus.OPEN, WOStatus.IN_PROGRESS, WOStatus.ON_HOLD)
        ]
        red_sla = [wo for wo in open_wos if wo.sla_status == SLAStatus.RED]
        yellow_sla = [wo for wo in open_wos if wo.sla_status == SLAStatus.YELLOW]
        pms_due = [pm for pm in self.pm_tasks if pm.due_date <= self.report_date and not pm.is_completed]
        pms_overdue = [pm for pm in self.pm_tasks if pm.is_overdue]
        active_techs = [t for t in self.technicians if t.status == TechStatus.ACTIVE]

        bullets: list[str] = []

        # Bullet 1: Open WO count
        p1_count = sum(1 for wo in open_wos if wo.priority == Priority.P1)
        p2_count = sum(1 for wo in open_wos if wo.priority == Priority.P2)
        bullet_wo = f"{len(open_wos)} open work orders"
        if p1_count or p2_count:
            parts = []
            if p1_count:
                parts.append(f"{p1_count} P1-Emergency")
            if p2_count:
                parts.append(f"{p2_count} P2-Urgent")
            bullet_wo += f" ({', '.join(parts)})"
        bullets.append(bullet_wo)

        # Bullet 2: PM status
        pm_text = f"{len(pms_due)} PMs due today"
        if pms_overdue:
            pm_text += f", {len(pms_overdue)} overdue"
        if self.daily_metrics:
            pm_text += f" | Compliance: {self.daily_metrics.pm_compliance_pct:.0f}%"
        bullets.append(pm_text)

        # Bullet 3: SLA risks
        if red_sla or yellow_sla:
            sla_text = "SLA Alert:"
            if red_sla:
                sla_text += f" {len(red_sla)} BREACHED"
            if yellow_sla:
                sla_text += f", {len(yellow_sla)} at risk (within {self.config.sla_warning_hours}h)"
            bullets.append(sla_text)
        else:
            bullets.append("SLA Status: All work orders within SLA targets")

        # Bullet 4: Staffing
        total_techs = len(self.technicians)
        bullets.append(
            f"Staffing: {len(active_techs)}/{total_techs} technicians active"
        )

        summary = {
            "bullets": bullets,
            "total_open_wos": len(open_wos),
            "p1_count": p1_count,
            "p2_count": p2_count,
            "sla_breached": len(red_sla),
            "sla_at_risk": len(yellow_sla),
            "pms_due": len(pms_due),
            "pms_overdue": len(pms_overdue),
            "active_techs": len(active_techs),
            "total_techs": total_techs,
        }
        logger.info("Executive summary: %s", json.dumps(summary, default=str))
        return summary

    def section_work_order_status(self) -> dict[str, Any]:
        """Open WOs grouped by priority with full detail table."""
        open_wos = [
            wo for wo in self.work_orders
            if wo.status not in (WOStatus.COMPLETED, WOStatus.CLOSED)
        ]

        grouped: dict[str, list[dict[str, Any]]] = {}
        for priority in Priority:
            priority_wos = [wo for wo in open_wos if wo.priority == priority]
            if priority_wos:
                # Sort by age descending (oldest first)
                priority_wos.sort(key=lambda w: w.age_hours, reverse=True)
                grouped[priority.value] = [wo.to_dict() for wo in priority_wos]

        return {
            "total_open": len(open_wos),
            "by_priority": grouped,
            "priority_counts": {
                p.value: sum(1 for wo in open_wos if wo.priority == p)
                for p in Priority
            },
        }

    def section_pm_schedule(self) -> dict[str, Any]:
        """Today's PM tasks: due, overdue, completion rate trending."""
        due_today = [
            pm for pm in self.pm_tasks if pm.due_date == self.report_date
        ]
        overdue = [pm for pm in self.pm_tasks if pm.is_overdue]
        completed = [pm for pm in self.pm_tasks if pm.is_completed]

        total_applicable = len(due_today) + len(overdue)
        completed_count = sum(1 for pm in due_today if pm.is_completed)
        completion_rate = (
            round((completed_count / len(due_today)) * 100, 1)
            if due_today else 0.0
        )

        # Trending from weekly metrics
        trend: list[float] = []
        for m in self.weekly_metrics[-7:]:
            trend.append(m.pm_compliance_pct)

        return {
            "due_today": [pm.to_dict() for pm in due_today],
            "overdue": [pm.to_dict() for pm in overdue],
            "due_today_count": len(due_today),
            "overdue_count": len(overdue),
            "completed_today_count": completed_count,
            "total_completed": len(completed),
            "completion_rate_today": completion_rate,
            "compliance_trend_7d": trend,
        }

    def section_technician_workload(self) -> dict[str, Any]:
        """Per-technician breakdown: tasks, hours, location, status."""
        tech_data = [t.to_dict() for t in self.technicians]

        total_tasks = sum(t.tasks_today for t in self.technicians)
        total_hours = sum(t.estimated_hours for t in self.technicians)
        total_completed = sum(t.completed_today for t in self.technicians)

        return {
            "technicians": tech_data,
            "total_tasks_assigned": total_tasks,
            "total_estimated_hours": round(total_hours, 1),
            "total_completed_today": total_completed,
            "status_breakdown": {
                s.value: sum(1 for t in self.technicians if t.status == s)
                for s in TechStatus
            },
        }

    def section_sla_watchlist(self) -> dict[str, Any]:
        """WOs approaching SLA breach within the warning window."""
        now = datetime.now()
        warning_cutoff = now + timedelta(hours=self.config.sla_warning_hours)

        watchlist: list[dict[str, Any]] = []
        for wo in self.work_orders:
            if wo.status in (WOStatus.COMPLETED, WOStatus.CLOSED):
                continue

            if wo.sla_status == SLAStatus.RED:
                entry = wo.to_dict()
                entry["breach_type"] = "BREACHED"
                hours_past = (now - wo.sla_deadline).total_seconds() / 3600
                entry["hours_past_sla"] = round(max(0, hours_past), 1)
                watchlist.append(entry)
            elif wo.sla_deadline <= warning_cutoff:
                entry = wo.to_dict()
                entry["breach_type"] = "AT_RISK"
                hours_remaining = (wo.sla_deadline - now).total_seconds() / 3600
                entry["hours_until_breach"] = round(max(0, hours_remaining), 1)
                watchlist.append(entry)

        # Sort: breached first, then by time remaining
        watchlist.sort(
            key=lambda w: (
                0 if w["breach_type"] == "BREACHED" else 1,
                w.get("hours_until_breach", 0),
            )
        )

        return {
            "watchlist": watchlist,
            "total_at_risk": len([w for w in watchlist if w["breach_type"] == "AT_RISK"]),
            "total_breached": len([w for w in watchlist if w["breach_type"] == "BREACHED"]),
            "warning_window_hours": self.config.sla_warning_hours,
        }

    def section_yesterdays_completions(self) -> dict[str, Any]:
        """What got done yesterday: counts, cycle time, callbacks."""
        yesterday = self.report_date - timedelta(days=1)
        completed_wos = [
            wo for wo in self.work_orders
            if wo.status in (WOStatus.COMPLETED, WOStatus.CLOSED)
        ]

        # Use daily metrics for yesterday if available
        yesterday_metrics = None
        for m in self.weekly_metrics:
            if m.date == yesterday:
                yesterday_metrics = m
                break

        if yesterday_metrics:
            return {
                "date": yesterday.isoformat(),
                "completed_count": yesterday_metrics.total_closed_today,
                "created_count": yesterday_metrics.total_created_today,
                "avg_cycle_hours": round(yesterday_metrics.avg_cycle_hours, 1),
                "callbacks": yesterday_metrics.callbacks,
                "sla_breaches": yesterday_metrics.sla_breaches,
            }

        return {
            "date": yesterday.isoformat(),
            "completed_count": len(completed_wos),
            "created_count": 0,
            "avg_cycle_hours": 0.0,
            "callbacks": 0,
            "sla_breaches": 0,
        }

    def section_vendor_activity(self) -> dict[str, Any]:
        """Open vendor WOs, pending arrivals, invoice status."""
        open_vendor = [
            v for v in self.vendor_orders if v.status != "Completed"
        ]
        pending_arrival = [
            v for v in open_vendor
            if v.scheduled_date and v.scheduled_date == self.report_date
        ]
        pending_invoices = [
            v for v in self.vendor_orders if v.invoice_status == "Pending"
        ]
        total_outstanding = sum(v.estimated_cost for v in pending_invoices)

        return {
            "open_vendor_wos": [v.to_dict() for v in open_vendor],
            "total_open": len(open_vendor),
            "arriving_today": [v.to_dict() for v in pending_arrival],
            "arriving_today_count": len(pending_arrival),
            "pending_invoices_count": len(pending_invoices),
            "total_outstanding_cost": round(total_outstanding, 2),
        }

    def section_weekly_trends(self) -> dict[str, Any]:
        """Mini sparkline data for last 7 days of metrics."""
        last_7 = self.weekly_metrics[-7:]

        wo_volume: list[int] = []
        cycle_times: list[float] = []
        pm_compliance: list[float] = []
        dates: list[str] = []

        for m in last_7:
            dates.append(m.date.isoformat())
            wo_volume.append(m.total_created_today)
            cycle_times.append(round(m.avg_cycle_hours, 1))
            pm_compliance.append(round(m.pm_compliance_pct, 1))

        # Compute simple trend direction
        def _trend_direction(values: list[float]) -> str:
            if len(values) < 2:
                return "flat"
            recent_avg = sum(values[-3:]) / min(3, len(values[-3:]))
            earlier_avg = sum(values[:3]) / min(3, len(values[:3]))
            diff = recent_avg - earlier_avg
            if diff > 0.5:
                return "up"
            if diff < -0.5:
                return "down"
            return "flat"

        return {
            "dates": dates,
            "wo_volume": wo_volume,
            "wo_volume_trend": _trend_direction([float(v) for v in wo_volume]),
            "avg_cycle_time": cycle_times,
            "cycle_time_trend": _trend_direction(cycle_times),
            "pm_compliance": pm_compliance,
            "pm_compliance_trend": _trend_direction(pm_compliance),
        }

    def section_action_items(self) -> dict[str, Any]:
        """Flagged items needing Tony or Juan's attention today."""
        items_by_owner: dict[str, list[dict[str, Any]]] = {}
        for item in self.action_items:
            items_by_owner.setdefault(item.owner, []).append(item.to_dict())

        high_urgency = [a for a in self.action_items if a.urgency == "high"]

        return {
            "items": [a.to_dict() for a in self.action_items],
            "total_count": len(self.action_items),
            "high_urgency_count": len(high_urgency),
            "by_owner": items_by_owner,
        }

    def section_adr_highlights(self) -> dict[str, Any]:
        """New ADRs from yesterday that management should review."""
        yesterday = self.report_date - timedelta(days=1)
        recent = [
            adr for adr in self.adr_highlights
            if adr.date_created >= yesterday
        ]

        return {
            "highlights": [a.to_dict() for a in recent],
            "total_new": len(recent),
        }

    # ------------------------------------------------------------------
    # Output formatters
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for API consumption."""
        if not self._sections:
            self.generate_full_report()
        return {
            "report_date": self.report_date.isoformat(),
            "campus_name": self.campus_name,
            "generated_at": (
                self._generated_at.isoformat() if self._generated_at else None
            ),
            "sections": self._sections,
        }

    def to_text(self) -> str:
        """Clean plain-text version for email, well-formatted with dividers."""
        if not self._sections:
            self.generate_full_report()

        lines: list[str] = []
        w = 78  # line width

        # Header
        lines.append("=" * w)
        lines.append(
            f"  DAILY MAINTENANCE REPORT -- {self.campus_name.upper()}"
        )
        lines.append(f"  {self.report_date.strftime('%A, %B %d, %Y')}")
        lines.append(f"  Fort Worth, TX | JLL Facility Management")
        lines.append("=" * w)
        lines.append("")

        # Executive Summary
        exec_sum = self._sections.get("executive_summary", {})
        lines.append("-" * w)
        lines.append("  EXECUTIVE SUMMARY")
        lines.append("-" * w)
        for bullet in exec_sum.get("bullets", []):
            lines.append(f"  * {bullet}")
        lines.append("")

        # Work Order Status
        wo_status = self._sections.get("work_order_status", {})
        lines.append("-" * w)
        lines.append(f"  OPEN WORK ORDERS ({wo_status.get('total_open', 0)})")
        lines.append("-" * w)

        header = (
            f"  {'WO#':<12} {'Description':<28} {'Bldg':<14} "
            f"{'Trade':<10} {'Tech':<14} {'Age(h)':<7} {'SLA':>5}"
        )
        lines.append(header)
        lines.append("  " + "-" * (w - 4))

        for priority_key, wos in wo_status.get("by_priority", {}).items():
            lines.append(f"  [{priority_key}] {Priority(priority_key).label}")
            for wo in wos:
                sla_icon = {"green": "[OK]", "yellow": "[!!]", "red": "[XX]"}.get(
                    wo["sla_status"], "[??]"
                )
                desc = wo["title"][:26] + ".." if len(wo["title"]) > 28 else wo["title"]
                bldg = wo["building"][:12] + ".." if len(wo["building"]) > 14 else wo["building"]
                tech = wo["assigned_tech"][:12] + ".." if len(wo["assigned_tech"]) > 14 else wo["assigned_tech"]
                lines.append(
                    f"  {wo['wo_number']:<12} {desc:<28} {bldg:<14} "
                    f"{wo['trade']:<10} {tech:<14} {wo['age_hours']:<7.1f} {sla_icon:>5}"
                )
            lines.append("")

        # PM Schedule
        pm_sched = self._sections.get("pm_schedule", {})
        lines.append("-" * w)
        lines.append(
            f"  PM SCHEDULE -- Due Today: {pm_sched.get('due_today_count', 0)} | "
            f"Overdue: {pm_sched.get('overdue_count', 0)} | "
            f"Completion: {pm_sched.get('completion_rate_today', 0):.0f}%"
        )
        lines.append("-" * w)
        for pm in pm_sched.get("due_today", []):
            status_mark = "[DONE]" if pm["is_completed"] else "[    ]"
            overdue_mark = " (OVERDUE)" if pm["is_overdue"] else ""
            lines.append(
                f"  {status_mark} {pm['pm_id']:<10} {pm['title'][:40]:<40} "
                f"{pm['building'][:14]:<14}{overdue_mark}"
            )
        if pm_sched.get("overdue", []):
            lines.append("  OVERDUE:")
            for pm in pm_sched["overdue"]:
                lines.append(
                    f"  [LATE] {pm['pm_id']:<10} {pm['title'][:40]:<40} "
                    f"{pm['building'][:14]}"
                )
        lines.append("")

        # Technician Workload
        tech_wl = self._sections.get("technician_workload", {})
        lines.append("-" * w)
        lines.append("  TECHNICIAN WORKLOAD")
        lines.append("-" * w)
        lines.append(
            f"  {'Name':<20} {'Trade':<12} {'Tasks':<7} {'Est.Hrs':<9} "
            f"{'Done':<6} {'Status':<12} {'Current Task'}"
        )
        lines.append("  " + "-" * (w - 4))
        for tech in tech_wl.get("technicians", []):
            task_str = tech.get("current_task", "")[:20]
            lines.append(
                f"  {tech['name']:<20} {tech['trade']:<12} "
                f"{tech['tasks_today']:<7} {tech['estimated_hours']:<9.1f} "
                f"{tech['completed_today']:<6} {tech['status']:<12} {task_str}"
            )
        lines.append("")

        # SLA Watchlist
        sla_wl = self._sections.get("sla_watchlist", {})
        watchlist = sla_wl.get("watchlist", [])
        if watchlist:
            lines.append("-" * w)
            lines.append(
                f"  *** SLA WATCHLIST *** "
                f"({sla_wl.get('total_breached', 0)} breached, "
                f"{sla_wl.get('total_at_risk', 0)} at risk)"
            )
            lines.append("-" * w)
            for entry in watchlist:
                if entry["breach_type"] == "BREACHED":
                    flag = f"BREACHED (+{entry.get('hours_past_sla', 0):.1f}h)"
                else:
                    flag = f"AT RISK ({entry.get('hours_until_breach', 0):.1f}h remaining)"
                lines.append(
                    f"  {entry['wo_number']:<12} {entry['title'][:30]:<30} "
                    f"{entry['priority']:<4} {flag}"
                )
            lines.append("")

        # Yesterday's Completions
        yest = self._sections.get("yesterdays_completions", {})
        lines.append("-" * w)
        lines.append(
            f"  YESTERDAY'S COMPLETIONS ({yest.get('date', '')})"
        )
        lines.append("-" * w)
        lines.append(f"  Completed: {yest.get('completed_count', 0)}")
        lines.append(f"  Created:   {yest.get('created_count', 0)}")
        lines.append(f"  Avg Cycle: {yest.get('avg_cycle_hours', 0):.1f} hours")
        lines.append(f"  Callbacks: {yest.get('callbacks', 0)}")
        lines.append(f"  SLA Breaches: {yest.get('sla_breaches', 0)}")
        lines.append("")

        # Vendor Activity
        vendor = self._sections.get("vendor_activity", {})
        lines.append("-" * w)
        lines.append(
            f"  VENDOR ACTIVITY -- Open: {vendor.get('total_open', 0)} | "
            f"Arriving Today: {vendor.get('arriving_today_count', 0)}"
        )
        lines.append("-" * w)
        for v in vendor.get("open_vendor_wos", []):
            lines.append(
                f"  {v['wo_number']:<12} {v['vendor_name']:<20} "
                f"{v['description'][:30]:<30} {v['status']}"
            )
        if vendor.get("pending_invoices_count", 0):
            lines.append(
                f"  Pending Invoices: {vendor['pending_invoices_count']} "
                f"(${vendor.get('total_outstanding_cost', 0):,.2f} outstanding)"
            )
        lines.append("")

        # Weekly Trends
        trends = self._sections.get("weekly_trends", {})
        lines.append("-" * w)
        lines.append("  7-DAY TRENDS")
        lines.append("-" * w)
        if trends.get("dates"):
            lines.append(
                f"  WO Volume (trend: {trends.get('wo_volume_trend', 'flat')}):  "
                + " -> ".join(str(v) for v in trends.get("wo_volume", []))
            )
            lines.append(
                f"  Cycle Time (trend: {trends.get('cycle_time_trend', 'flat')}): "
                + " -> ".join(f"{v:.1f}h" for v in trends.get("avg_cycle_time", []))
            )
            lines.append(
                f"  PM Compliance (trend: {trends.get('pm_compliance_trend', 'flat')}): "
                + " -> ".join(f"{v:.0f}%" for v in trends.get("pm_compliance", []))
            )
        else:
            lines.append("  No trend data available for this period.")
        lines.append("")

        # Action Items
        actions = self._sections.get("action_items", {})
        if actions.get("total_count", 0):
            lines.append("-" * w)
            lines.append(
                f"  ACTION ITEMS REQUIRING ATTENTION "
                f"({actions['total_count']} items, "
                f"{actions.get('high_urgency_count', 0)} high urgency)"
            )
            lines.append("-" * w)
            for owner, items in actions.get("by_owner", {}).items():
                lines.append(f"  {owner}:")
                for item in items:
                    urgency_flag = (
                        ">>>" if item["urgency"] == "high" else
                        " >>" if item["urgency"] == "medium" else
                        "  >"
                    )
                    wo_ref = f" [ref: {item['related_wo']}]" if item.get("related_wo") else ""
                    lines.append(
                        f"    {urgency_flag} {item['description']}"
                        f"{wo_ref} -- {item['due_by']}"
                    )
            lines.append("")

        # ADR Highlights
        adrs = self._sections.get("adr_highlights", {})
        if adrs.get("total_new", 0):
            lines.append("-" * w)
            lines.append(
                f"  NEW ADRs FOR REVIEW ({adrs['total_new']})"
            )
            lines.append("-" * w)
            for adr in adrs.get("highlights", []):
                lines.append(
                    f"  [{adr['adr_id']}] {adr['title']} ({adr['status']})"
                )
                lines.append(f"    Impact: {adr['impact_area']}")
                lines.append(f"    {adr['summary']}")
            lines.append("")

        # Footer
        lines.append("=" * w)
        generated_str = (
            self._generated_at.strftime("%Y-%m-%d %H:%M:%S %Z")
            if self._generated_at else "N/A"
        )
        lines.append(f"  Generated: {generated_str} CT")
        lines.append(f"  Powered by CoreSkills MAO | JLL Technologies")
        lines.append("=" * w)

        return "\n".join(lines)

    def to_html(self) -> str:
        """Professional HTML email template with JLL orange/dark branding."""
        if not self._sections:
            self.generate_full_report()

        exec_sum = self._sections.get("executive_summary", {})
        wo_status = self._sections.get("work_order_status", {})
        pm_sched = self._sections.get("pm_schedule", {})
        tech_wl = self._sections.get("technician_workload", {})
        sla_wl = self._sections.get("sla_watchlist", {})
        yest = self._sections.get("yesterdays_completions", {})
        vendor = self._sections.get("vendor_activity", {})
        trends = self._sections.get("weekly_trends", {})
        actions = self._sections.get("action_items", {})
        adrs = self._sections.get("adr_highlights", {})

        # Priority badge colors
        priority_colors = {
            "P1": "#dc3545",  # red
            "P2": "#fd7e14",  # orange
            "P3": "#ffc107",  # yellow
            "P4": "#28a745",  # green
            "P5": "#6c757d",  # gray
        }

        # SLA traffic light unicode
        sla_icons = {
            "green": "&#x1F7E2;",   # green circle
            "yellow": "&#x1F7E1;",  # yellow circle
            "red": "&#x1F534;",     # red circle
        }

        generated_str = (
            self._generated_at.strftime("%B %d, %Y at %I:%M %p CT")
            if self._generated_at else "N/A"
        )

        # --- Build HTML ---
        html_parts: list[str] = []

        html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily Maintenance Report - {self.campus_name} - {self.report_date.strftime('%B %d, %Y')}</title>
<style>
  /* Reset */
  body, table, td, th {{ margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: #2d2d2d;
    background-color: #f4f4f4;
  }}
  .container {{
    max-width: 900px;
    margin: 0 auto;
    background: #ffffff;
  }}
  /* Header - JLL Dark/Orange */
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #ffffff;
    padding: 24px 32px;
  }}
  .header h1 {{
    font-size: 22px;
    margin: 0 0 4px 0;
    font-weight: 700;
    letter-spacing: 0.5px;
  }}
  .header .subtitle {{
    font-size: 14px;
    color: #e8730e;
    font-weight: 600;
  }}
  .header .date-line {{
    font-size: 13px;
    color: #a0aec0;
    margin-top: 6px;
  }}
  /* JLL orange accent bar */
  .accent-bar {{
    height: 4px;
    background: linear-gradient(90deg, #e8730e 0%, #ff9a3c 100%);
  }}
  /* Sections */
  .section {{
    padding: 20px 32px;
    border-bottom: 1px solid #e2e8f0;
  }}
  .section:last-of-type {{
    border-bottom: none;
  }}
  .section-title {{
    font-size: 16px;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e8730e;
    display: inline-block;
  }}
  /* Executive summary bullets */
  .exec-bullets {{
    list-style: none;
    padding: 0;
    margin: 0;
  }}
  .exec-bullets li {{
    padding: 8px 12px;
    margin-bottom: 6px;
    background: #f8f9fa;
    border-left: 4px solid #e8730e;
    border-radius: 0 4px 4px 0;
    font-size: 14px;
  }}
  /* Tables */
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 8px;
  }}
  .data-table th {{
    background: #1a1a2e;
    color: #ffffff;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .data-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #e2e8f0;
  }}
  .data-table tr:nth-child(even) td {{
    background: #f8f9fa;
  }}
  .data-table tr:hover td {{
    background: #fff3e6;
  }}
  /* Priority badges */
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .badge-p1 {{ background: #dc3545; }}
  .badge-p2 {{ background: #fd7e14; }}
  .badge-p3 {{ background: #ffc107; color: #1a1a2e; }}
  .badge-p4 {{ background: #28a745; }}
  .badge-p5 {{ background: #6c757d; }}
  /* SLA indicators */
  .sla-indicator {{
    font-size: 16px;
  }}
  /* Watchlist alert */
  .watchlist-alert {{
    background: #fff5f5;
    border: 1px solid #feb2b2;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
  }}
  .watchlist-alert .alert-title {{
    color: #dc3545;
    font-weight: 700;
    font-size: 14px;
  }}
  /* KPI cards */
  .kpi-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 8px;
  }}
  .kpi-card {{
    flex: 1;
    min-width: 120px;
    background: #f8f9fa;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
    border-top: 3px solid #e8730e;
  }}
  .kpi-card .kpi-value {{
    font-size: 24px;
    font-weight: 700;
    color: #1a1a2e;
  }}
  .kpi-card .kpi-label {{
    font-size: 11px;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  /* Trend arrows */
  .trend-up {{ color: #dc3545; }}
  .trend-down {{ color: #28a745; }}
  .trend-flat {{ color: #718096; }}
  .trend-up-good {{ color: #28a745; }}
  .trend-down-bad {{ color: #dc3545; }}
  /* Action items */
  .action-high {{
    background: #fff5f5;
    border-left: 4px solid #dc3545;
    padding: 8px 12px;
    margin-bottom: 6px;
    border-radius: 0 4px 4px 0;
  }}
  .action-medium {{
    background: #fffbeb;
    border-left: 4px solid #ffc107;
    padding: 8px 12px;
    margin-bottom: 6px;
    border-radius: 0 4px 4px 0;
  }}
  .action-low {{
    background: #f0fff4;
    border-left: 4px solid #28a745;
    padding: 8px 12px;
    margin-bottom: 6px;
    border-radius: 0 4px 4px 0;
  }}
  /* Footer */
  .footer {{
    background: #1a1a2e;
    color: #a0aec0;
    padding: 16px 32px;
    font-size: 12px;
    text-align: center;
  }}
  .footer a {{
    color: #e8730e;
    text-decoration: none;
  }}
  .footer .powered-by {{
    margin-top: 8px;
    font-size: 11px;
    color: #718096;
  }}
  /* PM status */
  .pm-done {{
    color: #28a745;
    font-weight: 600;
  }}
  .pm-pending {{
    color: #ffc107;
    font-weight: 600;
  }}
  .pm-overdue {{
    color: #dc3545;
    font-weight: 600;
  }}
  /* Priority group header */
  .priority-group {{
    background: #edf2f7;
    padding: 6px 12px;
    font-weight: 700;
    font-size: 13px;
    color: #1a1a2e;
  }}
  /* Responsive */
  @media (max-width: 600px) {{
    .header {{ padding: 16px; }}
    .section {{ padding: 16px; }}
    .header h1 {{ font-size: 18px; }}
    .data-table {{ font-size: 11px; }}
    .data-table th, .data-table td {{ padding: 6px 8px; }}
    .kpi-row {{ flex-direction: column; }}
    .kpi-card {{ min-width: 100%; }}
  }}
</style>
</head>
<body>
<div class="container">
""")

        # Header
        html_parts.append(f"""
<div class="header">
  <h1>Daily Maintenance Report</h1>
  <div class="subtitle">{self.campus_name} &mdash; JLL Facility Management</div>
  <div class="date-line">{self.report_date.strftime('%A, %B %d, %Y')} &bull; {self.CAMPUS_ADDRESS}</div>
</div>
<div class="accent-bar"></div>
""")

        # Executive Summary
        html_parts.append('<div class="section">')
        html_parts.append('<h2 class="section-title">Executive Summary</h2>')
        html_parts.append('<ul class="exec-bullets">')
        for bullet in exec_sum.get("bullets", []):
            html_parts.append(f"  <li>{self._html_escape(bullet)}</li>")
        html_parts.append("</ul>")

        # KPI cards
        html_parts.append('<div class="kpi-row">')
        kpi_data = [
            (str(exec_sum.get("total_open_wos", 0)), "Open WOs"),
            (str(exec_sum.get("pms_due", 0)), "PMs Due"),
            (str(exec_sum.get("sla_breached", 0)), "SLA Breached"),
            (str(exec_sum.get("active_techs", 0)) + "/" + str(exec_sum.get("total_techs", 0)), "Techs Active"),
        ]
        for val, label in kpi_data:
            html_parts.append(f"""
  <div class="kpi-card">
    <div class="kpi-value">{val}</div>
    <div class="kpi-label">{label}</div>
  </div>""")
        html_parts.append("</div></div>")

        # Work Order Status
        html_parts.append('<div class="section">')
        html_parts.append(
            f'<h2 class="section-title">Open Work Orders ({wo_status.get("total_open", 0)})</h2>'
        )
        html_parts.append("""
<table class="data-table">
<thead>
  <tr>
    <th>WO#</th>
    <th>Description</th>
    <th>Building</th>
    <th>Trade</th>
    <th>Assigned Tech</th>
    <th>Age (hrs)</th>
    <th>SLA</th>
  </tr>
</thead>
<tbody>""")
        for pkey, wos in wo_status.get("by_priority", {}).items():
            badge_class = f"badge-{pkey.lower()}"
            html_parts.append(
                f'<tr><td colspan="7" class="priority-group">'
                f'<span class="badge {badge_class}">{pkey}</span> '
                f'{Priority(pkey).label} ({len(wos)})</td></tr>'
            )
            for wo in wos:
                sla_icon = sla_icons.get(wo["sla_status"], "")
                html_parts.append(f"""  <tr>
    <td><strong>{self._html_escape(wo['wo_number'])}</strong></td>
    <td>{self._html_escape(wo['title'])}</td>
    <td>{self._html_escape(wo['building'])}</td>
    <td>{self._html_escape(wo['trade'])}</td>
    <td>{self._html_escape(wo['assigned_tech'])}</td>
    <td>{wo['age_hours']:.1f}</td>
    <td class="sla-indicator">{sla_icon}</td>
  </tr>""")
        html_parts.append("</tbody></table></div>")

        # PM Schedule
        html_parts.append('<div class="section">')
        html_parts.append('<h2 class="section-title">PM Schedule</h2>')
        html_parts.append('<div class="kpi-row">')
        pm_kpis = [
            (str(pm_sched.get("due_today_count", 0)), "Due Today"),
            (str(pm_sched.get("overdue_count", 0)), "Overdue"),
            (f"{pm_sched.get('completion_rate_today', 0):.0f}%", "Completion Rate"),
        ]
        for val, label in pm_kpis:
            html_parts.append(f"""
  <div class="kpi-card">
    <div class="kpi-value">{val}</div>
    <div class="kpi-label">{label}</div>
  </div>""")
        html_parts.append("</div>")

        if pm_sched.get("due_today") or pm_sched.get("overdue"):
            html_parts.append("""
<table class="data-table" style="margin-top:12px">
<thead><tr>
  <th>PM ID</th><th>Task</th><th>Building</th><th>Trade</th>
  <th>Technician</th><th>Status</th>
</tr></thead><tbody>""")
            for pm in pm_sched.get("due_today", []):
                if pm["is_completed"]:
                    status_html = '<span class="pm-done">Completed</span>'
                elif pm["is_overdue"]:
                    status_html = '<span class="pm-overdue">OVERDUE</span>'
                else:
                    status_html = '<span class="pm-pending">Pending</span>'
                html_parts.append(f"""<tr>
    <td>{self._html_escape(pm['pm_id'])}</td>
    <td>{self._html_escape(pm['title'])}</td>
    <td>{self._html_escape(pm['building'])}</td>
    <td>{self._html_escape(pm['trade'])}</td>
    <td>{self._html_escape(pm['assigned_tech'])}</td>
    <td>{status_html}</td>
  </tr>""")
            for pm in pm_sched.get("overdue", []):
                html_parts.append(f"""<tr>
    <td>{self._html_escape(pm['pm_id'])}</td>
    <td>{self._html_escape(pm['title'])}</td>
    <td>{self._html_escape(pm['building'])}</td>
    <td>{self._html_escape(pm['trade'])}</td>
    <td>{self._html_escape(pm['assigned_tech'])}</td>
    <td><span class="pm-overdue">OVERDUE</span></td>
  </tr>""")
            html_parts.append("</tbody></table>")
        html_parts.append("</div>")

        # Technician Workload
        html_parts.append('<div class="section">')
        html_parts.append('<h2 class="section-title">Technician Workload</h2>')
        html_parts.append("""
<table class="data-table">
<thead><tr>
  <th>Name</th><th>Trade</th><th>Tasks</th><th>Est. Hours</th>
  <th>Completed</th><th>Status</th><th>Current Task</th>
</tr></thead><tbody>""")
        for tech in tech_wl.get("technicians", []):
            status_color = {
                "active": "#28a745",
                "break": "#ffc107",
                "off-site": "#17a2b8",
                "not-started": "#6c757d",
            }.get(tech["status"], "#6c757d")
            html_parts.append(f"""<tr>
    <td><strong>{self._html_escape(tech['name'])}</strong></td>
    <td>{self._html_escape(tech['trade'])}</td>
    <td>{tech['tasks_today']}</td>
    <td>{tech['estimated_hours']:.1f}</td>
    <td>{tech['completed_today']}</td>
    <td><span style="color:{status_color};font-weight:600">{tech['status'].upper()}</span></td>
    <td>{self._html_escape(tech.get('current_task', ''))}</td>
  </tr>""")
        html_parts.append("</tbody></table></div>")

        # SLA Watchlist
        watchlist = sla_wl.get("watchlist", [])
        if watchlist:
            html_parts.append('<div class="section">')
            html_parts.append('<h2 class="section-title">SLA Watchlist</h2>')
            html_parts.append(f"""
<div class="watchlist-alert">
  <div class="alert-title">&#x26A0; {sla_wl.get('total_breached', 0)} Breached &bull;
  {sla_wl.get('total_at_risk', 0)} At Risk (within {sla_wl.get('warning_window_hours', 4)}h)</div>
</div>""")
            html_parts.append("""
<table class="data-table">
<thead><tr>
  <th>WO#</th><th>Description</th><th>Priority</th>
  <th>Status</th><th>Detail</th>
</tr></thead><tbody>""")
            for entry in watchlist:
                if entry["breach_type"] == "BREACHED":
                    status_html = (
                        f'<span style="color:#dc3545;font-weight:700">'
                        f'BREACHED (+{entry.get("hours_past_sla", 0):.1f}h)</span>'
                    )
                else:
                    status_html = (
                        f'<span style="color:#fd7e14;font-weight:700">'
                        f'AT RISK ({entry.get("hours_until_breach", 0):.1f}h left)</span>'
                    )
                badge_class = f"badge-{entry['priority'].lower()}"
                html_parts.append(f"""<tr>
    <td><strong>{self._html_escape(entry['wo_number'])}</strong></td>
    <td>{self._html_escape(entry['title'])}</td>
    <td><span class="badge {badge_class}">{entry['priority']}</span></td>
    <td>{status_html}</td>
    <td>{self._html_escape(entry.get('building', ''))}</td>
  </tr>""")
            html_parts.append("</tbody></table></div>")

        # Yesterday's Completions
        html_parts.append('<div class="section">')
        html_parts.append(
            f'<h2 class="section-title">Yesterday\'s Completions ({yest.get("date", "")})</h2>'
        )
        html_parts.append('<div class="kpi-row">')
        yest_kpis = [
            (str(yest.get("completed_count", 0)), "Completed"),
            (str(yest.get("created_count", 0)), "New WOs Created"),
            (f"{yest.get('avg_cycle_hours', 0):.1f}h", "Avg Cycle Time"),
            (str(yest.get("callbacks", 0)), "Callbacks"),
            (str(yest.get("sla_breaches", 0)), "SLA Breaches"),
        ]
        for val, label in yest_kpis:
            html_parts.append(f"""
  <div class="kpi-card">
    <div class="kpi-value">{val}</div>
    <div class="kpi-label">{label}</div>
  </div>""")
        html_parts.append("</div></div>")

        # Vendor Activity
        html_parts.append('<div class="section">')
        html_parts.append('<h2 class="section-title">Vendor Activity</h2>')
        if vendor.get("open_vendor_wos"):
            html_parts.append("""
<table class="data-table">
<thead><tr>
  <th>WO#</th><th>Vendor</th><th>Description</th><th>Building</th>
  <th>Status</th><th>Scheduled</th><th>Invoice</th>
</tr></thead><tbody>""")
            for v in vendor.get("open_vendor_wos", []):
                sched = v.get("scheduled_date", "TBD") or "TBD"
                html_parts.append(f"""<tr>
    <td><strong>{self._html_escape(v['wo_number'])}</strong></td>
    <td>{self._html_escape(v['vendor_name'])}</td>
    <td>{self._html_escape(v['description'])}</td>
    <td>{self._html_escape(v['building'])}</td>
    <td>{self._html_escape(v['status'])}</td>
    <td>{sched}</td>
    <td>{self._html_escape(v['invoice_status'])}</td>
  </tr>""")
            html_parts.append("</tbody></table>")
        if vendor.get("pending_invoices_count"):
            html_parts.append(
                f'<p style="margin-top:8px;color:#718096;">Pending Invoices: '
                f'{vendor["pending_invoices_count"]} '
                f'(${vendor.get("total_outstanding_cost", 0):,.2f} outstanding)</p>'
            )
        html_parts.append("</div>")

        # Weekly Trends
        html_parts.append('<div class="section">')
        html_parts.append('<h2 class="section-title">7-Day Trends</h2>')
        if trends.get("dates"):
            trend_arrow = {"up": "&#x2191;", "down": "&#x2193;", "flat": "&#x2194;"}

            html_parts.append('<table class="data-table"><thead><tr><th>Metric</th>')
            for d in trends["dates"]:
                html_parts.append(f"<th>{d[-5:]}</th>")
            html_parts.append("<th>Trend</th></tr></thead><tbody>")

            # WO Volume row
            wo_trend_dir = trends.get("wo_volume_trend", "flat")
            arrow = trend_arrow.get(wo_trend_dir, "")
            html_parts.append("<tr><td><strong>WO Volume</strong></td>")
            for v in trends.get("wo_volume", []):
                html_parts.append(f"<td>{v}</td>")
            html_parts.append(f'<td class="trend-{wo_trend_dir}">{arrow} {wo_trend_dir}</td></tr>')

            # Cycle time row
            ct_trend_dir = trends.get("cycle_time_trend", "flat")
            ct_class = "trend-down" if ct_trend_dir == "down" else "trend-up" if ct_trend_dir == "up" else "trend-flat"
            arrow = trend_arrow.get(ct_trend_dir, "")
            html_parts.append("<tr><td><strong>Avg Cycle (hrs)</strong></td>")
            for v in trends.get("avg_cycle_time", []):
                html_parts.append(f"<td>{v:.1f}</td>")
            html_parts.append(f'<td class="{ct_class}">{arrow} {ct_trend_dir}</td></tr>')

            # PM compliance row
            pm_trend_dir = trends.get("pm_compliance_trend", "flat")
            pm_class = "trend-up-good" if pm_trend_dir == "up" else "trend-down-bad" if pm_trend_dir == "down" else "trend-flat"
            arrow = trend_arrow.get(pm_trend_dir, "")
            html_parts.append("<tr><td><strong>PM Compliance (%)</strong></td>")
            for v in trends.get("pm_compliance", []):
                html_parts.append(f"<td>{v:.0f}%</td>")
            html_parts.append(f'<td class="{pm_class}">{arrow} {pm_trend_dir}</td></tr>')

            html_parts.append("</tbody></table>")
        else:
            html_parts.append("<p>No trend data available for this period.</p>")
        html_parts.append("</div>")

        # Action Items
        if actions.get("total_count", 0):
            html_parts.append('<div class="section">')
            html_parts.append(
                f'<h2 class="section-title">Action Items ({actions["total_count"]})</h2>'
            )
            for owner, items in actions.get("by_owner", {}).items():
                html_parts.append(
                    f'<h3 style="margin:12px 0 8px 0;color:#1a1a2e;">{self._html_escape(owner)}</h3>'
                )
                for item in items:
                    css_class = f"action-{item['urgency']}"
                    wo_ref = (
                        f' <span style="color:#718096">[{self._html_escape(item["related_wo"])}]</span>'
                        if item.get("related_wo") else ""
                    )
                    urgency_badge = (
                        f'<span class="badge badge-p1">{item["urgency"].upper()}</span>'
                        if item["urgency"] == "high" else
                        f'<span class="badge badge-p3">{item["urgency"].upper()}</span>'
                        if item["urgency"] == "medium" else
                        f'<span class="badge badge-p4">{item["urgency"].upper()}</span>'
                    )
                    html_parts.append(
                        f'<div class="{css_class}">'
                        f'{urgency_badge} {self._html_escape(item["description"])}'
                        f'{wo_ref} &mdash; <em>{self._html_escape(item["due_by"])}</em></div>'
                    )
            html_parts.append("</div>")

        # ADR Highlights
        if adrs.get("total_new", 0):
            html_parts.append('<div class="section">')
            html_parts.append(
                f'<h2 class="section-title">New ADRs for Review ({adrs["total_new"]})</h2>'
            )
            for adr in adrs.get("highlights", []):
                html_parts.append(f"""
<div style="background:#f8f9fa;border-radius:6px;padding:12px 16px;margin-bottom:8px;">
  <div style="font-weight:700;color:#1a1a2e;">
    [{self._html_escape(adr['adr_id'])}] {self._html_escape(adr['title'])}
    <span class="badge badge-p3">{self._html_escape(adr['status'])}</span>
  </div>
  <div style="font-size:12px;color:#718096;margin-top:4px;">
    Impact: {self._html_escape(adr['impact_area'])}
  </div>
  <div style="margin-top:4px;">{self._html_escape(adr['summary'])}</div>
</div>""")
            html_parts.append("</div>")

        # Footer
        html_parts.append(f"""
<div class="footer">
  <div>Generated: {generated_str}</div>
  <div>Report for: {self._html_escape(', '.join(self.config.recipients))}</div>
  <div class="powered-by">Powered by <strong>CoreSkills MAO</strong> &bull; JLL Technologies</div>
</div>
</div>
</body>
</html>""")

        return "\n".join(html_parts)

    def to_markdown(self) -> str:
        """Markdown version suitable for Microsoft Teams / Slack posting."""
        if not self._sections:
            self.generate_full_report()

        parts: list[str] = []

        exec_sum = self._sections.get("executive_summary", {})
        wo_status = self._sections.get("work_order_status", {})
        pm_sched = self._sections.get("pm_schedule", {})
        tech_wl = self._sections.get("technician_workload", {})
        sla_wl = self._sections.get("sla_watchlist", {})
        yest = self._sections.get("yesterdays_completions", {})
        vendor = self._sections.get("vendor_activity", {})
        trends = self._sections.get("weekly_trends", {})
        actions = self._sections.get("action_items", {})
        adrs = self._sections.get("adr_highlights", {})

        # Header
        parts.append(
            f"# Daily Maintenance Report - {self.campus_name}"
        )
        parts.append(
            f"**{self.report_date.strftime('%A, %B %d, %Y')}** | "
            f"Fort Worth, TX | JLL Facility Management"
        )
        parts.append("")

        # Executive Summary
        parts.append("## Executive Summary")
        for bullet in exec_sum.get("bullets", []):
            parts.append(f"- {bullet}")
        parts.append("")

        # Work Order Status
        parts.append(f"## Open Work Orders ({wo_status.get('total_open', 0)})")
        parts.append("")
        parts.append(
            "| WO# | Description | Building | Trade | Tech | Age(h) | SLA |"
        )
        parts.append(
            "|------|-------------|----------|-------|------|--------|-----|"
        )
        for pkey, wos in wo_status.get("by_priority", {}).items():
            for wo in wos:
                sla_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(
                    wo["sla_status"], "⚪"
                )
                parts.append(
                    f"| {wo['wo_number']} | {wo['title'][:35]} | "
                    f"{wo['building'][:15]} | {wo['trade']} | "
                    f"{wo['assigned_tech'][:15]} | {wo['age_hours']:.1f} | "
                    f"{sla_emoji} |"
                )
        parts.append("")

        # PM Schedule
        parts.append("## PM Schedule")
        parts.append(
            f"- Due Today: **{pm_sched.get('due_today_count', 0)}** | "
            f"Overdue: **{pm_sched.get('overdue_count', 0)}** | "
            f"Completion: **{pm_sched.get('completion_rate_today', 0):.0f}%**"
        )
        parts.append("")

        # Technician Workload
        parts.append("## Technician Workload")
        parts.append("| Name | Trade | Tasks | Est.Hrs | Done | Status |")
        parts.append("|------|-------|-------|---------|------|--------|")
        for tech in tech_wl.get("technicians", []):
            parts.append(
                f"| {tech['name']} | {tech['trade']} | {tech['tasks_today']} | "
                f"{tech['estimated_hours']:.1f} | {tech['completed_today']} | "
                f"{tech['status']} |"
            )
        parts.append("")

        # SLA Watchlist
        watchlist = sla_wl.get("watchlist", [])
        if watchlist:
            parts.append("## SLA Watchlist")
            for entry in watchlist:
                if entry["breach_type"] == "BREACHED":
                    parts.append(
                        f"- **BREACHED** {entry['wo_number']}: "
                        f"{entry['title']} (+{entry.get('hours_past_sla', 0):.1f}h past SLA)"
                    )
                else:
                    parts.append(
                        f"- **AT RISK** {entry['wo_number']}: "
                        f"{entry['title']} ({entry.get('hours_until_breach', 0):.1f}h remaining)"
                    )
            parts.append("")

        # Yesterday
        parts.append(f"## Yesterday's Completions ({yest.get('date', '')})")
        parts.append(f"- Completed: **{yest.get('completed_count', 0)}**")
        parts.append(f"- Avg Cycle: **{yest.get('avg_cycle_hours', 0):.1f}h**")
        parts.append(f"- Callbacks: **{yest.get('callbacks', 0)}**")
        parts.append("")

        # Vendor
        parts.append(f"## Vendor Activity ({vendor.get('total_open', 0)} open)")
        for v in vendor.get("open_vendor_wos", []):
            parts.append(
                f"- {v['wo_number']}: {v['vendor_name']} - {v['description']} [{v['status']}]"
            )
        parts.append("")

        # Trends
        if trends.get("dates"):
            parts.append("## 7-Day Trends")
            parts.append(
                f"- WO Volume: {' > '.join(str(v) for v in trends.get('wo_volume', []))} "
                f"({trends.get('wo_volume_trend', 'flat')})"
            )
            parts.append(
                f"- Cycle Time: {' > '.join(f'{v:.1f}h' for v in trends.get('avg_cycle_time', []))} "
                f"({trends.get('cycle_time_trend', 'flat')})"
            )
            parts.append(
                f"- PM Compliance: {' > '.join(f'{v:.0f}%' for v in trends.get('pm_compliance', []))} "
                f"({trends.get('pm_compliance_trend', 'flat')})"
            )
            parts.append("")

        # Action Items
        if actions.get("total_count"):
            parts.append(f"## Action Items ({actions['total_count']})")
            for owner, items in actions.get("by_owner", {}).items():
                parts.append(f"**{owner}:**")
                for item in items:
                    urgency_marker = {
                        "high": "🔴", "medium": "🟡", "low": "🟢"
                    }.get(item["urgency"], "")
                    wo_ref = f" [{item['related_wo']}]" if item.get("related_wo") else ""
                    parts.append(
                        f"  - {urgency_marker} {item['description']}{wo_ref} -- {item['due_by']}"
                    )
            parts.append("")

        # ADRs
        if adrs.get("total_new"):
            parts.append(f"## New ADRs ({adrs['total_new']})")
            for adr in adrs.get("highlights", []):
                parts.append(
                    f"- **[{adr['adr_id']}]** {adr['title']} ({adr['status']}) - {adr['summary']}"
                )
            parts.append("")

        # Footer
        generated_str = (
            self._generated_at.strftime("%Y-%m-%d %H:%M:%S CT")
            if self._generated_at else "N/A"
        )
        parts.append("---")
        parts.append(f"*Generated: {generated_str} | Powered by CoreSkills MAO*")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Delivery methods
    # ------------------------------------------------------------------

    def send_email(
        self,
        smtp_host: str = "smtp.office365.com",
        smtp_port: int = 587,
        sender: str = "mao-reports@jll.com",
        username: str | None = None,
        password: str | None = None,
    ) -> bool:
        """Send the report via SMTP email.

        Parameters
        ----------
        smtp_host : str
            SMTP server hostname.
        smtp_port : int
            SMTP server port (587 for STARTTLS).
        sender : str
            From address for the email.
        username : str or None
            SMTP authentication username. Falls back to ``sender`` if None.
        password : str or None
            SMTP authentication password. Required for actual delivery.

        Returns
        -------
        bool
            True if email was sent (or simulated) successfully.

        Notes
        -----
        This is a placeholder implementation. In production, configure with
        real SMTP credentials or integrate with SendGrid / Microsoft Graph API.
        """
        logger.info("send_email | preparing report for %d recipients", len(self.config.recipients))

        subject = (
            f"Daily Maintenance Report - {self.campus_name} - "
            f"{self.report_date.strftime('%m/%d/%Y')}"
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(self.config.recipients)

        # Attach plain text and HTML versions
        text_part = MIMEText(self.to_text(), "plain", "utf-8")
        html_part = MIMEText(self.to_html(), "html", "utf-8")
        msg.attach(text_part)
        msg.attach(html_part)

        if password is None:
            logger.warning(
                "send_email | No SMTP password configured. "
                "Email delivery simulated (not sent)."
            )
            logger.info(
                "send_email | Would send to: %s | Subject: %s",
                ", ".join(self.config.recipients),
                subject,
            )
            return True

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(username or sender, password)
                server.send_message(msg)
            logger.info("send_email | Sent successfully to %s", ", ".join(self.config.recipients))
            return True
        except Exception as exc:
            logger.error("send_email | Failed: %s", exc)
            return False

    def post_to_teams(
        self,
        webhook_url: str | None = None,
    ) -> bool:
        """Post the report summary to a Microsoft Teams channel.

        Parameters
        ----------
        webhook_url : str or None
            Teams incoming webhook URL. When None, delivery is simulated.

        Returns
        -------
        bool
            True if post succeeded or was simulated.

        Notes
        -----
        This is a placeholder implementation. In production, configure with a
        real Teams incoming webhook URL from the target channel.
        """
        logger.info("post_to_teams | preparing markdown report")
        markdown_content = self.to_markdown()

        # Teams Adaptive Card payload
        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": (
                                    f"Daily Maintenance Report - {self.campus_name} - "
                                    f"{self.report_date.strftime('%m/%d/%Y')}"
                                ),
                                "weight": "Bolder",
                                "size": "Large",
                            },
                            {
                                "type": "TextBlock",
                                "text": markdown_content,
                                "wrap": True,
                            },
                        ],
                    },
                }
            ],
        }

        if webhook_url is None:
            logger.warning(
                "post_to_teams | No webhook URL configured. "
                "Teams post simulated (not sent)."
            )
            return True

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in (200, 202):
                    logger.info("post_to_teams | Posted successfully")
                    return True
                logger.warning("post_to_teams | Unexpected status: %d", resp.status)
                return False
        except (urllib.error.URLError, Exception) as exc:
            logger.error("post_to_teams | Failed: %s", exc)
            return False

    def save_to_file(
        self,
        output_dir: str | Path = ".",
        prefix: str = "daily_report",
    ) -> dict[str, str]:
        """Save report to local filesystem as HTML and text.

        Parameters
        ----------
        output_dir : str or Path
            Directory to write files into.
        prefix : str
            Filename prefix (date is appended automatically).

        Returns
        -------
        dict
            Mapping of format name to the absolute file path written.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        date_str = self.report_date.strftime("%Y-%m-%d")
        paths: dict[str, str] = {}

        # HTML
        html_file = output_path / f"{prefix}_{date_str}.html"
        html_file.write_text(self.to_html(), encoding="utf-8")
        paths["html"] = str(html_file.resolve())
        logger.info("save_to_file | HTML -> %s", paths["html"])

        # Text
        text_file = output_path / f"{prefix}_{date_str}.txt"
        text_file.write_text(self.to_text(), encoding="utf-8")
        paths["text"] = str(text_file.resolve())
        logger.info("save_to_file | Text -> %s", paths["text"])

        # JSON
        json_file = output_path / f"{prefix}_{date_str}.json"
        json_file.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        paths["json"] = str(json_file.resolve())
        logger.info("save_to_file | JSON -> %s", paths["json"])

        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _html_escape(text: str) -> str:
        """Minimal HTML escaping for user-supplied strings."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


# ---------------------------------------------------------------------------
# Sample data factory
# ---------------------------------------------------------------------------

BUILDINGS: list[str] = [
    "Main HQ",
    "Network Operations Center",
    "Data Center",
    "Maintenance Shop",
    "Parking Garage A",
    "Parking Garage B",
    "Cafeteria",
    "Training Center",
]

TRADES: list[str] = [
    "HVAC",
    "Electrical",
    "Plumbing",
    "General",
    "Elevator",
    "Fire Safety",
]

TECH_ROSTER: list[dict[str, str]] = [
    {"id": "T-201", "name": "Mike Rodriguez", "trade": "HVAC"},
    {"id": "T-202", "name": "Carlos Reyes", "trade": "Electrical"},
    {"id": "T-203", "name": "James Parker", "trade": "Plumbing"},
    {"id": "T-204", "name": "David Kim", "trade": "General"},
    {"id": "T-205", "name": "Robert Chen", "trade": "HVAC"},
    {"id": "T-206", "name": "Marcus Williams", "trade": "Elevator"},
    {"id": "T-207", "name": "Anthony Garcia", "trade": "Fire Safety"},
]

WO_DESCRIPTIONS: list[dict[str, Any]] = [
    {"title": "HVAC unit 3rd floor east wing not cooling", "trade": "HVAC", "building": "Main HQ", "floor": "3", "priority": Priority.P2},
    {"title": "Replace ballasts in parking garage B level 2", "trade": "Electrical", "building": "Parking Garage B", "floor": "2", "priority": Priority.P4},
    {"title": "Water leak under sink in 2nd floor men's restroom", "trade": "Plumbing", "building": "Main HQ", "floor": "2", "priority": Priority.P3},
    {"title": "Emergency exit sign flickering - west stairwell", "trade": "Electrical", "building": "Training Center", "floor": "1", "priority": Priority.P2},
    {"title": "Elevator B intermittent door sensor fault", "trade": "Elevator", "building": "Main HQ", "floor": "Lobby", "priority": Priority.P1},
    {"title": "Roof exhaust fan #4 bearing noise", "trade": "HVAC", "building": "Data Center", "floor": "Roof", "priority": Priority.P3},
    {"title": "Broken tile in cafeteria main dining area", "trade": "General", "building": "Cafeteria", "floor": "1", "priority": Priority.P5},
    {"title": "UPS battery replacement - server room B", "trade": "Electrical", "building": "Data Center", "floor": "1", "priority": Priority.P2},
    {"title": "Hot water heater pilot light out - maintenance shop", "trade": "Plumbing", "building": "Maintenance Shop", "floor": "1", "priority": Priority.P3},
    {"title": "Fire alarm pull station loose - NOC entrance", "trade": "Fire Safety", "building": "Network Operations Center", "floor": "1", "priority": Priority.P1},
    {"title": "Parking garage A level 3 light outage section C", "trade": "Electrical", "building": "Parking Garage A", "floor": "3", "priority": Priority.P4},
    {"title": "CRAC unit high discharge temp alert - Data Center", "trade": "HVAC", "building": "Data Center", "floor": "1", "priority": Priority.P1},
    {"title": "Clogged floor drain - training center lobby", "trade": "Plumbing", "building": "Training Center", "floor": "1", "priority": Priority.P4},
]

PM_DESCRIPTIONS: list[dict[str, str]] = [
    {"title": "PM - Quarterly fire extinguisher inspection Bldg A", "trade": "Fire Safety", "building": "Main HQ", "frequency": "quarterly"},
    {"title": "PM - Monthly HVAC filter change - Data Center", "trade": "HVAC", "building": "Data Center", "frequency": "monthly"},
    {"title": "PM - Weekly cooling tower water treatment", "trade": "HVAC", "building": "Main HQ", "frequency": "weekly"},
    {"title": "PM - Monthly elevator safety inspection", "trade": "Elevator", "building": "Main HQ", "frequency": "monthly"},
    {"title": "PM - Quarterly emergency generator load test", "trade": "Electrical", "building": "Network Operations Center", "frequency": "quarterly"},
    {"title": "PM - Monthly backflow preventer test", "trade": "Plumbing", "building": "Main HQ", "frequency": "monthly"},
    {"title": "PM - Weekly parking garage lighting check", "trade": "Electrical", "building": "Parking Garage A", "frequency": "weekly"},
    {"title": "PM - Monthly AHU belt inspection", "trade": "HVAC", "building": "Training Center", "frequency": "monthly"},
]

VENDOR_LIST: list[dict[str, Any]] = [
    {"name": "ThyssenKrupp Elevator", "specialty": "Elevator"},
    {"name": "Johnson Controls", "specialty": "HVAC/BAS"},
    {"name": "Siemens Fire Safety", "specialty": "Fire Safety"},
    {"name": "ABM Janitorial", "specialty": "General"},
    {"name": "Greenscape Landscaping", "specialty": "Grounds"},
]


def generate_sample_data(report_date: date | None = None) -> DailyReportGenerator:
    """Create a fully populated DailyReportGenerator with realistic BNSF campus data.

    Parameters
    ----------
    report_date : date or None
        The report date. Defaults to today.

    Returns
    -------
    DailyReportGenerator
        A generator pre-loaded with sample work orders, technicians, PMs,
        vendor orders, action items, ADR highlights, and 7 days of metrics.
    """
    rng = random.Random(42)  # Deterministic for reproducibility
    today = report_date or date.today()
    now = datetime.now()

    gen = DailyReportGenerator(report_date=today, campus_name="BNSF HQ Campus")

    # --- Work Orders (8-15 open) -------------------------------------------
    num_open = rng.randint(8, 15)
    wo_pool = WO_DESCRIPTIONS.copy()
    rng.shuffle(wo_pool)
    selected_wos = wo_pool[:num_open]

    for i, wo_def in enumerate(selected_wos):
        age_hours = rng.uniform(0.5, 72.0)
        created_at = now - timedelta(hours=age_hours)
        priority = wo_def["priority"]
        sla_deadline = created_at + timedelta(hours=priority.sla_hours)

        # Determine SLA status
        if now > sla_deadline:
            sla_status = SLAStatus.RED
        elif (sla_deadline - now).total_seconds() < gen.config.sla_warning_hours * 3600:
            sla_status = SLAStatus.YELLOW
        else:
            sla_status = SLAStatus.GREEN

        # Assign a technician (prefer matching trade)
        matching_techs = [t for t in TECH_ROSTER if t["trade"] == wo_def["trade"]]
        tech = rng.choice(matching_techs) if matching_techs else rng.choice(TECH_ROSTER)

        status_choices = [WOStatus.OPEN, WOStatus.IN_PROGRESS, WOStatus.ON_HOLD]
        if priority in (Priority.P1, Priority.P2):
            status_choices = [WOStatus.IN_PROGRESS, WOStatus.OPEN]
        wo_status = rng.choice(status_choices)

        gen.work_orders.append(WorkOrderSummary(
            wo_number=f"WO-{50100 + i}",
            title=wo_def["title"],
            building=wo_def["building"],
            floor=wo_def["floor"],
            trade=wo_def["trade"],
            priority=priority,
            assigned_tech=tech["name"],
            created_at=created_at,
            sla_deadline=sla_deadline,
            sla_status=sla_status,
            age_hours=round(age_hours, 1),
            status=wo_status,
        ))

    # Add a couple of completed WOs for yesterday's stats
    for i in range(rng.randint(5, 9)):
        comp_age = rng.uniform(2, 36)
        comp_created = now - timedelta(hours=comp_age + rng.uniform(1, 8))
        comp_priority = rng.choice(list(Priority))
        tech = rng.choice(TECH_ROSTER)
        gen.work_orders.append(WorkOrderSummary(
            wo_number=f"WO-{50200 + i}",
            title=f"Completed: {rng.choice(['Replace', 'Repair', 'Inspect', 'Service'])} "
                  f"{rng.choice(['fixture', 'unit', 'valve', 'panel', 'sensor'])} "
                  f"in {rng.choice(BUILDINGS)}",
            building=rng.choice(BUILDINGS),
            floor=str(rng.randint(1, 5)),
            trade=tech["trade"],
            priority=comp_priority,
            assigned_tech=tech["name"],
            created_at=comp_created,
            sla_deadline=comp_created + timedelta(hours=comp_priority.sla_hours),
            sla_status=SLAStatus.GREEN,
            age_hours=round(comp_age, 1),
            status=WOStatus.COMPLETED,
        ))

    # --- PM Tasks (4-6 due today + some overdue) ---------------------------
    pm_pool = PM_DESCRIPTIONS.copy()
    rng.shuffle(pm_pool)
    num_pm_today = rng.randint(4, 6)
    selected_pms = pm_pool[:num_pm_today]

    for i, pm_def in enumerate(selected_pms):
        matching = [t for t in TECH_ROSTER if t["trade"] == pm_def["trade"]]
        tech = rng.choice(matching) if matching else rng.choice(TECH_ROSTER)
        is_completed = rng.random() < 0.3  # 30% already done
        is_overdue = not is_completed and rng.random() < 0.2  # 20% of remaining are overdue
        due = today if not is_overdue else today - timedelta(days=rng.randint(1, 3))

        gen.pm_tasks.append(PMTask(
            pm_id=f"PM-{8000 + i}",
            title=pm_def["title"],
            building=pm_def["building"],
            floor=str(rng.randint(1, 3)),
            trade=pm_def["trade"],
            assigned_tech=tech["name"],
            due_date=due,
            frequency=pm_def["frequency"],
            is_overdue=is_overdue,
            is_completed=is_completed,
            estimated_minutes=rng.choice([30, 45, 60, 90, 120]),
        ))

    # Add an extra overdue PM
    gen.pm_tasks.append(PMTask(
        pm_id="PM-8099",
        title="PM - Overdue quarterly chiller inspection",
        building="Data Center",
        floor="Roof",
        trade="HVAC",
        assigned_tech="Mike Rodriguez",
        due_date=today - timedelta(days=5),
        frequency="quarterly",
        is_overdue=True,
        is_completed=False,
        estimated_minutes=120,
    ))

    # --- Technicians (5-7 with varied workloads) ---------------------------
    for tech_def in TECH_ROSTER:
        tasks_count = rng.randint(1, 5)
        completed_count = rng.randint(0, min(2, tasks_count))
        est_hours = round(tasks_count * rng.uniform(0.75, 2.0), 1)

        # Determine status
        if rng.random() < 0.1:
            t_status = TechStatus.BREAK
            current = "On break"
        elif rng.random() < 0.1:
            t_status = TechStatus.OFF_SITE
            current = "Vendor pickup at supply house"
        elif rng.random() < 0.1:
            t_status = TechStatus.NOT_STARTED
            current = "Shift not started"
        else:
            t_status = TechStatus.ACTIVE
            tech_wos = [
                wo for wo in gen.work_orders
                if wo.assigned_tech == tech_def["name"]
                and wo.status == WOStatus.IN_PROGRESS
            ]
            current = tech_wos[0].title[:50] if tech_wos else "En route to assignment"

        gen.technicians.append(TechnicianStatus(
            tech_id=tech_def["id"],
            name=tech_def["name"],
            trade=tech_def["trade"],
            tasks_today=tasks_count,
            estimated_hours=est_hours,
            completed_today=completed_count,
            current_task=current,
            status=t_status,
        ))

    # --- Vendor Work Orders ------------------------------------------------
    gen.vendor_orders = [
        VendorWorkOrder(
            wo_number="VWO-3001",
            vendor_name="ThyssenKrupp Elevator",
            description="Elevator B door operator replacement",
            building="Main HQ",
            status="Scheduled",
            scheduled_date=today,
            invoice_status="Pending",
            estimated_cost=4500.00,
        ),
        VendorWorkOrder(
            wo_number="VWO-3002",
            vendor_name="Johnson Controls",
            description="BAS controller firmware update - all AHUs",
            building="Data Center",
            status="In Progress",
            scheduled_date=today - timedelta(days=1),
            invoice_status="Pending",
            estimated_cost=8200.00,
        ),
        VendorWorkOrder(
            wo_number="VWO-3003",
            vendor_name="Siemens Fire Safety",
            description="Annual fire alarm panel inspection",
            building="Network Operations Center",
            status="Awaiting Parts",
            scheduled_date=today + timedelta(days=3),
            invoice_status="Not Invoiced",
            estimated_cost=3100.00,
        ),
        VendorWorkOrder(
            wo_number="VWO-3004",
            vendor_name="Greenscape Landscaping",
            description="Monthly grounds maintenance and irrigation check",
            building="Main HQ",
            status="Completed",
            scheduled_date=today - timedelta(days=2),
            invoice_status="Pending",
            estimated_cost=1850.00,
        ),
    ]

    # --- Daily Metrics (today) ---------------------------------------------
    open_count = len([
        wo for wo in gen.work_orders
        if wo.status not in (WOStatus.COMPLETED, WOStatus.CLOSED)
    ])
    completed_today_count = len([
        wo for wo in gen.work_orders
        if wo.status in (WOStatus.COMPLETED, WOStatus.CLOSED)
    ])
    pm_completed = sum(1 for pm in gen.pm_tasks if pm.is_completed)
    pm_due = len(gen.pm_tasks)
    pm_pct = round((pm_completed / pm_due) * 100, 1) if pm_due else 0.0

    gen.daily_metrics = DailyMetrics(
        date=today,
        total_open=open_count,
        total_closed_today=completed_today_count,
        total_created_today=rng.randint(3, 8),
        avg_cycle_hours=round(rng.uniform(4.0, 18.0), 1),
        pm_due=pm_due,
        pm_completed=pm_completed,
        pm_compliance_pct=pm_pct,
        sla_breaches=len([wo for wo in gen.work_orders if wo.sla_status == SLAStatus.RED]),
        callbacks=rng.randint(0, 2),
    )

    # --- Weekly Metrics (7 days) -------------------------------------------
    for day_offset in range(7, 0, -1):
        d = today - timedelta(days=day_offset)
        day_open = rng.randint(8, 18)
        day_closed = rng.randint(4, 10)
        day_created = rng.randint(3, 9)
        day_pm_due = rng.randint(3, 7)
        day_pm_done = rng.randint(2, day_pm_due)
        day_pct = round((day_pm_done / day_pm_due) * 100, 1) if day_pm_due else 100.0

        gen.weekly_metrics.append(DailyMetrics(
            date=d,
            total_open=day_open,
            total_closed_today=day_closed,
            total_created_today=day_created,
            avg_cycle_hours=round(rng.uniform(5.0, 16.0), 1),
            pm_due=day_pm_due,
            pm_completed=day_pm_done,
            pm_compliance_pct=day_pct,
            sla_breaches=rng.randint(0, 3),
            callbacks=rng.randint(0, 2),
        ))

    # --- Action Items ------------------------------------------------------
    gen.action_items = [
        ActionItem(
            item_id="AI-001",
            description="Approve overtime for Carlos Reyes - UPS battery replacement requires after-hours work",
            urgency="high",
            related_wo="WO-50107",
            owner="Tony Vita",
            due_by="noon",
        ),
        ActionItem(
            item_id="AI-002",
            description="Review ThyssenKrupp elevator repair proposal ($4,500) - door operator replacement",
            urgency="high",
            related_wo="VWO-3001",
            owner="Tony Vita",
            due_by="EOD",
        ),
        ActionItem(
            item_id="AI-003",
            description="Reassign PM backlog - 1 overdue chiller inspection needs scheduling",
            urgency="medium",
            related_wo="PM-8099",
            owner="Juan Guerra",
            due_by="EOD",
        ),
        ActionItem(
            item_id="AI-004",
            description="Coordinate with NOC manager on fire alarm pull station repair window",
            urgency="high",
            related_wo="WO-50109",
            owner="Juan Guerra",
            due_by="ASAP",
        ),
        ActionItem(
            item_id="AI-005",
            description="Review weekly PM compliance trend - declining 3 days running",
            urgency="medium",
            related_wo=None,
            owner="Juan Guerra",
            due_by="EOD",
        ),
    ]

    # --- ADR Highlights ----------------------------------------------------
    gen.adr_highlights = [
        ADRHighlight(
            adr_id="ADR-047",
            title="Adopt predictive maintenance scheduling for HVAC fleet",
            status="proposed",
            summary=(
                "Replace calendar-based PM schedules with sensor-driven predictive "
                "triggers for the 12 main AHUs. Expected to reduce unnecessary PMs "
                "by 20% while catching failures earlier."
            ),
            impact_area="HVAC Operations / PM Scheduling",
            date_created=today - timedelta(days=1),
        ),
        ADRHighlight(
            adr_id="ADR-048",
            title="Standardize vendor invoice workflow through Corrigo",
            status="proposed",
            summary=(
                "Route all vendor invoices through Corrigo WO close-out process "
                "instead of separate AP email chain. Eliminates duplicate data entry "
                "and provides audit trail."
            ),
            impact_area="Vendor Management / Finance",
            date_created=today - timedelta(days=1),
        ),
    ]

    logger.info(
        "generate_sample_data | WOs=%d, PMs=%d, Techs=%d, Vendors=%d, Actions=%d, ADRs=%d",
        len(gen.work_orders),
        len(gen.pm_tasks),
        len(gen.technicians),
        len(gen.vendor_orders),
        len(gen.action_items),
        len(gen.adr_highlights),
    )
    return gen


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate a full sample report and output text + save HTML."""
    logger.setLevel(logging.INFO)
    logger.info("=" * 60)
    logger.info("Daily Report Generator - Sample Run")
    logger.info("Campus: BNSF HQ Campus - Fort Worth, TX")
    logger.info("Recipients: Tony Vita, Juan Guerra")
    logger.info("=" * 60)

    # Generate sample data and full report
    generator = generate_sample_data()
    generator.generate_full_report()

    # Print text version to stdout
    print(generator.to_text())

    # Save HTML to sample_reports directory
    sample_dir = Path("/home/user/JLL-BNSF-Corrigo-MOA/sample_reports")
    sample_dir.mkdir(parents=True, exist_ok=True)

    html_path = sample_dir / "daily_report_sample.html"
    html_path.write_text(generator.to_html(), encoding="utf-8")
    logger.info("HTML report saved to: %s", html_path)

    # Also save text version
    text_path = sample_dir / "daily_report_sample.txt"
    text_path.write_text(generator.to_text(), encoding="utf-8")
    logger.info("Text report saved to: %s", text_path)

    # Save JSON version
    json_path = sample_dir / "daily_report_sample.json"
    json_path.write_text(
        json.dumps(generator.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("JSON report saved to: %s", json_path)

    # Print markdown version
    print("\n" + "=" * 60)
    print("MARKDOWN VERSION (for Teams/Slack)")
    print("=" * 60)
    print(generator.to_markdown())

    # Simulate delivery
    print("\n" + "=" * 60)
    print("DELIVERY SIMULATION")
    print("=" * 60)
    generator.send_email()
    generator.post_to_teams()

    logger.info("Sample run complete.")


if __name__ == "__main__":
    main()

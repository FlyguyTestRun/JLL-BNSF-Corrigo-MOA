"""
MAO System Configuration
Central configuration for all agents, API connections, and operational parameters.

Environment variables are loaded from .env file (not committed to git).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class CorrigoSettings:
    """Corrigo API connection settings."""
    client_id: str = os.getenv("CORRIGO_CLIENT_ID", "")
    client_secret: str = os.getenv("CORRIGO_CLIENT_SECRET", "")
    company_name: str = os.getenv("CORRIGO_COMPANY", "jll-bnsf")
    base_url: str = os.getenv("CORRIGO_API_URL", "https://am-api.corrigo.com/api/v1")
    portal_url: str = "https://jll-bnsf.corrigo.com"
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass
class CampusSettings:
    """BNSF campus configuration."""
    name: str = "BNSF Railway Headquarters Campus"
    address: str = "2400 Lou Menk Dr, Fort Worth, TX 76131"
    timezone: str = "America/Chicago"
    buildings: list[str] = field(default_factory=lambda: [
        "Main HQ",
        "Network Operations Center",
        "Data Center",
        "Maintenance Shop",
        "Parking Garage A",
        "Parking Garage B",
        "Cafeteria",
        "Training Center",
    ])
    work_start_hour: int = 6  # 6 AM CST
    work_end_hour: int = 17   # 5 PM CST


@dataclass
class TokenBudgetSettings:
    """Token conservation settings for MAO agents."""
    max_tokens_per_agent_call: int = 500
    max_tokens_per_orchestration_cycle: int = 5000
    max_tokens_per_day: int = 50000
    warn_at_pct: float = 0.8  # Warn when 80% of budget used
    log_all_usage: bool = True


@dataclass
class ReportSettings:
    """Daily report configuration."""
    recipients: list[str] = field(default_factory=lambda: [
        "tony.vita@jll.com",
        "juan.guerra@jll.com",
    ])
    send_time: str = "06:00"  # 6 AM CST
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
    report_output_dir: str = str(PROJECT_ROOT / "sample_reports")


@dataclass
class AgentSettings:
    """Per-agent configuration."""
    shaw_goals_enabled: bool = True
    pm_systematic_enabled: bool = True
    morning_todo_enabled: bool = True
    adr_engine_enabled: bool = True

    # PM Systematic Agent
    pm_overdue_threshold_days: int = 1
    pm_efficiency_target_pct: float = 85.0

    # Morning TODO Agent
    priority_weights: dict[str, float] = field(default_factory=lambda: {
        "sla_urgency": 0.40,
        "safety_impact": 0.25,
        "occupant_impact": 0.15,
        "efficiency_grouping": 0.10,
        "asset_criticality": 0.10,
    })

    # ADR Engine
    adr_auto_triggers: dict[str, bool] = field(default_factory=lambda: {
        "sla_breach": True,
        "pm_overrun_20pct": True,
        "chronic_asset_3wo_30d": True,
        "rework_within_7d": True,
        "parts_delay_2hr": True,
        "safety_concern": True,
    })


@dataclass
class MAOConfig:
    """Master configuration combining all settings."""
    corrigo: CorrigoSettings = field(default_factory=CorrigoSettings)
    campus: CampusSettings = field(default_factory=CampusSettings)
    tokens: TokenBudgetSettings = field(default_factory=TokenBudgetSettings)
    reports: ReportSettings = field(default_factory=ReportSettings)
    agents: AgentSettings = field(default_factory=AgentSettings)
    debug_mode: bool = os.getenv("MAO_DEBUG", "false").lower() == "true"
    demo_mode: bool = True  # Use sample data instead of live API


# Singleton config instance
config = MAOConfig()

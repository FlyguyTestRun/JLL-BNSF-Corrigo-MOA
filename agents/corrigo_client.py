"""
Corrigo Enterprise REST API Client
Shared client used by all MAO agents to interact with the Corrigo CMMS.

Authentication: OAuth 2.0 Bearer Token
Base URL: https://am-api.corrigo.com (Americas region)
Docs: https://developer.corrigo.com

This client handles:
- OAuth token acquisition and refresh
- Work order CRUD operations
- Asset and location queries
- PM schedule retrieval
- Webhook event parsing
- Rate limiting and retry logic
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CorrigoRegion(Enum):
    """Corrigo API regional endpoints."""
    AMERICAS = "https://am-api.corrigo.com"
    EMEA = "https://eu-api.corrigo.com"
    APAC = "https://ap-api.corrigo.com"


class WOStatus(Enum):
    """Corrigo work order status values."""
    NEW = "New"
    DISPATCHED = "Dispatched"
    IN_PROGRESS = "InProgress"
    ON_HOLD = "OnHold"
    COMPLETED = "Completed"
    CLOSED = "Closed"
    CANCELLED = "Cancelled"


class WOPriority(Enum):
    """Work order priority levels."""
    EMERGENCY = "P1"
    URGENT = "P2"
    HIGH = "P3"
    MEDIUM = "P4"
    LOW = "P5"


@dataclass
class CorrigoConfig:
    """Configuration for Corrigo API connection."""
    client_id: str = ""
    client_secret: str = ""
    company_name: str = "jll-bnsf"
    region: CorrigoRegion = CorrigoRegion.AMERICAS
    api_version: str = "v1"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    @property
    def base_url(self) -> str:
        return f"{self.region.value}/api/{self.api_version}"

    @property
    def token_url(self) -> str:
        return f"{self.region.value}/oauth/token"


@dataclass
class APIResponse:
    """Standardized API response wrapper."""
    success: bool
    status_code: int
    data: Any = None
    error: str | None = None
    request_time_ms: float = 0.0
    token_cost: int = 0  # Track token usage for MAO budget


@dataclass
class WorkOrder:
    """Corrigo work order data model."""
    id: int = 0
    wo_number: str = ""
    title: str = ""
    description: str = ""
    status: str = "New"
    priority: str = "P4"
    trade: str = ""
    building: str = ""
    floor: str = ""
    space: str = ""
    asset_id: int | None = None
    asset_name: str = ""
    assigned_tech_id: int | None = None
    assigned_tech_name: str = ""
    vendor_id: int | None = None
    vendor_name: str = ""
    created_at: str = ""
    updated_at: str = ""
    due_date: str = ""
    sla_deadline: str = ""
    completed_at: str | None = None
    is_pm: bool = False
    pm_schedule_id: int | None = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    nte_amount: float = 0.0
    actual_cost: float = 0.0
    custom_fields: dict = field(default_factory=dict)

    @property
    def age_hours(self) -> float:
        if not self.created_at:
            return 0.0
        try:
            created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            return (datetime.now(created.tzinfo) - created).total_seconds() / 3600
        except (ValueError, TypeError):
            return 0.0

    @property
    def sla_status(self) -> str:
        """Return green/yellow/red based on SLA deadline proximity."""
        if not self.sla_deadline:
            return "green"
        try:
            deadline = datetime.fromisoformat(self.sla_deadline.replace("Z", "+00:00"))
            now = datetime.now(deadline.tzinfo)
            remaining = (deadline - now).total_seconds() / 3600
            if remaining < 0:
                return "red"
            elif remaining < 4:
                return "yellow"
            return "green"
        except (ValueError, TypeError):
            return "green"


@dataclass
class Asset:
    """Corrigo asset data model."""
    id: int = 0
    name: str = ""
    asset_type: str = ""
    building: str = ""
    floor: str = ""
    space: str = ""
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    install_date: str = ""
    warranty_end: str = ""
    status: str = "Active"
    wo_count_30d: int = 0
    last_pm_date: str = ""
    next_pm_date: str = ""


@dataclass
class PMSchedule:
    """Corrigo preventive maintenance schedule."""
    id: int = 0
    name: str = ""
    asset_id: int = 0
    asset_name: str = ""
    trade: str = ""
    frequency: str = ""  # Daily, Weekly, Monthly, Quarterly, Annual
    building: str = ""
    floor: str = ""
    last_completed: str = ""
    next_due: str = ""
    estimated_minutes: int = 0
    checklist_id: int | None = None
    assigned_tech_id: int | None = None
    is_overdue: bool = False


class CorrigoClient:
    """
    Client for the Corrigo Enterprise REST API.

    Handles authentication, request/response formatting, rate limiting,
    and provides typed methods for common operations.

    Usage:
        config = CorrigoConfig(client_id="xxx", client_secret="yyy")
        client = CorrigoClient(config)
        client.authenticate()
        work_orders = client.get_open_work_orders()
    """

    def __init__(self, config: CorrigoConfig | None = None):
        self.config = config or CorrigoConfig()
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._request_count: int = 0
        self._total_request_time_ms: float = 0.0
        logger.info("CorrigoClient initialized for company: %s", self.config.company_name)

    def authenticate(self) -> bool:
        """Acquire OAuth 2.0 bearer token from Corrigo."""
        logger.info("Authenticating with Corrigo API at %s", self.config.token_url)
        # In production, this would POST to the token endpoint:
        # POST {token_url}
        # Content-Type: application/x-www-form-urlencoded
        # Body: grant_type=client_credentials&client_id={id}&client_secret={secret}
        #
        # Response: { "access_token": "...", "token_type": "Bearer", "expires_in": 3600 }
        self._access_token = "placeholder_token"
        self._token_expires_at = time.time() + 3600
        logger.info("Authentication successful (placeholder mode)")
        return True

    def _ensure_auth(self) -> None:
        """Refresh token if expired."""
        if not self._access_token or time.time() >= self._token_expires_at:
            self.authenticate()

    def _headers(self) -> dict[str, str]:
        """Build request headers with auth token."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "CompanyName": self.config.company_name,
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        body: dict | None = None,
    ) -> APIResponse:
        """
        Execute an API request with retry logic.

        In production, this would use httpx or requests.
        Currently returns placeholder data for development.
        """
        self._ensure_auth()
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        start = time.time()
        self._request_count += 1

        logger.debug("%s %s params=%s", method, url, params)

        # Placeholder response — in production, replace with actual HTTP call:
        # response = httpx.request(method, url, headers=self._headers(),
        #                          params=params, json=body,
        #                          timeout=self.config.timeout_seconds)
        elapsed_ms = (time.time() - start) * 1000
        self._total_request_time_ms += elapsed_ms

        return APIResponse(
            success=True,
            status_code=200,
            data=[],
            request_time_ms=elapsed_ms,
        )

    # --- Work Order Operations ---

    def get_open_work_orders(
        self,
        building: str | None = None,
        trade: str | None = None,
        priority: str | None = None,
        limit: int = 100,
    ) -> list[WorkOrder]:
        """Fetch all open work orders, optionally filtered."""
        params: dict[str, Any] = {
            "status": "New,Dispatched,InProgress,OnHold",
            "limit": limit,
        }
        if building:
            params["building"] = building
        if trade:
            params["trade"] = trade
        if priority:
            params["priority"] = priority

        resp = self._request("GET", "/WorkOrders", params=params)
        if not resp.success:
            logger.error("Failed to fetch work orders: %s", resp.error)
            return []

        # In production, parse resp.data into WorkOrder objects
        return _generate_sample_work_orders()

    def get_work_order(self, wo_id: int) -> WorkOrder | None:
        """Fetch a single work order by ID."""
        resp = self._request("GET", f"/WorkOrders/{wo_id}")
        return WorkOrder(id=wo_id) if resp.success else None

    def update_work_order(self, wo_id: int, updates: dict) -> bool:
        """Update a work order's fields."""
        resp = self._request("PATCH", f"/WorkOrders/{wo_id}", body=updates)
        return resp.success

    def create_work_order(self, wo_data: dict) -> WorkOrder | None:
        """Create a new work order."""
        resp = self._request("POST", "/WorkOrders", body=wo_data)
        return WorkOrder(**wo_data) if resp.success else None

    def close_work_order(self, wo_id: int, completion_notes: str = "") -> bool:
        """Close/complete a work order."""
        return self.update_work_order(wo_id, {
            "status": "Completed",
            "completionNotes": completion_notes,
            "completedAt": datetime.now().isoformat(),
        })

    # --- PM Schedule Operations ---

    def get_pm_schedules(
        self,
        due_date: str | None = None,
        overdue_only: bool = False,
    ) -> list[PMSchedule]:
        """Fetch PM schedules, optionally filtered by due date."""
        params: dict[str, Any] = {}
        if due_date:
            params["dueDate"] = due_date
        if overdue_only:
            params["overdue"] = True

        resp = self._request("GET", "/PMSchedules", params=params)
        if not resp.success:
            return []
        return _generate_sample_pm_schedules()

    def get_overdue_pms(self) -> list[PMSchedule]:
        """Convenience method: fetch all overdue PMs."""
        return self.get_pm_schedules(overdue_only=True)

    # --- Asset Operations ---

    def get_asset(self, asset_id: int) -> Asset | None:
        """Fetch asset details."""
        resp = self._request("GET", f"/Assets/{asset_id}")
        return Asset(id=asset_id) if resp.success else None

    def get_assets_by_building(self, building: str) -> list[Asset]:
        """Fetch all assets in a building."""
        resp = self._request("GET", "/Assets", params={"building": building})
        return []

    def get_chronic_assets(self, wo_threshold: int = 3, days: int = 30) -> list[Asset]:
        """Find assets with repeated failures (3+ WOs in 30 days)."""
        resp = self._request("GET", "/Assets", params={
            "minWoCount": wo_threshold,
            "dayRange": days,
        })
        return []

    # --- Technician Operations ---

    def get_technicians(self) -> list[dict]:
        """Fetch all active technicians."""
        resp = self._request("GET", "/Employees", params={"role": "Technician"})
        return _generate_sample_technicians()

    # --- Webhook Parsing ---

    @staticmethod
    def parse_webhook_event(payload: dict) -> dict:
        """Parse an incoming Corrigo webhook event."""
        return {
            "event_type": payload.get("EventType", "Unknown"),
            "entity_type": payload.get("EntityType", ""),
            "entity_id": payload.get("EntityId", 0),
            "timestamp": payload.get("Timestamp", ""),
            "changes": payload.get("Changes", {}),
        }

    # --- Utility ---

    def get_stats(self) -> dict:
        """Return client usage statistics."""
        return {
            "total_requests": self._request_count,
            "total_time_ms": round(self._total_request_time_ms, 2),
            "avg_time_ms": round(
                self._total_request_time_ms / max(self._request_count, 1), 2
            ),
            "authenticated": self._access_token is not None,
        }


# --- Sample Data Generators (used in dev/demo mode) ---

def _generate_sample_work_orders() -> list[WorkOrder]:
    """Generate realistic sample work orders for the BNSF campus."""
    now = datetime.now()
    return [
        WorkOrder(
            id=10001, wo_number="WO-2026-10001",
            title="HVAC unit not cooling — 3rd floor east wing",
            description="Occupant reports AC blowing warm air in conference room 3E-204",
            status="InProgress", priority="P2", trade="HVAC",
            building="Main HQ", floor="3", space="3E-204",
            asset_id=5001, asset_name="RTU-3E-01",
            assigned_tech_id=101, assigned_tech_name="Mike Torres",
            created_at=(now - timedelta(hours=6)).isoformat(),
            sla_deadline=(now + timedelta(hours=2)).isoformat(),
            estimated_hours=2.0, is_pm=False,
        ),
        WorkOrder(
            id=10002, wo_number="WO-2026-10002",
            title="Bathroom faucet leak — 2nd floor men's restroom",
            description="Dripping faucet in 2nd floor men's room near elevator bank",
            status="New", priority="P4", trade="Plumbing",
            building="Main HQ", floor="2", space="2-MRR",
            assigned_tech_id=102, assigned_tech_name="Carlos Mendez",
            created_at=(now - timedelta(hours=18)).isoformat(),
            sla_deadline=(now + timedelta(hours=6)).isoformat(),
            estimated_hours=1.0, is_pm=False,
        ),
        WorkOrder(
            id=10003, wo_number="WO-2026-10003",
            title="Replace ballasts — Parking Garage B Level 2",
            description="Multiple light fixtures out on PG-B level 2, safety concern",
            status="Dispatched", priority="P3", trade="Electrical",
            building="Parking Garage B", floor="2",
            assigned_tech_id=103, assigned_tech_name="David Kim",
            created_at=(now - timedelta(hours=24)).isoformat(),
            sla_deadline=(now + timedelta(hours=4)).isoformat(),
            estimated_hours=3.0, is_pm=False,
        ),
        WorkOrder(
            id=10004, wo_number="WO-2026-10004",
            title="Elevator stuck on 4th floor",
            description="Elevator 3 in Main HQ stuck between floors, vendor dispatched",
            status="InProgress", priority="P1", trade="Elevator",
            building="Main HQ", floor="4",
            vendor_id=201, vendor_name="Otis Elevator Co.",
            created_at=(now - timedelta(hours=2)).isoformat(),
            sla_deadline=(now + timedelta(hours=1)).isoformat(),
            estimated_hours=4.0, is_pm=False,
        ),
        WorkOrder(
            id=10005, wo_number="WO-2026-10005",
            title="Kitchen hood fan vibration — Cafeteria",
            description="Exhaust hood fan making unusual vibration noise during lunch service",
            status="New", priority="P3", trade="HVAC",
            building="Cafeteria", floor="1",
            asset_id=5010, asset_name="EXH-CAF-01",
            created_at=(now - timedelta(hours=8)).isoformat(),
            sla_deadline=(now + timedelta(hours=8)).isoformat(),
            estimated_hours=1.5, is_pm=False,
        ),
        WorkOrder(
            id=10006, wo_number="WO-2026-10006",
            title="UPS battery alarm — Data Center",
            description="UPS system showing battery fault alarm on unit DC-UPS-02",
            status="InProgress", priority="P1", trade="Electrical",
            building="Data Center", floor="1",
            asset_id=5020, asset_name="DC-UPS-02",
            assigned_tech_id=104, assigned_tech_name="James Wright",
            created_at=(now - timedelta(hours=1)).isoformat(),
            sla_deadline=(now + timedelta(hours=2)).isoformat(),
            estimated_hours=2.0, is_pm=False,
        ),
        WorkOrder(
            id=10007, wo_number="WO-2026-10007",
            title="Exterior door closer broken — Training Center",
            description="Main entrance door closer failed, door won't close properly",
            status="New", priority="P3", trade="General",
            building="Training Center", floor="1",
            created_at=(now - timedelta(hours=12)).isoformat(),
            sla_deadline=(now + timedelta(hours=12)).isoformat(),
            estimated_hours=1.0, is_pm=False,
        ),
        WorkOrder(
            id=10008, wo_number="WO-2026-10008",
            title="Fire alarm panel trouble signal — NOC",
            description="Fire alarm panel showing supervisory trouble on zone 3",
            status="Dispatched", priority="P2", trade="Fire Safety",
            building="Network Operations Center", floor="1",
            vendor_id=202, vendor_name="Simplex Grinnell",
            created_at=(now - timedelta(hours=3)).isoformat(),
            sla_deadline=(now + timedelta(hours=3)).isoformat(),
            estimated_hours=2.0, is_pm=False,
        ),
        WorkOrder(
            id=10009, wo_number="WO-2026-10009",
            title="PM — Quarterly fire extinguisher inspection Bldg A",
            description="Quarterly inspection of all fire extinguishers in Main HQ",
            status="New", priority="P4", trade="Fire Safety",
            building="Main HQ", floor="ALL",
            assigned_tech_id=105, assigned_tech_name="Robert Chen",
            created_at=(now - timedelta(hours=1)).isoformat(),
            sla_deadline=(now + timedelta(hours=48)).isoformat(),
            estimated_hours=4.0, is_pm=True, pm_schedule_id=301,
        ),
        WorkOrder(
            id=10010, wo_number="WO-2026-10010",
            title="PM — Monthly HVAC filter change Data Center",
            description="Monthly filter replacement on all CRAC units in data center",
            status="New", priority="P3", trade="HVAC",
            building="Data Center", floor="1",
            assigned_tech_id=101, assigned_tech_name="Mike Torres",
            created_at=now.isoformat(),
            sla_deadline=(now + timedelta(hours=24)).isoformat(),
            estimated_hours=3.0, is_pm=True, pm_schedule_id=302,
        ),
    ]


def _generate_sample_pm_schedules() -> list[PMSchedule]:
    """Generate sample PM schedules."""
    now = datetime.now()
    return [
        PMSchedule(
            id=301, name="Fire Extinguisher Quarterly Inspection",
            asset_id=0, asset_name="All Fire Extinguishers",
            trade="Fire Safety", frequency="Quarterly",
            building="Main HQ", floor="ALL",
            last_completed=(now - timedelta(days=88)).isoformat(),
            next_due=now.isoformat(),
            estimated_minutes=240, assigned_tech_id=105, is_overdue=False,
        ),
        PMSchedule(
            id=302, name="CRAC Unit Filter Change",
            asset_id=5021, asset_name="CRAC-DC-01 through CRAC-DC-06",
            trade="HVAC", frequency="Monthly",
            building="Data Center", floor="1",
            last_completed=(now - timedelta(days=32)).isoformat(),
            next_due=now.isoformat(),
            estimated_minutes=180, assigned_tech_id=101, is_overdue=True,
        ),
        PMSchedule(
            id=303, name="Elevator Monthly Inspection",
            asset_id=5030, asset_name="Elevators 1-6",
            trade="Elevator", frequency="Monthly",
            building="Main HQ",
            last_completed=(now - timedelta(days=28)).isoformat(),
            next_due=(now + timedelta(days=2)).isoformat(),
            estimated_minutes=120, is_overdue=False,
        ),
        PMSchedule(
            id=304, name="Generator Weekly Run Test",
            asset_id=5040, asset_name="GEN-01 Emergency Generator",
            trade="Electrical", frequency="Weekly",
            building="Data Center", floor="EXT",
            last_completed=(now - timedelta(days=8)).isoformat(),
            next_due=(now - timedelta(days=1)).isoformat(),
            estimated_minutes=45, assigned_tech_id=104, is_overdue=True,
        ),
        PMSchedule(
            id=305, name="Cooling Tower Chemical Treatment",
            asset_id=5050, asset_name="CT-01, CT-02",
            trade="HVAC", frequency="Weekly",
            building="Main HQ", floor="Roof",
            last_completed=(now - timedelta(days=6)).isoformat(),
            next_due=(now + timedelta(days=1)).isoformat(),
            estimated_minutes=60, assigned_tech_id=101, is_overdue=False,
        ),
    ]


def _generate_sample_technicians() -> list[dict]:
    """Generate sample technician roster."""
    return [
        {"id": 101, "name": "Mike Torres", "trade": "HVAC", "status": "Active", "phone": "817-555-0101"},
        {"id": 102, "name": "Carlos Mendez", "trade": "Plumbing", "status": "Active", "phone": "817-555-0102"},
        {"id": 103, "name": "David Kim", "trade": "Electrical", "status": "Active", "phone": "817-555-0103"},
        {"id": 104, "name": "James Wright", "trade": "Electrical", "status": "Active", "phone": "817-555-0104"},
        {"id": 105, "name": "Robert Chen", "trade": "General/Fire Safety", "status": "Active", "phone": "817-555-0105"},
        {"id": 106, "name": "Anthony Ruiz", "trade": "General", "status": "Active", "phone": "817-555-0106"},
        {"id": 107, "name": "Brandon Lee", "trade": "HVAC", "status": "Off Today", "phone": "817-555-0107"},
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    client = CorrigoClient()
    client.authenticate()

    print("\n=== Open Work Orders ===")
    wos = client.get_open_work_orders()
    for wo in wos:
        sla = wo.sla_status
        icon = {"green": "[OK]", "yellow": "[!!]", "red": "[XX]"}.get(sla, "")
        print(f"  {wo.wo_number} | {wo.priority} | {wo.trade:<12} | {wo.title[:50]} | SLA: {icon}")

    print(f"\n=== PM Schedules ===")
    pms = client.get_pm_schedules()
    for pm in pms:
        flag = " ** OVERDUE **" if pm.is_overdue else ""
        print(f"  {pm.name[:45]:<45} | {pm.frequency:<10} | {pm.building}{flag}")

    print(f"\n=== Technician Roster ===")
    techs = client.get_technicians()
    for t in techs:
        print(f"  {t['name']:<20} | {t['trade']:<18} | {t['status']}")

    print(f"\n=== Client Stats ===")
    print(json.dumps(client.get_stats(), indent=2))

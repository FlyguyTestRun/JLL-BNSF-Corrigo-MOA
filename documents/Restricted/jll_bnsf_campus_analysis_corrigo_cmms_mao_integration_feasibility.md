# JLL × BNSF Campus Analysis
## Corrigo CMMS Deep Dive & Multi-Agent Orchestration Integration Feasibility

**Location:** 2400 Lou Menk Dr, Fort Worth, TX 76131  
**Prepared by:** CoreSkills  
**Date:** February 2026

---

## 1. JLL Corporate Overview

Jones Lang LaSalle Incorporated (**NYSE: JLL**) is a Fortune 500 global commercial real estate and investment management company headquartered in Chicago, IL. Founded over 200 years ago, JLL operates in **80+ countries** with **112,000+ employees** and generates approximately **$23.4B in annual revenue**.

### 1.1 Global CEO & Executive Leadership

The CEO of JLL is **Christian Ulbrich**. Christian has served as Global CEO and President since 2016 and sits on the World Economic Forum board. He has been driving JLL’s transformation into what he describes as *“a technology company servicing the real estate sector.”*

#### JLL Global Executive Board (as of Feb 2026)

| Name | Title | Focus |
|---|---|---|
| Christian Ulbrich | Global CEO & President | Strategic direction, global operations |
| Kelly Howe | Chief Financial Officer | Finance (effective July 2025) |
| Neil Murray | CEO, Work Dynamics | Facilities management, technology services |
| Sanjay Rishi | CEO, Work Dynamics Americas & Head of Industries | Americas FM operations |
| John Gates | CEO, Americas Markets | Regional market strategy |
| Mihir Shah | CEO, JLL Technologies | Corrigo, JLL Spark, PropTech |
| Karen Brennan | CEO, Leasing Advisory (Global) | Leasing (effective July 2025) |
| Richard Bloxam | CEO, Capital Markets | Investment sales, debt & equity |
| Mark Gabbay | CEO, LaSalle Investment Management | Investment management arm |

---

## 2. JLL Work Dynamics — Your Division

**Work Dynamics** is the JLL business line responsible for facilities, projects, and portfolio management. This is the division under which the BNSF campus is managed. Work Dynamics oversees **2+ billion square feet** globally and employs **60,000+ specialists**.

### 2.1 Work Dynamics Hierarchy (Corporate → BNSF Campus)

| Level | Name / Role | Scope |
|---|---|---|
| Global CEO | Neil Murray — CEO, Work Dynamics | Global WD operations |
| Americas CEO | Sanjay Rishi — CEO, Work Dynamics Americas | Americas FM, projects, portfolios |
| Americas COO | Cheryl Carron — Americas COO, Work Dynamics | Operational execution |
| WPM Americas Lead | Michael Thompson | Workplace Management Americas |
| WPM Tech Lead | Tim Bernardez — Global Head, WPM Technologies | Corrigo, FM SaaS, JLL Marketplace |
| BNSF Account VP | Jill Wilbanks — VP, JLL | BNSF real estate & economic development |
| BNSF Facilities Manager | Tony Vita | Campus FM operations |
| Maintenance Manager | Juan Guerra | Maintenance crew supervision |
| Maintenance Technician | You | On-the-ground execution |

### 2.2 Workplace Management (WPM) Sub-Business Line

In early 2025, JLL consolidated facilities management, technical services, sustainability, and workplace experience into **Workplace Management (WPM)**, led by **Paul Morgan**. WPM explicitly owns **Corrigo** within JLL.

**Key WPM leaders relevant to this proposal:**
- Paul Morgan — Global Head, WPM
- Christian Whitaker — Global Head, Technical Services & Sustainable Operations
- Tim Bernardez — Global Head, WPM Technologies
- Gabriela Stephenson — Head, WPM Transformation
- Michael Thompson — WPM Americas Lead

---

## 3. BNSF Campus Service Hierarchy

The BNSF headquarters campus at **2400 / 2650 Lou Menk Dr, Fort Worth, TX 76131** is owned by **BNSF Railway**, a Berkshire Hathaway subsidiary. JLL operates as the primary outsourced facilities management partner.

### 3.1 Client Side: BNSF Railway

- BNSF Railway Company — Asset Owner
- BNSF Corporate Real Estate — JLL relationship management
- Campus occupants — End users submitting work orders via Corrigo

### 3.2 Primary FM Partner: JLL

JLL operates under an **Integrated Facilities Management (IFM)** contract, responsible for:
- Work order management via Corrigo (**jll-bnsf.corrigo.com**)
- Mechanical, electrical, and plumbing maintenance
- Engineering and critical systems
- Janitorial services
- Grounds and landscaping
- Brokerage, land sales, and leasing (2650 Lou Menk Dr)
- Economic development project management
- Energy management and sustainability

### 3.3 Subcontractors & Service Providers

JLL coordinates specialty vendors via the **CorrigoPro Network** (60,000+ providers across 130+ trades), including:
- HVAC (Johnson Controls, Trane, Carrier)
- Elevators (Otis, Schindler, KONE)
- Fire/Life Safety (Simplex Grinnell, Siemens)
- Janitorial (ABM, C&W Services)
- Landscaping (BrightView, Yellowstone)
- Pest Control (Terminix, Orkin)
- Security, roofing, paving, electrical, generators, plumbing

> *Exact subcontractors for the BNSF campus must be verified in the Corrigo provider directory.*

---

## 4. Corrigo CMMS — Deep Analysis

### 4.1 What Corrigo Is

Corrigo is a cloud-based **CMMS** owned by **JLL Technologies (JLLT)** and led by CEO **Mihir Shah**. It is JLL’s flagship FM platform, deployed internally and sold externally as a SaaS product.

### 4.2 Scale & Market Position

| Metric | Data |
|---|---|
| Facilities deployed | 1.1M+ across 140+ countries |
| Users | 7M+ |
| Annual work orders | 18.5M+ |
| Spend managed | $6B+ annually |
| Provider network | 60,000+ providers |
| Platform uptime | 99.98% |
| Mobile apps | iOS & Android (offline capable) |

### 4.3 How Corrigo Works

**Core Modules**
- Work Order Management
- Asset Management
- Vendor & Provider Management (CorrigoPro)
- Business Intelligence & Dashboards
- Mobility (GPS, offline mode)
- Procurement (JLL Marketplace)

**Work Order Lifecycle**
1. Request submission
2. Auto-triage
3. Dispatch
4. Technician check-in
5. Work execution & documentation
6. Completion & comments
7. Quality verification
8. Invoice reconciliation
9. BI reporting

### 4.4 Corrigo API Architecture (Critical)

Corrigo exposes two robust API ecosystems:

| API | Purpose | Docs |
|---|---|---|
| Corrigo Enterprise REST API | JLL-managed accounts | developer.corrigo.com |
| CorrigoPro Direct API | Vendor integrations | developer.corrigopro.com |

*Shaw Question 1? Why the Enterprise REST API and this Corrigo Direct API? What are the vendor vs. JLL-managed accounts differences? Why did they chose or structure it this way?*

**Technical Highlights**
- RESTful architecture (legacy SOAP supported)
- OAuth 2.0 authentication
- Swagger / OpenAPI specs
- Webhooks (event-driven)
- Regional endpoints (US, APAC, EMEA)
- Sandbox environments
- iPaaS support (Celigo)

*Shaw Question 2, How did they structure the iPaas Support w/Celigo?*

---

## 5. Multi-Agent Orchestration (MAO) Integration Opportunity

### 5.1 MAO Touchpoints in Corrigo

| Agent | Function | Impact |
|---|---|---|
| Triage Agent | NLP-based classification | Faster routing |
| Dispatch Optimizer | Smart assignment | Lower costs |
| PM Compliance Agent | PM automation | Higher compliance |
| Escalation Agent | SLA monitoring | Fewer breaches |
| Reporting Agent | Automated summaries | Time savings |
| Quality Agent | Asset analytics | Better ROI |
| Orchestrator | Agent coordination | Unified visibility |

*Shaw Goal one with integration oppertunities. This app is used nationally, Specified agent to understand routine task completion of PMs (Preventative Maintencance), using industrial engineering practices to complete PMs in a more systematic way.*

*Morning Identification and TODO lists for task priority and mapping based on the flow and workflow of the BNSF campus. Maps that interlace and track best patterns to completion of task.*

### 5.2 Metric Improvements

MAO directly improves:
- Work order cycle time
- First-time fix rate
- SLA compliance
- PM completion
- Cost per work order
- Technician utilization
- Client satisfaction (CSAT)

---

## 6. Strategic Advice: Who to Pitch & How

### 6.1 Reality Check

JLL owns Corrigo and maintains internal AI and technology teams. Any pitch must demonstrate **speed-to-value**, **campus specificity**, or capabilities that are not easily prioritized internally.

### 6.2 Approach

**Phase 1 (Weeks 1–2):**
- Build a sandbox prototype
- Focus on a single high-impact agent
- Validate with Juan Guerra

**Phase 2 (Weeks 2-6):**
- Present pilot results to Tony Vita
- Propose a 30-day campus trial

**Phase 3:**
- Escalate via account leadership (Jill Wilbanks)
- Engage WPM Americas or WPM Technologies

### 6.3 Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Conflict of interest | Review employment agreement, transparency |
| Internal competition | Emphasize speed & customization |
| Data security | Admin approval, sandbox first |
| Budget authority | Free/low-cost pilot |

### 6.4 Bottom Line Recommendation

A CoreSkills MAO proof-of-concept **technically viablity** and **strategy**, but success depends on careful navigation of internal governance, employment agreements, and stakeholder alignment. The strongest path forward is to **build first, prove locally, and formalize later**.



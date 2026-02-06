import { useState, useRef, useCallback, useEffect } from "react";

const NODES = [
  // Input Sources (Layer 0)
  { id: "src_portal", label: "Corrigo Portal\n(jll-bnsf.corrigo.com)", layer: 0, col: 0, type: "source", desc: "BNSF occupants submit requests via web portal" },
  { id: "src_mobile", label: "Corrigo Mobile\nApp", layer: 0, col: 1, type: "source", desc: "Technicians & requestors submit via iOS/Android" },
  { id: "src_email", label: "Email / Phone\nRequests", layer: 0, col: 2, type: "source", desc: "Legacy requests routed through dispatch" },
  { id: "src_iot", label: "IoT Sensors\n& BAS Alerts", layer: 0, col: 3, type: "source", desc: "Building automation system triggers (HVAC, electrical, fire)" },
  { id: "src_pm", label: "PM Schedules\n(Corrigo)", layer: 0, col: 4, type: "source", desc: "Preventive maintenance calendar auto-generates WOs" },

  // MAO Orchestrator (Layer 1)
  { id: "orchestrator", label: "MAO\nORCHESTRATOR", layer: 1, col: 2, type: "orchestrator", desc: "Central coordinator — manages agent state, routes tasks, handles inter-agent communication via Corrigo REST API + Webhooks" },

  // Agent Layer (Layer 2)
  { id: "agent_triage", label: "Triage\nAgent", layer: 2, col: 0, type: "agent", desc: "NLP classification of incoming WOs by trade, priority, SLA tier. Reduces misroutes and speeds first-response." },
  { id: "agent_dispatch", label: "Dispatch\nOptimizer", layer: 2, col: 1, type: "agent", desc: "Intelligent assignment: internal tech vs vendor, skill match, proximity, backlog balance, cost optimization." },
  { id: "agent_escalation", label: "SLA Escalation\nAgent", layer: 2, col: 2, type: "agent", desc: "Monitors SLA clocks via webhooks, auto-escalates BEFORE breach, proactive manager alerts." },
  { id: "agent_pm", label: "PM Compliance\nAgent", layer: 2, col: 3, type: "agent", desc: "Tracks PM schedule adherence, auto-generates overdue alerts, pre-fills recurring WOs, flags gaps." },
  { id: "agent_quality", label: "Quality &\nAnalytics Agent", layer: 2, col: 4, type: "agent", desc: "Pattern detection: repeat failures, chronic assets, vendor performance scoring, repair vs replace recommendations." },

  // Action Layer (Layer 3)
  { id: "act_wo", label: "Work Order\nCRUD", layer: 3, col: 0, type: "action", desc: "Create, update, close WOs via Corrigo Enterprise REST API" },
  { id: "act_vendor", label: "Vendor\nDispatch", layer: 3, col: 1, type: "action", desc: "Route to CorrigoPro Network (60K+ service pros, 130+ trades)" },
  { id: "act_notify", label: "Notifications\n& Alerts", layer: 3, col: 2, type: "action", desc: "Push to Juan, Tony, crew leads — email, SMS, in-app" },
  { id: "act_report", label: "Auto\nReporting", layer: 3, col: 3, type: "action", desc: "Daily/weekly metric dashboards pushed to management" },
  { id: "act_asset", label: "Asset\nInsights", layer: 3, col: 4, type: "action", desc: "Repair vs replace recommendations, lifecycle tracking, capital planning" },

  // Output / Metrics (Layer 4)
  { id: "out_cycle", label: "↓ Cycle Time", layer: 4, col: 0, type: "metric", desc: "Reduce avg WO completion time" },
  { id: "out_sla", label: "↑ SLA Compliance", layer: 4, col: 1, type: "metric", desc: "Proactive escalation prevents breaches" },
  { id: "out_cost", label: "↓ Cost/WO", layer: 4, col: 2, type: "metric", desc: "Optimized tech vs vendor routing" },
  { id: "out_pm", label: "↑ PM Rate", layer: 4, col: 3, type: "metric", desc: "Automated scheduling closes compliance gaps" },
  { id: "out_csat", label: "↑ CSAT Score", layer: 4, col: 4, type: "metric", desc: "Faster response + fewer repeats = happy occupants" },
];

const EDGES = [
  // Sources → Orchestrator
  { from: "src_portal", to: "orchestrator" },
  { from: "src_mobile", to: "orchestrator" },
  { from: "src_email", to: "orchestrator" },
  { from: "src_iot", to: "orchestrator" },
  { from: "src_pm", to: "orchestrator" },
  // Orchestrator → Agents
  { from: "orchestrator", to: "agent_triage" },
  { from: "orchestrator", to: "agent_dispatch" },
  { from: "orchestrator", to: "agent_escalation" },
  { from: "orchestrator", to: "agent_pm" },
  { from: "orchestrator", to: "agent_quality" },
  // Agents → Actions
  { from: "agent_triage", to: "act_wo" },
  { from: "agent_triage", to: "act_vendor" },
  { from: "agent_dispatch", to: "act_wo" },
  { from: "agent_dispatch", to: "act_vendor" },
  { from: "agent_escalation", to: "act_notify" },
  { from: "agent_pm", to: "act_wo" },
  { from: "agent_pm", to: "act_notify" },
  { from: "agent_quality", to: "act_report" },
  { from: "agent_quality", to: "act_asset" },
  // Actions → Metrics
  { from: "act_wo", to: "out_cycle" },
  { from: "act_vendor", to: "out_cost" },
  { from: "act_notify", to: "out_sla" },
  { from: "act_report", to: "out_csat" },
  { from: "act_wo", to: "out_pm" },
  { from: "act_asset", to: "out_cost" },
];

const LAYER_Y = [40, 160, 300, 440, 560];
const COL_X = [60, 250, 440, 630, 820];
const NODE_W = 130;
const NODE_H = 52;

const TYPE_STYLES = {
  source: { fill: "#1a2744", stroke: "#3b82f6", text: "#93c5fd", radius: 8 },
  orchestrator: { fill: "#7c2d12", stroke: "#f97316", text: "#fed7aa", radius: 14 },
  agent: { fill: "#1e3a2f", stroke: "#22c55e", text: "#bbf7d0", radius: 10 },
  action: { fill: "#3b1764", stroke: "#a855f7", text: "#e9d5ff", radius: 8 },
  metric: { fill: "#1e3a4f", stroke: "#06b6d4", text: "#a5f3fc", radius: 8 },
};

const LAYER_LABELS = ["INPUT SOURCES", "ORCHESTRATION", "AI AGENTS", "ACTIONS (Corrigo API)", "METRIC OUTCOMES"];

function getNodeCenter(node) {
  return { x: COL_X[node.col] + NODE_W / 2, y: LAYER_Y[node.layer] + NODE_H / 2 };
}

const SUMMARY_DATA = {
  company: {
    title: "JLL Corporate Summary",
    content: `Jones Lang LaSalle (NYSE: JLL) is a Fortune 500 global commercial real estate and investment management company. $23.4B annual revenue, 112,000+ employees, 80+ countries. CEO: Christian Ulbrich (since 2016).

Key business lines: Work Dynamics (facilities management — YOUR division), Capital Markets, Leasing Advisory, JLL Technologies (owns Corrigo), and LaSalle Investment Management.

JLL's stated strategic priority is becoming "a technology company servicing the real estate sector." Their AI tools already contributed to 30% of capital market deals in Q1 2025.`
  },
  workDynamics: {
    title: "Work Dynamics (Your Division)",
    content: `Work Dynamics manages 2B+ sq ft globally with 60,000+ specialists. In 2025, JLL reorganized FM into "Workplace Management" (WPM) under Paul Morgan, unifying facilities management, technical services, energy/sustainability, and experience.

Key leaders: Neil Murray (Global CEO), Sanjay Rishi (Americas CEO), Cheryl Carron (Americas COO), Michael Thompson (Americas WPM Lead), Tim Bernardez (WPM Technologies — owns Corrigo), Christian Whitaker (Technical Services).

Your BNSF account sits under Work Dynamics Americas → WPM → BNSF Account.`
  },
  corrigo: {
    title: "Corrigo Platform",
    content: `Corrigo is JLL's flagship CMMS — they OWN it (not a third-party license). 1.1M+ facilities deployed, 7M+ users, 18.5M work orders/year, $6B in managed spend, 99.98% uptime.

Critical for your MAO pitch: Corrigo has a full REST API (developer.corrigo.com) with OAuth 2.0, webhooks, and sandbox environments. Also has CorrigoPro Direct API for vendor integration. Pre-built connectors exist for Sage Intacct, Avalara, Infogrid, HqO, and others.

Your campus portal: jll-bnsf.corrigo.com (confirmed active). The woman in your maintenance building likely manages dispatch through this portal.`
  },
  bnsf: {
    title: "BNSF Campus Chain of Command",
    content: `Christian Ulbrich (Global CEO) → Neil Murray (CEO Work Dynamics) → Sanjay Rishi (CEO WD Americas) → Cheryl Carron (COO WD Americas) → Michael Thompson (WPM Americas) → Jill Wilbanks (VP, BNSF Account) → Tony Vita (Facilities Manager, BNSF) → Juan Guerra (Maintenance Manager) → YOU

Corrigo tech ownership: Mihir Shah (CEO JLL Technologies) → Tim Bernardez (Global Head WPM Tech) → Scott Boekweg (Product Leader, Corrigo — DFW based)

Tony Vita's known colleagues on account: George Saliba, Larry Drummonds, Jason Smalls, Kerry Sovanski (Area Maintenance Mgr).`
  },
  recommendation: {
    title: "Strategic Recommendation",
    content: `APPROACH: Bottom-up, not top-down. Do NOT pitch corporate leadership directly.

1. Talk to the Corrigo dispatch person in your maintenance building — learn the pain points, who the admin is, what's automated vs manual.

2. Build a working prototype against the Corrigo sandbox API — start with ONE agent (SLA Escalation or PM Compliance).

3. Show Juan informally. Get his buy-in, then ask for intro to Tony Vita.

4. Offer a free 30-day pilot at the campus level. Let results speak.

CRITICAL RISK: Check your JLL employment agreement for moonlighting/non-compete/IP clauses before proposing CoreSkills as a vendor. You work for JLL — pitching them to hire your side company has real ethical and legal implications.`
  }
};

export default function CorrigoMAODag() {
  const [selected, setSelected] = useState(null);
  const [activeTab, setActiveTab] = useState("dag");
  const [summaryTab, setSummaryTab] = useState("company");
  const svgRef = useRef(null);
  const [hoveredEdge, setHoveredEdge] = useState(null);
  const [animKey, setAnimKey] = useState(0);

  useEffect(() => { setAnimKey(k => k + 1); }, [activeTab]);

  const nodeMap = {};
  NODES.forEach(n => { nodeMap[n.id] = n; });

  const selectedNode = selected ? nodeMap[selected] : null;

  const renderEdges = useCallback(() => {
    return EDGES.map((e, i) => {
      const from = nodeMap[e.from];
      const to = nodeMap[e.to];
      const fc = getNodeCenter(from);
      const tc = getNodeCenter(to);
      const isHighlight = selected && (e.from === selected || e.to === selected);
      const isHovered = hoveredEdge === i;
      const midY = (fc.y + tc.y) / 2;
      const path = `M${fc.x},${fc.y + NODE_H / 2} C${fc.x},${midY} ${tc.x},${midY} ${tc.x},${tc.y - NODE_H / 2}`;
      return (
        <path
          key={i}
          d={path}
          fill="none"
          stroke={isHighlight ? "#f59e0b" : isHovered ? "#94a3b8" : "#334155"}
          strokeWidth={isHighlight ? 2.5 : 1.2}
          opacity={selected && !isHighlight ? 0.15 : isHighlight ? 1 : 0.5}
          onMouseEnter={() => setHoveredEdge(i)}
          onMouseLeave={() => setHoveredEdge(null)}
          style={{ transition: "all 0.3s ease", cursor: "pointer" }}
          strokeDasharray={isHighlight ? "none" : "4 3"}
        />
      );
    });
  }, [selected, hoveredEdge, nodeMap]);

  const renderNodes = useCallback(() => {
    return NODES.map((n, i) => {
      const x = COL_X[n.col];
      const y = LAYER_Y[n.layer];
      const s = TYPE_STYLES[n.type];
      const isSel = selected === n.id;
      const isConnected = selected && EDGES.some(e => (e.from === selected && e.to === n.id) || (e.to === selected && e.from === n.id));
      const dimmed = selected && !isSel && !isConnected;
      const lines = n.label.split("\n");
      return (
        <g
          key={n.id}
          onClick={() => setSelected(isSel ? null : n.id)}
          style={{ cursor: "pointer", transition: "opacity 0.3s" }}
          opacity={dimmed ? 0.2 : 1}
        >
          <rect
            x={x} y={y} width={NODE_W} height={NODE_H} rx={s.radius}
            fill={isSel ? s.stroke : s.fill}
            stroke={s.stroke}
            strokeWidth={isSel ? 3 : 1.5}
            style={{ filter: isSel ? `drop-shadow(0 0 8px ${s.stroke})` : "none", transition: "all 0.2s ease" }}
          />
          {lines.map((line, li) => (
            <text
              key={li}
              x={x + NODE_W / 2} y={y + (lines.length === 1 ? 30 : 20 + li * 16)}
              textAnchor="middle"
              fill={isSel ? (n.type === "orchestrator" ? "#fff" : "#111") : s.text}
              fontSize={n.type === "orchestrator" ? 11 : 10}
              fontWeight={n.type === "orchestrator" ? 800 : 600}
              fontFamily="'JetBrains Mono', 'SF Mono', monospace"
              style={{ pointerEvents: "none" }}
            >
              {line}
            </text>
          ))}
        </g>
      );
    });
  }, [selected]);

  const renderLayerLabels = useCallback(() => {
    return LAYER_LABELS.map((label, i) => (
      <text
        key={i}
        x={12}
        y={LAYER_Y[i] + NODE_H / 2 + 4}
        fill="#475569"
        fontSize={8}
        fontWeight={700}
        fontFamily="'JetBrains Mono', monospace"
        textAnchor="start"
        transform={`rotate(-90, 12, ${LAYER_Y[i] + NODE_H / 2})`}
        style={{ textTransform: "uppercase", letterSpacing: "1.5px" }}
      >
        {label}
      </text>
    ));
  }, []);

  return (
    <div style={{
      background: "#0a0f1a",
      minHeight: "100vh",
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      color: "#e2e8f0"
    }}>
      {/* Header */}
      <div style={{
        padding: "20px 24px 0",
        borderBottom: "1px solid #1e293b"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
          <div>
            <h1 style={{
              fontSize: 20, fontWeight: 800, margin: 0, letterSpacing: "-0.5px",
              background: "linear-gradient(135deg, #f97316, #f59e0b)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
            }}>
              CORRIGO × MAO WORKFLOW DAG
            </h1>
            <p style={{ fontSize: 11, color: "#64748b", margin: "4px 0 0" }}>
              CoreSkills Multi-Agent Orchestration — BNSF Campus Integration Blueprint
            </p>
          </div>
          <div style={{ display: "flex", gap: 2, background: "#111827", borderRadius: 8, padding: 3 }}>
            {[{ key: "dag", label: "DAG" }, { key: "summary", label: "INTEL" }].map(t => (
              <button
                key={t.key}
                onClick={() => { setActiveTab(t.key); setSelected(null); }}
                style={{
                  padding: "6px 16px", fontSize: 10, fontWeight: 700,
                  fontFamily: "inherit", letterSpacing: "1px",
                  border: "none", borderRadius: 6, cursor: "pointer",
                  background: activeTab === t.key ? "#f97316" : "transparent",
                  color: activeTab === t.key ? "#0a0f1a" : "#64748b",
                  transition: "all 0.2s"
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {activeTab === "dag" ? (
        <div style={{ display: "flex", height: "calc(100vh - 90px)" }}>
          {/* SVG Canvas */}
          <div style={{ flex: 1, overflow: "auto", padding: "16px 8px" }}>
            <svg
              ref={svgRef}
              viewBox="0 0 1000 630"
              style={{ width: "100%", maxHeight: "100%" }}
              onClick={(e) => { if (e.target === e.currentTarget || e.target.tagName === 'svg') setSelected(null); }}
            >
              {/* Grid */}
              {LAYER_Y.map((y, i) => (
                <line key={i} x1={30} x2={970} y1={y + NODE_H + 10} y2={y + NODE_H + 10}
                  stroke="#1e293b" strokeWidth={0.5} strokeDasharray="2 6" />
              ))}
              {renderLayerLabels()}
              {renderEdges()}
              {renderNodes()}
              {/* Legend */}
              <g transform="translate(30, 605)">
                {Object.entries(TYPE_STYLES).map(([type, s], i) => (
                  <g key={type} transform={`translate(${i * 180}, 0)`}>
                    <rect x={0} y={-6} width={12} height={12} rx={3} fill={s.fill} stroke={s.stroke} strokeWidth={1.5} />
                    <text x={18} y={4} fill="#64748b" fontSize={9} fontFamily="inherit" style={{ textTransform: "capitalize" }}>
                      {type === "orchestrator" ? "MAO Core" : type}
                    </text>
                  </g>
                ))}
              </g>
            </svg>
          </div>

          {/* Detail Panel */}
          <div style={{
            width: 280, borderLeft: "1px solid #1e293b", padding: "16px",
            background: "#0d1321", overflowY: "auto"
          }}>
            {selectedNode ? (
              <div key={animKey + selected}>
                <div style={{
                  fontSize: 8, fontWeight: 700, letterSpacing: "1.5px", color: TYPE_STYLES[selectedNode.type].stroke,
                  textTransform: "uppercase", marginBottom: 6
                }}>
                  {selectedNode.type}
                </div>
                <h3 style={{ fontSize: 14, fontWeight: 700, margin: "0 0 12px", color: "#f1f5f9" }}>
                  {selectedNode.label.replace("\n", " ")}
                </h3>
                <p style={{ fontSize: 12, lineHeight: 1.6, color: "#94a3b8", margin: 0 }}>
                  {selectedNode.desc}
                </p>
                <div style={{ marginTop: 16, borderTop: "1px solid #1e293b", paddingTop: 12 }}>
                  <div style={{ fontSize: 9, fontWeight: 700, color: "#475569", marginBottom: 8, letterSpacing: "1px" }}>CONNECTIONS</div>
                  {EDGES.filter(e => e.from === selected || e.to === selected).map((e, i) => {
                    const other = e.from === selected ? nodeMap[e.to] : nodeMap[e.from];
                    const dir = e.from === selected ? "→" : "←";
                    return (
                      <div
                        key={i}
                        onClick={() => setSelected(other.id)}
                        style={{
                          padding: "6px 8px", marginBottom: 4, borderRadius: 6,
                          background: "#111827", cursor: "pointer", fontSize: 10,
                          color: TYPE_STYLES[other.type].text,
                          border: `1px solid ${TYPE_STYLES[other.type].stroke}33`
                        }}
                      >
                        {dir} {other.label.replace("\n", " ")}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div style={{ color: "#475569", fontSize: 11, lineHeight: 1.8 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", marginBottom: 12 }}>
                  Click any node
                </div>
                <p>Select a node in the DAG to see its details, description, and connections.</p>
                <p style={{ marginTop: 16, fontSize: 10, color: "#334155" }}>
                  This DAG represents the recommended automation workflow for integrating a Multi-Agent Orchestration system with the Corrigo CMMS at the JLL-managed BNSF campus.
                </p>
                <div style={{
                  marginTop: 20, padding: 12, borderRadius: 8,
                  border: "1px solid #f9731633", background: "#f9731608"
                }}>
                  <div style={{ fontSize: 9, fontWeight: 700, color: "#f97316", letterSpacing: "1px", marginBottom: 6 }}>API INTEGRATION POINT</div>
                  <p style={{ fontSize: 10, color: "#94a3b8", margin: 0, lineHeight: 1.6 }}>
                    All agents connect via Corrigo Enterprise REST API (OAuth 2.0) + Webhooks for real-time event streaming.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* INTEL TAB */
        <div style={{ padding: "16px 24px", height: "calc(100vh - 90px)", display: "flex", gap: 16 }}>
          {/* Sidebar Nav */}
          <div style={{ width: 200, flexShrink: 0 }}>
            {Object.entries(SUMMARY_DATA).map(([key, data]) => (
              <button
                key={key}
                onClick={() => setSummaryTab(key)}
                style={{
                  display: "block", width: "100%", textAlign: "left",
                  padding: "10px 12px", marginBottom: 4, borderRadius: 8,
                  border: summaryTab === key ? "1px solid #f97316" : "1px solid transparent",
                  background: summaryTab === key ? "#f9731612" : "#111827",
                  color: summaryTab === key ? "#f97316" : "#94a3b8",
                  fontSize: 11, fontWeight: 600, fontFamily: "inherit",
                  cursor: "pointer", transition: "all 0.2s"
                }}
              >
                {data.title}
              </button>
            ))}
          </div>
          {/* Content */}
          <div style={{
            flex: 1, background: "#0d1321", borderRadius: 12,
            border: "1px solid #1e293b", padding: "24px", overflowY: "auto"
          }}>
            <h2 style={{
              fontSize: 18, fontWeight: 800, margin: "0 0 4px",
              color: "#f97316"
            }}>
              {SUMMARY_DATA[summaryTab].title}
            </h2>
            <div style={{
              width: 40, height: 3, background: "#f97316", borderRadius: 2, marginBottom: 20
            }} />
            {SUMMARY_DATA[summaryTab].content.split("\n\n").map((para, i) => (
              <p key={i} style={{
                fontSize: 13, lineHeight: 1.8, color: "#cbd5e1",
                margin: "0 0 16px", maxWidth: 700
              }}>
                {para}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

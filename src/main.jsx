import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  BadgeCheck,
  BrainCircuit,
  Camera,
  CheckCircle2,
  Factory,
  FileWarning,
  Flame,
  Gauge,
  GitBranch,
  HardHat,
  LayoutDashboard,
  LockKeyhole,
  MapPinned,
  Play,
  Radar,
  ShieldAlert,
  Upload,
  Waves,
  X
} from "lucide-react";
import "./styles.css";

const riskClasses = [
  { label: "Low", limit: 29, color: "green" },
  { label: "Medium", limit: 59, color: "yellow" },
  { label: "High", limit: 84, color: "orange" },
  { label: "Critical", limit: 100, color: "red" }
];

const baseEvents = [
  {
    id: 1,
    title: "Worker entered restricted zone",
    source: "CCTV-2",
    zone: "Zone B",
    severity: "High",
    points: 30,
    what: "A worker crossed the configured geofence around the pump maintenance area.",
    why: "Restricted areas usually contain moving equipment, hot work, or process hazards. Entry during abnormal plant conditions raises injury probability.",
    factors: ["Restricted zone entry", "Active maintenance permit", "Pump maintenance status"],
    action: "Notify area supervisor, clear the zone, and preserve CCTV evidence."
  },
  {
    id: 2,
    title: "PPE compliance gap",
    source: "PPE AI",
    zone: "Zone A",
    severity: "Medium",
    points: 20,
    what: "Helmet or vest compliance is below the configured threshold.",
    why: "PPE deviation increases injury severity if a worker is exposed to falling objects, moving equipment, or emergency evacuation.",
    factors: ["PPE violation", "Worker detected", "Checklist incomplete"],
    action: "Recheck PPE before re-entry and record supervisor acknowledgement."
  },
  {
    id: 3,
    title: "Gas level rising near permit area",
    source: "Gas Sensor",
    zone: "Reactor Zone",
    severity: "Critical",
    points: 40,
    what: "CH4 and CO readings are elevated near a work-permit zone.",
    why: "Gas accumulation combined with work activity can rapidly escalate into fire, toxicity, or evacuation scenarios.",
    factors: ["CH4 above normal", "Permit overlap", "Shift handover approaching"],
    action: "Increase ventilation, pause permit activity, and verify readings with the safety officer."
  }
];

const sampleVideos = [
  "Factory floor - maintenance area",
  "Warehouse aisle - forklift crossing",
  "Reactor bay - restricted access"
];

function classifyRisk(score) {
  return riskClasses.find((item) => score <= item.limit) ?? riskClasses.at(-1);
}

function calculateRisk(inputs, monitoring) {
  let score = 0;
  if (inputs.ppeViolation) score += 20;
  if (inputs.zoneEntry) score += 30;
  if (inputs.gasLevel >= 70) score += 40;
  else if (inputs.gasLevel >= 40) score += 20;
  if (inputs.activePermit) score += 15;
  if (inputs.equipment === "Maintenance") score += 12;
  if (inputs.equipment === "Fault") score += 25;
  if (inputs.checklist < 70) score += 18;
  if (!monitoring) score = Math.min(score, 45);
  return Math.min(100, score);
}

function LandingPage({ onDemo }) {
  return (
    <main className="landing">
      <section className="hero-grid">
        <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} className="hero-copy">
          <span className="eyebrow">Industrial AI Safety Platform</span>
          <h1>SafeVision AI</h1>
          <p className="tagline">Context-aware industrial safety intelligence for zero-harm operations</p>
          <p className="hero-text">
            Industrial plants already have CCTV, gas sensors, permits, equipment logs and compliance
            checklists. The problem is that these systems often work separately. SafeVision AI brings
            those signals together so safety teams can detect compound risk before it becomes an incident.
          </p>
          <button className="primary-btn" onClick={onDemo}>
            <LayoutDashboard size={18} />
            View Demo Dashboard
          </button>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} className="hero-visual">
          <div className="live-card">
            <div className="live-top">
              <span className="live-dot" />
              LIVE CCTV INTELLIGENCE
              <span className="risk-pill red">Critical</span>
            </div>
            <div className="camera-frame">
              <div className="worker-box worker-main">worker</div>
              <div className="zone-box">RESTRICTED ZONE</div>
              <div className="sensor-chip">CH4 14% LEL</div>
              <div className="ppe-chip">PPE gap</div>
            </div>
          </div>
        </motion.div>
      </section>

      <section className="section">
        <div className="section-head">
          <span className="eyebrow">Problem</span>
          <h2>Safety signals are missed when plant systems operate in silos.</h2>
        </div>
        <div className="problem-grid">
          {[
            ["CCTV", "Detects people but often misses permit, gas or maintenance context.", Camera],
            ["Gas Sensors", "Raise threshold alerts but do not know where workers are.", Gauge],
            ["Permits", "Approve work but may not react to real-time plant risk.", FileWarning],
            ["Compliance", "Records checklist status but rarely connects to live events.", BadgeCheck]
          ].map(([title, text, Icon]) => (
            <div className="light-card" key={title}>
              <Icon />
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section solution-section">
        <div className="section-head">
          <span className="eyebrow">Solution</span>
          <h2>One fused view for vision, plant context and response.</h2>
        </div>
        <div className="solution-grid">
          {[
            "CCTV/PPE detection",
            "Restricted zone monitoring",
            "Gas sensor readings",
            "Work permit status",
            "Equipment status",
            "Shift handover notes",
            "Compliance checklist"
          ].map((item) => (
            <div className="solution-item" key={item}>
              <CheckCircle2 size={18} />
              {item}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

function ArchitectureModal({ onClose }) {
  const modules = [
    ["Multi-camera CCTV", "Plant video sources and uploaded sample feeds."],
    ["Vision Processing", "Video decoding, frame sampling and stream processing."],
    ["Computer Vision Engine", "Worker, PPE and restricted-zone detection."],
    ["Safety Fusion Engine", "Combines CCTV events with gas, permit, equipment and shift context."],
    ["Safety Rule Engine", "Applies rules for gas + zone entry, PPE + permit and repeated violations."],
    ["Risk Engine", "Calculates a 0-100 risk score and classifies severity."],
    ["AI Safety Advisor", "Explains risk and recommends preventive actions."],
    ["Dashboard + Heatmap + Reports", "Presents operator-ready intelligence and evidence."]
  ];

  return (
    <motion.div className="modal-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <motion.div className="modal" initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 40, opacity: 0 }}>
        <button className="icon-btn close" onClick={onClose} aria-label="Close architecture modal">
          <X size={18} />
        </button>
        <h2>SafeVision AI Pipeline</h2>
        <p>Multi-camera Industrial Safety Intelligence Platform</p>
        <div className="pipeline">
          {modules.map(([title, text], index) => (
            <div className="pipeline-row" key={title}>
              <div className="pipeline-index">{index + 1}</div>
              <div>
                <h3>{title}</h3>
                <p>{text}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="core-principle">
          <b>Core Principle:</b> Vision Intelligence + Operational Context + Safety Rules =
          Real-Time Risk Intelligence.
        </div>
      </motion.div>
    </motion.div>
  );
}

function DashboardPage({ onHome }) {
  const [monitoring, setMonitoring] = useState(false);
  const [architectureOpen, setArchitectureOpen] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState(baseEvents[0]);
  const [uploadedName, setUploadedName] = useState("");
  const [sampleVideo, setSampleVideo] = useState(sampleVideos[0]);
  const [inputs, setInputs] = useState({
    ppeViolation: true,
    zoneEntry: true,
    gasLevel: 76,
    activePermit: true,
    equipment: "Maintenance",
    checklist: 62
  });

  const score = useMemo(() => calculateRisk(inputs, monitoring), [inputs, monitoring]);
  const risk = classifyRisk(score);
  const activeEvents = useMemo(() => {
    if (!monitoring) return baseEvents.slice(0, 1);
    return baseEvents.filter((event) => {
      if (event.title.includes("PPE")) return inputs.ppeViolation;
      if (event.title.includes("restricted")) return inputs.zoneEntry;
      if (event.title.includes("Gas")) return inputs.gasLevel >= 40;
      return true;
    });
  }, [inputs, monitoring]);

  const updateInput = (key, value) => setInputs((current) => ({ ...current, [key]: value }));

  return (
    <main className="dashboard">
      <nav className="topbar">
        <button className="ghost-btn" onClick={onHome}>SafeVision AI</button>
        <div className="topbar-actions">
          <button className="ghost-btn" onClick={() => setArchitectureOpen(true)}>
            <GitBranch size={17} />
            Architecture
          </button>
          <button className="primary-btn small" onClick={() => setMonitoring((value) => !value)}>
            <Play size={16} />
            {monitoring ? "Monitoring Active" : "Start Monitoring"}
          </button>
        </div>
      </nav>

      <section className="dashboard-grid">
        <aside className="sidebar-panel">
          <h2>Plant Camera Manager</h2>
          <label className="upload-box">
            <Upload size={22} />
            <span>{uploadedName || "Upload CCTV video"}</span>
            <small>MP4, MOV, AVI, MKV</small>
            <input
              type="file"
              accept="video/*"
              onChange={(event) => setUploadedName(event.target.files?.[0]?.name ?? "")}
            />
          </label>
          <label className="form-label">Sample video</label>
          <select value={sampleVideo} onChange={(event) => setSampleVideo(event.target.value)}>
            {sampleVideos.map((video) => <option key={video}>{video}</option>)}
          </select>

          <div className="camera-list">
            {["CCTV-1 → Zone A", "CCTV-2 → Zone B", "CCTV-3 → Reactor Zone"].map((camera, index) => (
              <button className={index === 0 ? "camera-card active" : "camera-card"} key={camera}>
                <Camera size={18} />
                <span>{camera}</span>
                <b>{monitoring ? "LIVE" : index === 2 ? "Pending" : "Ready"}</b>
              </button>
            ))}
          </div>

          <h2>Plant Signal Inputs</h2>
          <label className="toggle-row">
            <input type="checkbox" checked={inputs.ppeViolation} onChange={(e) => updateInput("ppeViolation", e.target.checked)} />
            PPE violation
          </label>
          <label className="toggle-row">
            <input type="checkbox" checked={inputs.zoneEntry} onChange={(e) => updateInput("zoneEntry", e.target.checked)} />
            Restricted zone entry
          </label>
          <label className="form-label">Gas level</label>
          <input type="range" min="0" max="100" value={inputs.gasLevel} onChange={(e) => updateInput("gasLevel", Number(e.target.value))} />
          <div className="range-value">CH4 risk index: {inputs.gasLevel}%</div>
          <label className="toggle-row">
            <input type="checkbox" checked={inputs.activePermit} onChange={(e) => updateInput("activePermit", e.target.checked)} />
            Active work permit
          </label>
          <label className="form-label">Equipment status</label>
          <select value={inputs.equipment} onChange={(event) => updateInput("equipment", event.target.value)}>
            <option>Normal</option>
            <option>Maintenance</option>
            <option>Fault</option>
          </select>
          <label className="form-label">Checklist completion</label>
          <input type="range" min="0" max="100" value={inputs.checklist} onChange={(e) => updateInput("checklist", Number(e.target.value))} />
          <div className="range-value">{inputs.checklist}% complete</div>
        </aside>

        <section className="workspace">
          <div className="dashboard-header">
            <div>
              <span className="eyebrow">Demo Dashboard</span>
              <h1>Industrial Safety Command Center</h1>
              <p>{sampleVideo} {uploadedName ? `· ${uploadedName}` : ""}</p>
            </div>
            <div className={`risk-meter ${risk.color}`}>
              <span>{risk.label}</span>
              <strong>{score}/100</strong>
            </div>
          </div>

          <div className="metric-grid">
            <Metric icon={Camera} label="Cameras" value="3" />
            <Metric icon={HardHat} label="PPE Compliance" value={`${inputs.ppeViolation ? 78 : 98}%`} />
            <Metric icon={ShieldAlert} label="Active Alerts" value={activeEvents.length} />
            <Metric icon={Waves} label="Gas Level" value={`${inputs.gasLevel}%`} />
          </div>

          <div className="main-panels">
            <section className="detection-panel panel">
              <div className="panel-title">
                <span className="live-dot" />
                Live Detection
              </div>
              <div className="mock-video">
                <div className="person-box">person</div>
                <div className="restricted-zone">Restricted Zone</div>
                {inputs.ppeViolation && <div className="warning-label"><AlertTriangle size={16} /> PPE Warning</div>}
                <div className="video-overlay">Risk: {score}</div>
              </div>
              <div className="zone-mock">
                <MapPinned />
                Restricted Zone Drawing
                <span>Saved for CCTV-1 / Zone A</span>
              </div>
            </section>

            <section className="panel">
              <div className="panel-title">Recent Safety Events</div>
              <div className="events-list">
                {activeEvents.map((event) => (
                  <motion.button
                    layout
                    className={`alert-row ${event.severity.toLowerCase()}`}
                    key={event.id}
                    onClick={() => setSelectedAlert(event)}
                  >
                    <span>{new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                    <b>{event.title}</b>
                    <small>{event.source} · {event.zone}</small>
                  </motion.button>
                ))}
              </div>
            </section>
          </div>

          <div className="insight-grid">
            <section className="panel explanation-card">
              <div className="panel-title">
                <BrainCircuit />
                Explain This Alert
              </div>
              <h3>{selectedAlert.title}</h3>
              <p><b>What happened:</b> {selectedAlert.what}</p>
              <p><b>Why risky:</b> {selectedAlert.why}</p>
              <div className="factor-list">
                {selectedAlert.factors.map((factor) => <span key={factor}>{factor}</span>)}
              </div>
              <p><b>Recommended action:</b> {selectedAlert.action}</p>
            </section>

            <section className="panel advisor-card">
              <div className="panel-title">
                <BrainCircuit />
                AI Safety Advisor
              </div>
              <h3>{risk.label} Risk · {score}/100</h3>
              <p>
                {score >= 85
                  ? "Immediate intervention recommended. Pause permit activity, verify gas readings and clear restricted zones."
                  : score >= 60
                    ? "Supervisor review recommended. Compound risk is building across CCTV and plant context."
                    : "Plant conditions are stable. Continue monitoring CCTV, permits and gas trend."}
              </p>
              <div className="advisor-actions">
                <span>Notify supervisor</span>
                <span>Preserve evidence</span>
                <span>Update checklist</span>
              </div>
            </section>
          </div>

          <section className="panel heatmap-panel">
            <div className="panel-title">
              <Radar />
              Risk Heatmap
            </div>
            <div className="heatmap-grid">
              <HeatZone name="Zone A" score={Math.min(100, score + 8)} workers="2 workers" />
              <HeatZone name="Zone B" score={Math.min(100, score)} workers="1 worker" />
              <HeatZone name="Control Room" score={Math.max(18, score - 30)} workers="4 operators" />
              <HeatZone name="Reactor Zone" score={Math.min(100, inputs.gasLevel + 10)} workers="Gas watch active" />
            </div>
          </section>
        </section>
      </section>

      <AnimatePresence>
        {architectureOpen && <ArchitectureModal onClose={() => setArchitectureOpen(false)} />}
      </AnimatePresence>
    </main>
  );
}

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="metric-card">
      <Icon size={20} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function HeatZone({ name, score, workers }) {
  const risk = classifyRisk(score);
  return (
    <div className={`heat-zone ${risk.color}`}>
      <div>
        <h3>{name}</h3>
        <p>{workers}</p>
      </div>
      <strong>{score}/100</strong>
      <span>{risk.label}</span>
    </div>
  );
}

function App() {
  const [page, setPage] = useState("landing");
  return page === "landing"
    ? <LandingPage onDemo={() => setPage("dashboard")} />
    : <DashboardPage onHome={() => setPage("landing")} />;
}

createRoot(document.getElementById("root")).render(<App />);

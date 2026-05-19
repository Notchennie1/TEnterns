/**
 * AEGIS v2 — Dashboard Frontend
 * ===============================
 * Polls the FastAPI backend every second and updates:
 *   - Bin grid with color-coded status
 *   - FSM gate progress
 *   - Hand tracking cards
 *   - System stats
 *   - Error overlay
 */

const POLL_INTERVAL = 1000; // ms

// ── FSM display config ──────────────────────────────
const FSM_DISPLAY = {
  idle:           { label: "IDLE",           css: "fsm-idle",    gates: [0, 0, 0] },
  gate_1_spatial: { label: "GATE 1: SPATIAL",css: "fsm-spatial", gates: [1, 0, 0] },
  gate_2_intent:  { label: "GATE 2: INTENT", css: "fsm-intent",  gates: [2, 1, 0] },
  gate_3_verify:  { label: "GATE 3: VERIFY", css: "fsm-verify",  gates: [2, 2, 1] },
  success:        { label: "SUCCESS",         css: "fsm-success", gates: [2, 2, 2] },
  error:          { label: "ERROR",           css: "fsm-error",   gates: [-1,-1,-1] },
};

// ── Main poll loop ──────────────────────────────────
async function poll() {
  try {
    const [binsRes, handsRes, fsmRes, errRes, statsRes] = await Promise.all([
      fetch("/api/bins"),
      fetch("/api/hands"),
      fetch("/api/fsm"),
      fetch("/api/errors"),
      fetch("/api/stats"),
    ]);

    if (!binsRes.ok) throw new Error("Backend error");

    const bins   = await binsRes.json();
    const hands  = await handsRes.json();
    const fsm    = await fsmRes.json();
    const errors = await errRes.json();
    const stats  = await statsRes.json();

    renderBins(bins);
    renderFSM(fsm);
    renderHands(hands);
    renderStats(stats);
    renderErrors(errors);

    document.getElementById("last-updated").textContent =
      "Updated: " + new Date().toLocaleTimeString();
    document.getElementById("fps-display").textContent =
      (stats.fps || 0).toFixed(0) + " FPS";

  } catch (err) {
    console.error("Poll error:", err);
    document.getElementById("last-updated").textContent = "Connection lost";
  }
}

// ── Bin grid ────────────────────────────────────────
function renderBins(bins) {
  const container = document.getElementById("bins");
  container.innerHTML = "";

  for (const bin of bins) {
    const box = document.createElement("div");
    const status = String(bin.status).trim().toLowerCase();

    box.className = "bin " + status;

    let inner = '<div class="bin-id">' + bin.id + '</div>';

    if (status !== "grey") {
      if (bin.total > 0) {
        inner += '<div class="quantity">' + bin.current + '/' + bin.total + '</div>';
      } else {
        inner += '<div class="quantity">' + bin.current + '</div>';
      }
    }

    if (bin.is_active && bin.handedness) {
      inner += '<div class="bin-badge" style="background:#3b82f6;color:#fff;">'
             + bin.handedness.toUpperCase() + '</div>';
    }

    box.innerHTML = inner;
    container.appendChild(box);
  }

  if (bins.length === 0) {
    container.innerHTML = '<div class="no-hands">No bins detected yet</div>';
  }
}

// ── FSM state ───────────────────────────────────────
function renderFSM(fsm) {
  const display = FSM_DISPLAY[fsm.state] || FSM_DISPLAY.idle;
  const label = document.getElementById("fsm-label");
  const binEl = document.getElementById("fsm-bin");
  const elapsedEl = document.getElementById("fsm-elapsed");

  // Remove all fsm-* classes, add the current one
  label.className = "";
  label.classList.add(display.css);
  label.textContent = display.label;

  binEl.textContent = fsm.bin_id ? ("Bin: " + fsm.bin_id) : "";
  elapsedEl.textContent = fsm.elapsed > 0 ? (fsm.elapsed.toFixed(1) + "s") : "";

  // Gate dots
  const dots = ["gate-1", "gate-2", "gate-3"];
  for (let i = 0; i < 3; i++) {
    const dot = document.getElementById(dots[i]);
    dot.className = "gate-dot";
    const g = display.gates[i];
    if (g === 2) dot.classList.add("passed");
    else if (g === 1) dot.classList.add("current");
    else if (g === -1) dot.classList.add("failed");
  }
}

// ── Hand cards ──────────────────────────────────────
function renderHands(hands) {
  const container = document.getElementById("hands");
  container.innerHTML = "";

  if (hands.length === 0) {
    container.innerHTML = '<div class="no-hands">No hands detected</div>';
    return;
  }

  for (const hand of hands) {
    const card = document.createElement("div");
    card.className = "hand-card";

    const grabClass = hand.is_grabbing ? "grabbing" : "open";
    const grabText  = hand.is_grabbing ? "GRABBING" : "OPEN";

    card.innerHTML =
      '<div class="hand-label">' + hand.handedness + ' hand</div>' +
      '<div class="hand-detail">Position: (' +
        Math.round(hand.x) + ', ' + Math.round(hand.y) + ')</div>' +
      '<div class="hand-detail">Bin: ' +
        (hand.assigned_bin || '—') + '</div>' +
      '<div>' +
        '<span class="hand-grab ' + grabClass + '">' + grabText +
        ' (' + (hand.grab_score * 100).toFixed(0) + '%)</span>' +
      '</div>';

    container.appendChild(card);
  }
}

// ── Stats ───────────────────────────────────────────
function renderStats(stats) {
  document.getElementById("stat-frames").textContent =
    (stats.frame_count || 0).toLocaleString();
  document.getElementById("stat-uptime").textContent =
    formatUptime(stats.uptime_seconds || 0);
  document.getElementById("stat-bins").textContent = stats.num_bins || 0;
  document.getElementById("stat-hands").textContent = stats.num_hands || 0;
}

function formatUptime(seconds) {
  if (seconds < 60) return Math.floor(seconds) + "s";
  if (seconds < 3600) return Math.floor(seconds / 60) + "m " + (Math.floor(seconds) % 60) + "s";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h + "h " + m + "m";
}

// ── Error overlay ───────────────────────────────────
function renderErrors(errors) {
  const overlay = document.getElementById("err-overlay");
  const content = document.getElementById("err-content");

  // Only show errors from the last 10 seconds
  const recent = errors.filter(function(e) {
    return (Date.now() / 1000 - e.timestamp) < 10;
  });

  if (recent.length > 0) {
    const lines = recent.map(function(e) {
      return e.bin_id + ": " + e.message;
    });
    content.innerHTML =
      "WARNING<br>" +
      "Gate Error Detected<br><br>" +
      lines.join("<br>");
    overlay.classList.remove("hidden");
  } else {
    overlay.classList.add("hidden");
  }
}

function dismissError() {
  document.getElementById("err-overlay").classList.add("hidden");
  fetch("/api/errors/clear", { method: "POST" });
}

// ── Start polling ───────────────────────────────────
poll();
setInterval(poll, POLL_INTERVAL);

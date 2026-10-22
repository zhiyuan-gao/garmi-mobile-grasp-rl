// Uses the existing Workspace's BASE and frames; credentials stay in its session.
const rlApi = BASE + "proxy/8781/";
const rlControls = document.getElementById("garmi-rl-controls");
const rlStart = document.getElementById("rl-start");
const rlStop = document.getElementById("rl-stop");
const rlStatus = document.getElementById("rl-status");
const rlError = document.getElementById("rl-error");
const rlStages = document.getElementById("rl-stages");
document.getElementById("rl-details").href = rlApi + "panel.html";
let rlToken = "", rlViewerKey = "", rlBusy = false;
let rlLatestStatus = null;
const rlSelect = document.getElementById("garmi-demo-select");
const rlRuntime = document.getElementById("rl-runtime");
const originalHelp = document.getElementById("garmi-original-help");

function rlSetViewer(url) {
  const frame = frames[0].iframes[0];
  frame.dataset.src = "";
  frame.src = url;
  frame.status.classList.remove("shown");
  tabs[0].querySelector("a").href = url;
}

function rlSelectDemo() {
  const selected = rlSelect.value === "garmi-rl";
  store("cram-workspace.demo", rlSelect.value);
  rlRuntime.hidden = !selected;
  originalHelp.hidden = selected;
  rlViewerKey = "";
  if (selected) {
    rlSetViewer(BASE + "proxy/8721/?live=" + encodeURIComponent(BASE + "proxy/8766"));
    if (rlLatestStatus) rlRender(rlLatestStatus);
  } else {
    rlSetViewer(BASE + "proxy/8711/");
    originalHelp.textContent = rlLatestStatus && rlLatestStatus.active
      ? "The RL task is still running. Switch back to GARMI RL and stop it before starting the original GARMI demo."
      : "Original GARMI NeemHub. Start the demo using the GARMI_apartment icon on the Desktop.";
  }
}

function rlRender(status) {
  rlLatestStatus = status;
  rlToken = status.csrf_token;
  rlStart.disabled = rlBusy || status.active;
  rlStop.disabled = rlBusy || !status.active || status.status === "stopping";
  const snapshot = (status.state || {}).snapshot || {};
  const elapsed = status.started_at
    ? Math.max(0, Math.floor((status.finished_at || Date.now() / 1000) - status.started_at)) : null;
  rlStatus.textContent = status.message + (elapsed === null ? "" : ` · ${elapsed} s`) +
    (snapshot.simulation_seconds == null ? "" : ` · Simulation ${snapshot.simulation_seconds.toFixed(1)} s`);
  rlStages.replaceChildren(...status.stages.map((name, index) => {
    const item = document.createElement("li");
    item.textContent = `${index + 1}. ${name}`;
    item.className = status.status === "completed" || index < status.stage ? "done"
      : status.active && index === status.stage ? "current" : "";
    return item;
  }));
  const viewerReady = (status.events || []).some(e => e.event === "viewer_ready") ||
    status.stage > 0 || status.recording;
  if (rlSelect.value === "garmi-rl" && status.run_id && viewerReady) {
    const recorded = status.recording && !status.active;
    const key = status.run_id + (recorded ? ":recorded" : ":live");
    if (key !== rlViewerKey) {
      rlViewerKey = key;
      const url = BASE + "proxy/8721/?live=" + encodeURIComponent(BASE + "proxy/8766") +
        "&run=" + encodeURIComponent(status.run_id) + (recorded ? "&scene=__recording__" : "");
      rlSetViewer(url);
    }
  }
}

async function rlFetchStatus() {
  const response = await fetch(rlApi + "api/status", {cache: "no-store"});
  if (!response.ok) throw new Error(`Status service returned ${response.status}`);
  rlRender(await response.json());
}

async function rlPoll() {
  try {
    await rlFetchStatus();
    if (rlError.dataset.connection) {
      rlError.textContent = "";
      delete rlError.dataset.connection;
    }
  } catch (error) {
    rlStatus.textContent = "Connecting to the RL service...";
    rlError.textContent = error.message;
    rlError.dataset.connection = "true";
    rlStart.disabled = rlStop.disabled = true;
  }
  setTimeout(rlPoll, 1500);
}

async function rlCommand(action) {
  rlBusy = true;
  rlStart.disabled = rlStop.disabled = true;
  rlError.textContent = "";
  delete rlError.dataset.connection;
  try {
    const response = await fetch(rlApi + "api/" + action, {
      method: "POST", headers: {"X-Demo-Token": rlToken}
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed ${response.status}`);
  } catch (error) {
    rlError.textContent = error.message;
  } finally {
    rlBusy = false;
    try { await rlFetchStatus(); } catch (error) { rlError.textContent = error.message; }
  }
}
rlStart.onclick = () => rlCommand("start");
rlStop.onclick = () => rlCommand("stop");
rlSelect.value = window.location.hash === "#garmi-rl" ? "garmi-rl"
  : stored("cram-workspace.demo") === "garmi" ? "garmi" : "garmi-rl";
rlSelect.onchange = rlSelectDemo;
rlSelectDemo();
rlPoll();

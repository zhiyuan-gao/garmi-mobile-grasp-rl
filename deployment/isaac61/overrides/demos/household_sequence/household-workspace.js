// Controls use the same Jupyter proxy paths locally and on Binder.
(() => {
  const header = document.querySelector('header');
  const controls = document.createElement('span');
  controls.style.cssText = 'display:inline-flex;align-items:center;gap:8px;font-size:12px';
  const start = document.createElement('button'), stop = document.createElement('button'), status = document.createElement('span');
  start.textContent = 'Start task'; stop.textContent = 'Stop';
  start.className = stop.className = 'tab';
  start.disabled = stop.disabled = true; status.textContent = 'Connecting…';
  status.setAttribute('role', 'status');
  controls.append(start, stop, status); header.append(controls);
  const base = location.origin + (location.pathname.match(/^(\/user\/[^/]+\/)/)?.[1] || '/');
  const api = base + 'proxy/8782/api/';
  let token = '', busy = false, commandError = '', commandLabel = '';
  async function poll() {
    try {
      const response = await fetch(api + 'status', {cache:'no-store'});
      if (!response.ok) throw Error('Starting task service…');
      const data = await response.json(); token = data.csrf_token;
      start.disabled = busy || data.active;
      stop.disabled = busy || (!data.active && !data.display_open) || data.status === 'stopping';
      const stage = data.stages?.[data.stage];
      const elapsed = Math.max(0, Math.floor(Date.now() / 1000 - (data.started_at || Date.now() / 1000)));
      start.textContent = busy ? 'Please wait…' : data.status === 'starting' ? 'Starting…' : data.active ? 'Running' : 'Start task';
      status.textContent = commandError || (busy ? commandLabel : data.status === 'starting'
        ? `${data.message} (${elapsed}s)` : stage && data.active ? stage : data.message);
      status.title = data.message;
    } catch(error) { status.textContent = error.message; start.disabled = stop.disabled = true; }
  }
  async function command(action) {
    busy = true; commandError = '';
    commandLabel = action === 'start' ? 'Starting task; preparing Isaac Sim…' : 'Stopping task…';
    status.textContent = commandLabel; start.textContent = 'Please wait…';
    start.disabled = stop.disabled = true;
    try {
      const response = await fetch(api + action, {method:'POST',headers:{'X-Demo-Token':token}});
      const data = await response.json(); if (!response.ok) throw Error(data.error);
    } catch(error) { commandError = error.message || 'Task request failed'; status.textContent = commandError; }
    finally { busy = false; await poll(); }
  }
  start.onclick = () => command('start'); stop.onclick = () => command('stop');
  poll(); setInterval(poll, 1500);
})();

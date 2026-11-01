// Integrate the live task into NeemHub's existing Task picker and playback area.
(function () {
  'use strict';
  const liveScene = '__live__';
  const params = new URLSearchParams(location.search);
  if (!params.has('scene') || params.get('scene') === 'garmi_rl_pick_up') {
    params.set('scene', liveScene);
    history.replaceState(null, '', location.pathname + '?' + params + location.hash);
  }
  const household = SceneContext.name() === liveScene;
  const proxied = /^(.*\/proxy\/)\d+(?:\/|$)/.exec(location.pathname);
  const service = port => proxied ? location.origin + proxied[1] + port
    : location.protocol + '//' + location.hostname + ':' + port;
  const api = service(8782);
  const live = service(8767);
  const originalLiveUrl = SceneContext.liveUrl;
  SceneContext.liveUrl = () => household ? live : originalLiveUrl();
  if (household) document.documentElement.classList.add('household-task');

  window.HouseholdTask = {
    async loadScene(base, name) {
      if (name !== liveScene) {
        const response = await fetch(base + 'scene.json');
        if (!response.ok) throw new Error('Scene unavailable');
        return response.json();
      }
      // The simulator may still be warming up. Wait for its real world bundle.
      for (;;) {
        try {
          const ready = await fetch(live + '/live_scene');
          const info = ready.ok && await ready.json();
          if (info && info.scene === liveScene) {
            const response = await fetch(base + 'scene.json', {cache: 'no-store'});
            if (response.ok) return response.json();
          }
        } catch (_) { /* retry while the task starts */ }
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
  };

  if (!household) return;
  let latest = null, busy = false;
  function mount() {
    const root = document.querySelector('.panel-robot-scene');
    const start = root && root.querySelector('#play-btn');
    if (!start) return;
    const stop = document.createElement('button');
    stop.id = 'household-stop'; stop.className = 'play-btn';
    stop.textContent = 'Stop task'; stop.disabled = true;
    start.after(stop);
    const status = document.createElement('span');
    status.id = 'household-status'; status.setAttribute('role', 'status');
    root.querySelector('.workflow-head').append(status);
    start.textContent = 'Start task';
    start.title = 'Run one bowl-to-table and mug-to-drawer task';

    function render(data) {
      latest = data;
      start.textContent = 'Start task';
      start.disabled = busy || data.active;
      stop.disabled = busy || (!data.active && !data.display_open) || data.status === 'stopping';
      const stage = data.stages && data.stages[data.stage];
      status.textContent = data.message + (stage ? ' · ' + (data.stage + 1) + '/' + data.stages.length + ': ' + stage : '');
    }
    async function poll() {
      try {
        const response = await fetch(api + '/api/status', {cache: 'no-store'});
        if (!response.ok) throw new Error('Task service unavailable');
        render(await response.json());
      } catch (_) {
        status.textContent = 'Connecting to Bowl and Mug…';
        start.disabled = stop.disabled = true;
      }
    }
    async function command(action) {
      if (busy || !latest) return;
      busy = true; start.disabled = stop.disabled = true;
      try {
        const response = await fetch(api + '/api/' + action, {
          method: 'POST', headers: {'X-Demo-Token': latest.csrf_token}
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Task command failed');
        await poll();
      } catch (error) {
        status.textContent = error.message;
      } finally {
        busy = false;
        if (latest) {
          start.disabled = latest.active;
          stop.disabled = (!latest.active && !latest.display_open) || latest.status === 'stopping';
        }
      }
    }
    // Capture prevents the recorded-trajectory handler from running for this entry.
    start.addEventListener('click', event => {
      event.preventDefault(); event.stopImmediatePropagation(); command('start');
    }, true);
    stop.addEventListener('click', () => command('stop'));
    poll();
    setInterval(poll, 1500);
  }
  document.addEventListener('DOMContentLoaded', mount, {once: true});
})();

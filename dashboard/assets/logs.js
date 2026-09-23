/* Public archive viewer. Log content only enters text nodes, never HTML. */
(() => {
  'use strict';
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const labels = {archived: 'Archived', completed: 'Completed', timed_out: 'Time limit', cancelled: 'Cancelled', failed: 'Failed', report: 'Report', success: 'Success'};
  const agents = {codex: 'Codex', 'claude-code': 'Claude Code', report: 'Research report'};
  let data, track = 'all', request = 0;
  const expandedTasks = new Map();
  const cache = new Map();
  const number = value => new Intl.NumberFormat('en-US').format(value);
  const score = value => Number.isFinite(value) ? new Intl.NumberFormat('en-US', {maximumFractionDigits: 5}).format(value) : '—';
  const model = value => value.replace(/^gpt-/i, 'GPT-').replace(/-sol$/, ' Sol').replace(/^claude-(opus|fable)-/, (_, family) => 'Claude ' + family[0].toUpperCase() + family.slice(1) + ' ');
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const badge = status => `<span class="badge ${escape(status)}">${escape(labels[status] || status.replaceAll('_', ' '))}</span>`;
  const modelLogo = run => {
    const provider = /^claude-/i.test(run.model) ? 'claude' : /^gpt-/i.test(run.model) ? 'openai' : null;
    return provider ? `<img class="model-logo" src="assets/logos/${provider}.${provider === 'claude' ? 'png' : 'svg'}" alt="" width="22" height="22">` : '<span class="agent-mark report" aria-hidden="true">▤</span>';
  };
  const identity = run => `<div class="model-line">${modelLogo(run)}<span class="model-name">${escape(model(run.model))}</span></div><div class="agent-note">${escape(agents[run.agent] || run.agent)}${run.effort ? ' · ' + escape(run.effort) : ''}</div>`;
  const stat = (label, value, note) => `<div class="card stat"><div class="stat-label">${escape(label)}</div><div class="stat-bottom"><strong class="stat-number">${escape(value)}</strong><span class="stat-note">${escape(note)}</span></div></div>`;
  const route = (id, tab = 'trajectory', page = 1) => `#run=${encodeURIComponent(id)}&tab=${encodeURIComponent(tab)}&page=${page}`;
  const date = value => new Date(value).toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric', timeZone:'UTC'});

  async function json(path) {
    const response = await fetch(`assets/logs-data/${path}${data ? '?v=' + data.commit : ''}`, {cache: 'no-cache'});
    if (!response.ok) throw new Error(`Archive request failed (${response.status})`);
    if (path.endsWith('.gz')) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      // Also accept hosts that already decoded Content-Encoding: gzip.
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
      }
      return JSON.parse(new TextDecoder().decode(bytes));
    }
    return response.json();
  }

  function renderRows() {
    const query = $('#search').value.trim().toLowerCase();
    const runs = data.runs.filter(run => (track === 'all' || run.track === track)
      && ($('#domain').value === 'all' || run.domain === $('#domain').value)
      && ($('#agent').value === 'all' || run.agent === $('#agent').value)
      && ($('#outcome').value === 'all' || run.status === $('#outcome').value)
      && `${run.task} ${run.title} ${run.model} ${model(run.model)} ${agents[run.agent] || run.agent}`.toLowerCase().includes(query));
    const byTask = new Map();
    for (const run of runs) {
      if (!byTask.has(run.task)) byTask.set(run.task, []);
      byTask.get(run.task).push(run);
    }
    const groups = [...byTask.values()];
    const sortKey = $('#task-sort').value;
    const total = (group, key) => group.reduce((sum, run) => sum + (key === 'submissions' ? run.submissions.length : run.events), 0);
    groups.sort((a, b) => (sortKey === 'task' ? 0 : total(b, sortKey) - total(a, sortKey)) || a[0].task.localeCompare(b[0].task));
    const filtering = query || ['domain', 'agent', 'outcome'].some(id => $('#' + id).value !== 'all');
    $('#runs').innerHTML = groups.map(group => {
      const task = group[0];
      const open = expandedTasks.has(task.task) ? expandedTasks.get(task.task) : Boolean(filtering);
      return `<details class="task-group card" data-task="${escape(task.task)}" ${open ? 'open' : ''}><summary><span class="task-chevron" aria-hidden="true">›</span><span class="task-heading"><span class="task-name">${escape(task.title)}</span><span class="task-meta"><span>${escape(task.domain)}</span><span>·</span><span class="${task.track === 'Signature' ? 'signature' : ''}">${task.track} task</span></span></span><span class="task-models" aria-hidden="true">${[...new Map(group.map(run => [run.model, run])).values()].map(modelLogo).join('')}</span><span class="task-count">${group.length} ${group.length === 1 ? 'record' : 'records'}</span></summary><div class="table-wrap"><table class="task-runs-table" aria-label="${escape(task.title)} runs"><thead><tr><th scope="col">AGENT / MODEL</th><th scope="col">OUTCOME</th><th scope="col" class="numeric">BEST SCORE</th><th scope="col" class="numeric">SUBMISSIONS</th><th scope="col" class="numeric">LOG EVENTS</th><th scope="col"><span class="sr-only">Open record</span></th></tr></thead><tbody>${group.map(run => `<tr class="run-row"><td>${identity(run)}</td><td>${badge(run.status)}</td><td class="numeric score">${score(run.best_score)}</td><td class="numeric mono">${run.submissions.length ? number(run.submissions.length) : '—'}</td><td class="numeric mono">${number(run.events)}</td><td class="numeric"><a class="row-link" aria-label="Open ${escape(run.task)} ${escape(run.model)} logs" href="${route(run.id)}">View log <span aria-hidden="true">→</span></a></td></tr>`).join('')}</tbody></table></div></details>`;
    }).join('');
    $$('.task-group').forEach(group => group.addEventListener('toggle', () => expandedTasks.set(group.dataset.task, group.open)));
    $('#empty-filter').hidden = runs.length !== 0;
    $('#result-count').textContent = `${groups.length} tasks · ${runs.length} of ${data.runs.length} records`;
    $('#record-count').textContent = `${groups.length} tasks · ${runs.length} records`;
  }

  function resetFilters() {
    track = 'all';
    $('#search').value = '';
    ['domain', 'agent', 'outcome'].forEach(id => $('#' + id).value = 'all');
    $$('.filter').forEach(button => {
      const active = button.dataset.track === 'all';
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    renderRows();
  }

  function chart(run) {
    const points = run.submissions;
    const valid = points.map((point, index) => ({...point, index})).filter(p => Number.isFinite(p.score) && p.status === 'completed');
    if (!valid.length) return '';
    let low = Math.min(...valid.map(p => p.score)), high = Math.max(...valid.map(p => p.score));
    const padding = (high - low || Math.abs(high) || 1) * .12;
    low -= padding; high += padding;
    const x = i => 70 + i / Math.max(points.length - 1, 1) * 1020;
    const y = v => 160 - (v - low) / (high - low) * 140;
    let best = null, path = '', dots = '';
    const minimize = run.direction.startsWith('min');
    for (const p of valid) {
      const previous = best;
      best = best === null ? p.score : (minimize ? Math.min(best, p.score) : Math.max(best, p.score));
      path += previous === null ? `M${x(p.index)},${y(best)}` : `H${x(p.index)}V${y(best)}`;
      dots += `<circle cx="${x(p.index)}" cy="${y(p.score)}" r="3.5" fill="white" stroke="#7d76e9" stroke-width="1.5"><title>${escape(p.round)}: ${score(p.score)}</title></circle>`;
    }
    if (best !== null) path += `H${x(points.length - 1)}`;
    const grid = [0, 1, 2, 3].map(i => {
      const v = low + (high-low) * i / 3;
      return `<line x1="70" x2="1090" y1="${y(v)}" y2="${y(v)}" stroke="#edf0f6"/><text x="59" y="${y(v)+4}" text-anchor="end">${score(v)}</text>`;
    }).join('');
    const ticks = [...new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])].map(i => `<text x="${x(i)}" y="184" text-anchor="middle">${i + 1}</text>`).join('');
    return `<div class="card chart-card"><div class="chart-header"><h2>Evaluation progress</h2><p class="muted">Native task score · ${minimize ? 'lower' : 'higher'} is better</p></div><svg class="chart" viewBox="0 0 1120 205" role="img" aria-label="Submission scores and best score over ${points.length} submissions">${grid}<path d="${path}" fill="none" stroke="#4f46e5" stroke-width="2"/>${dots}${ticks}<text x="580" y="204" text-anchor="middle">Submission</text></svg><div class="chart-legend"><span><i class="legend-dot"></i>Evaluated submission</span><span><i class="legend-line"></i>Best so far</span></div>${valid.length < points.length ? '<p class="score-note">Submissions with evaluation errors are listed below and excluded from the curve.</p>' : ''}</div>`;
  }

  // Small Markdown subset built from DOM nodes. Raw HTML, images, and links remain inert text.
  function prose(value) {
    const box = document.createElement('div'); box.className = 'prose';
    let buffer = [], fence = null, code = [], list = null;
    function inline(target, content) {
      for (const token of content.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g)) {
        if (token.startsWith('`') && token.endsWith('`')) {
          const node = document.createElement('code'); node.textContent = token.slice(1,-1); target.append(node);
        } else if (token.startsWith('**') && token.endsWith('**')) {
          const node = document.createElement('strong'); node.textContent = token.slice(2,-2); target.append(node);
        } else target.append(document.createTextNode(token));
      }
    }
    function flush() { if (buffer.length) { const p = document.createElement('p'); inline(p, buffer.join('\n')); box.append(p); buffer = []; } }
    function flushCode() { const pre = document.createElement('pre'); const node = document.createElement('code'); node.textContent = code.join('\n'); pre.append(node); box.append(pre); code = []; }
    for (const line of value.split('\n')) {
      const marker = line.match(/^\s{0,3}(`{3,}|~{3,})/);
      if (fence) {
        if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length) { flushCode(); fence = null; }
        else code.push(line);
      } else if (marker) { flush(); list = null; fence = marker[1]; }
      else if (/^#{1,6} /.test(line)) { flush(); list = null; const h = document.createElement('h3'); inline(h, line.replace(/^#{1,6} /, '')); box.append(h); }
      else if (/^[-*] /.test(line)) { flush(); if (!list) { list = document.createElement('ul'); box.append(list); } const li = document.createElement('li'); inline(li,line.slice(2)); list.append(li); }
      else if (!line.trim()) { flush(); list = null; }
      else { list = null; buffer.push(line); }
    }
    flush(); if (fence) flushCode();
    return box;
  }

  function eventCard(item) {
    const article = document.createElement('article'); article.className = `card event ${item.kind === 'user' ? 'user' : ''}${item.error ? ' error' : ''}`;
    const head = document.createElement('div'); head.className = 'event-head';
    const id = document.createElement('span'); id.className = 'event-id'; id.textContent = '#' + item.id;
    const kind = document.createElement('span'); kind.className = 'kind'; kind.textContent = item.kind;
    const title = document.createElement('strong'); title.textContent = item.title;
    head.append(id, kind, title);
    if (item.parent) { const tag = document.createElement('span'); tag.textContent = 'Subagent'; tag.title = item.parent; head.append(tag); }
    if (item.at) { const time = document.createElement('time'); time.textContent = item.at; head.append(time); }
    article.append(head);
    const isMessage = ['user', 'assistant', 'experiment'].includes(item.kind);
    if (isMessage && item.text.length < 6000) article.append(item.kind === 'experiment' ? pre(item.text) : prose(item.text));
    else {
      const details = document.createElement('details');
      const summary = document.createElement('summary'); summary.append(document.createTextNode(isMessage ? 'Read message' : item.kind === 'thinking' ? 'Show reasoning' : 'Show details'));
      const preview = document.createElement('span'); preview.className = 'preview'; preview.textContent = item.text.replace(/\s+/g, ' ').slice(0,120);
      summary.append(preview); details.append(summary);
      details.addEventListener('toggle', () => {
        if (details.open && details.children.length === 1) details.append(isMessage && item.kind !== 'experiment' ? prose(item.text) : pre(item.text));
      });
      article.append(details);
    }
    return article;
  }

  function pre(value) { const node = document.createElement('pre'); node.textContent = value; return node; }
  function errorPanel(container, retry) {
    container.innerHTML = '<div class="card detail-error" role="alert"><p>This part of the archive could not be loaded.</p><button class="button" type="button">Try again</button></div>';
    container.querySelector('button').addEventListener('click', retry);
  }

  function pager(run, page, count, bottom = false) {
    const box = document.createElement('form'); box.className = 'pager' + (bottom ? ' reader-bottom' : '');
    box.setAttribute('aria-label', bottom ? 'Bottom page navigation' : 'Page navigation');
    box.innerHTML = `<button class="button previous" type="button" ${page <= 1 ? 'disabled' : ''}>← Previous</button><label>Page <input type="number" min="1" max="${count}" value="${page}" aria-label="${bottom ? 'Bottom page number' : 'Page number'}"></label><span>of ${number(count)}</span><button class="button next" type="button" ${page >= count ? 'disabled' : ''}>Next →</button>`;
    const go = next => { location.hash = route(run.id, 'trajectory', Math.max(1,Math.min(count,next))); };
    box.querySelector('.previous').onclick = () => go(page-1);
    box.querySelector('.next').onclick = () => go(page+1);
    box.onsubmit = e => { e.preventDefault(); const value = Number(box.querySelector('input').value); if (Number.isInteger(value)) go(value); };
    return box;
  }

  async function renderTrace(run, page, token) {
    const panel = $('#tab-content');
    if (!run.pages.length) { panel.innerHTML = '<div class="card empty">No trajectory events in this record. Source files are available in Artifacts.</div>'; return; }
    const count = run.pages.length;
    page = Math.max(1, Math.min(count, page));
    const toolbar = document.createElement('div'); toolbar.className = 'reader-bar';
    toolbar.innerHTML = '<div><label class="sr-only" for="event-filter">Event type on this page</label><select id="event-filter"><option value="all">All event types</option><option value="assistant">Assistant</option><option value="user">User</option><option value="thinking">Reasoning</option><option value="tool">Tool calls</option><option value="result">Tool results</option><option value="system">System</option><option value="experiment">Experiments</option></select> <label class="sr-only" for="event-search">Search this page</label><input id="event-search" type="search" placeholder="Search this page…"></div>';
    toolbar.append(pager(run, page, count));
    const info = document.createElement('p'); info.className = 'reader-info'; info.setAttribute('role','status');
    const entries = document.createElement('div'); entries.className = 'events'; entries.innerHTML = '<div class="empty" role="status">Loading log events…</div>';
    panel.replaceChildren(toolbar, info, entries, pager(run, page, count, true));
    try {
      const rows = await json(`${run.id}/${run.pages[page-1].file}`);
      if (token !== request) return;
      function render() {
        const type = $('#event-filter').value, query = $('#event-search').value.toLowerCase();
        const visible = rows.filter(row => (type === 'all' || row.kind === type) && (row.text + row.title).toLowerCase().includes(query));
        entries.replaceChildren(...visible.map(eventCard));
        if (!visible.length) entries.innerHTML = '<div class="card empty">No matching events on this page. Try another page or clear the event filters.</div>';
        info.textContent = `Events ${number(rows[0].id)}–${number(rows.at(-1).id)} of ${number(run.events)} · ${visible.length} shown on this page · Tool calls and reasoning expand on click.`;
      }
      $('#event-filter').onchange = render; $('#event-search').oninput = render; render();
    } catch { if (token === request) errorPanel(entries, renderRoute); }
  }

  async function renderRoute() {
    if (!data) return;
    const token = ++request;
    const params = new URLSearchParams(location.hash.slice(1));
    const id = params.get('run');
    const summary = data.runs.find(run => run.id === id);
    $('#overview').hidden = Boolean(id); $('#detail').hidden = !id;
    document.title = 'RSI Logs · OpenRSI Index';
    if (!id) return;
    if (!summary) { $('#detail').innerHTML = '<div class="card empty"><h1 id="detail-title">Record not found</h1><p>This record may no longer be in the current archive.</p><a class="button" href="#">← All records</a></div>'; return; }
    document.title = `${summary.task} · RSI Logs`;
    const tab = ['trajectory', 'submissions', 'artifacts'].includes(params.get('tab')) ? params.get('tab') : 'trajectory';
    const requestedPage = Number(params.get('page'));
    const page = Number.isInteger(requestedPage) && requestedPage > 0 ? requestedPage : 1;
    const peers = data.runs.filter(run => run.task === summary.task);
    $('#detail').innerHTML = `<div class="breadcrumbs"><a href="#">← All research logs</a><span>/</span><span>${escape(summary.domain)}</span></div><div class="page-heading detail-heading"><div><p class="eyebrow">${escape(summary.track)} TASK · ${escape(summary.format)}</p><h1 id="detail-title">${escape(summary.title)}</h1>${identity(summary)}</div><div class="detail-actions">${peers.length > 1 ? `<select class="related-select" aria-label="Select agent run">${peers.map(run => `<option value="${run.id}" ${run.id === id ? 'selected' : ''}>${escape(model(run.model))}</option>`).join('')}</select>` : ''}<a class="button" href="${escape(summary.source_url)}" target="_blank" rel="noopener">Source files ↗</a></div></div><div class="stats detail-stats">${stat('Outcome', labels[summary.status] || summary.status, 'as recorded in the archive')}${stat('Best score', score(summary.best_score), summary.best_round || 'No aggregate score recorded')}${stat('Submissions', number(summary.submissions.length), 'evaluation records')}${stat('Log events', number(summary.events), number(summary.tools) + ' tool calls')}</div>${summary.notes.length ? `<div class="notice">${summary.notes.map(note => `<p>${escape(note)}</p>`).join('')}</div>` : ''}${chart(summary)}<nav class="detail-tabs" aria-label="Record sections">${['trajectory','submissions','artifacts'].map(name => `<a href="${route(id,name)}" ${tab === name ? 'aria-current="page"' : ''}>${name[0].toUpperCase()+name.slice(1)}</a>`).join('')}</nav><div id="tab-content"></div>`;
    const select = $('.related-select'); if (select) select.onchange = () => location.hash = route(select.value);
    const panel = $('#tab-content');
    if (tab === 'submissions') {
      panel.innerHTML = !summary.submissions.length ? '<div class="card empty">No Harness evaluation submissions in this record. See its trajectory or source artifacts.</div>' : `<div class="card table-wrap"><table class="submissions"><thead><tr><th>SUBMISSION</th><th>STATUS</th><th class="numeric">SCORE</th><th>RECORDED AT (UTC)</th></tr></thead><tbody>${summary.submissions.map(s => `<tr><td class="mono">${escape(s.round)}${s.round === summary.best_round ? ' <span class="badge completed">Best</span>' : ''}</td><td>${badge(s.status)}</td><td class="numeric score">${score(s.score)}</td><td>${s.at ? escape(new Date(s.at*1000).toISOString().replace('T',' ').slice(0,19)) : '—'}</td></tr>`).join('')}</tbody></table></div>`;
      return;
    }
    if (tab === 'artifacts') {
      panel.innerHTML = `${summary.report_url ? `<p class="notice"><a href="${escape(summary.report_url)}">Open the research report and its provenance notes →</a></p>` : ''}<div class="card"><ul class="artifact-list">${summary.files.map(file => `<li><a href="${escape(file.url)}" target="_blank" rel="noopener">${escape(file.name)} ↗</a><span>${number(Math.ceil(file.bytes/1024))} KB</span></li>`).join('')}</ul></div>`;
      return;
    }
    panel.innerHTML = '<div class="empty" role="status">Loading trajectory…</div>';
    try {
      let run = cache.get(id);
      if (!run) { run = await json(`${id}/index.json`); cache.set(id, run); }
      if (token !== request) return;
      await renderTrace(run, page, token);
    } catch { if (token === request) errorPanel(panel, renderRoute); }
  }

  async function init() {
    $('#loading').hidden = false; $('#load-error').hidden = true;
    try {
      data = await json('index.json');
      const runs = data.runs;
      $('#stats').innerHTML = stat('Research tasks', number(new Set(runs.map(run => run.task)).size), 'across the archive')
        + stat('Run records', number(runs.length), 'publicly available')
        + stat('Evaluations', number(runs.reduce((total, run) => total + run.submissions.length,0)), 'recorded submissions')
        + stat('Tool calls', number(runs.reduce((total, run) => total + run.tools,0)), 'inside agent trajectories');
      for (const [id, key, names] of [['domain','domain',{}],['agent','agent',agents],['outcome','status',labels]]) {
        const select = $('#' + id); select.length = 1;
        [...new Set(runs.map(run => run[key]))].sort().forEach(value => { const option = document.createElement('option'); option.value = value; option.textContent = names[value] || value; select.append(option); });
      }
      $('#snapshot').innerHTML = `Archive built ${escape(date(data.generated_at))} · <a href="${escape(data.repository)}/tree/${escape(data.commit)}/rsi-logs" target="_blank" rel="noopener">${escape(data.commit.slice(0,7))} ↗</a>`;
      $('#loading').hidden = true;
      renderRows(); await renderRoute();
    } catch (error) { $('#loading').hidden = true; $('#overview').hidden = true; $('#detail').hidden = true; $('#load-error').hidden = false; }
  }
  $$('.filter').forEach(button => button.onclick = () => { track = button.dataset.track; $$('.filter').forEach(other => { other.classList.toggle('active',other === button); other.setAttribute('aria-pressed',String(other === button)); }); renderRows(); });
  ['search','domain','agent','outcome'].forEach(id => $('#' + id).addEventListener(id === 'search' ? 'input' : 'change', renderRows));
  $('#task-sort').onchange = renderRows;
  for (const [id, open] of [['expand-tasks', true], ['collapse-tasks', false]]) {
    $('#' + id).onclick = () => $$('.task-group').forEach(group => { expandedTasks.set(group.dataset.task, open); group.open = open; });
  }
  $('#clear').onclick = resetFilters; $('#empty-reset').onclick = resetFilters; $('#retry').onclick = init;
  window.addEventListener('hashchange', () => { renderRoute(); window.scrollTo(0,0); });
  init();
})();

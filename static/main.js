// NarrAI — main.js

let sessionId = localStorage.getItem('narrai_session_id');
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem('narrai_session_id', sessionId);
}

function apiFetch(url, options = {}) {
  options.headers = { ...options.headers, 'X-Session-ID': sessionId };
  return fetch(url, options);
}

let chapters = [];
let sourceFiles = [];
let selectedIdx = -1;
let isGenerating = false;
let totalTokens = 0;

// ── Session Info ──────────────────────────────────────────────────────────────

let isHost = false;
apiFetch('/api/session/info')
  .then(r => r.json())
  .then(data => {
    isHost = data.is_host;
    if (!isHost) {
      document.querySelectorAll('#genCount option[value="3"], #genCount option[value="5"], #genCount option[value="10"]')
        .forEach(opt => opt.disabled = true);
    }
  });

// ── Load Session Chapters ─────────────────────────────────────────────────────

apiFetch('/api/chapters')
  .then(r => r.json())
  .then(data => {
    sourceFiles = data.source || [];
    chapters = data.generated || [];
    if (sourceFiles.length || chapters.length) renderChapterList();
  });

// ── File Upload ──────────────────────────────────────────────────────────────

document.getElementById('fileInput').addEventListener('change', (e) => {
  handleFiles(e.target.files);
});

document.getElementById('uploadZone').addEventListener('dragover', (e) => {
  e.preventDefault();
  e.currentTarget.style.borderColor = 'var(--accent)';
});

document.getElementById('uploadZone').addEventListener('dragleave', (e) => {
  e.currentTarget.style.borderColor = '';
});

document.getElementById('uploadZone').addEventListener('drop', (e) => {
  e.preventDefault();
  e.currentTarget.style.borderColor = '';
  handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
  const formData = new FormData();
  Array.from(files).forEach(file => formData.append('files', file));

  apiFetch('/api/upload', { method: 'POST', body: formData })
    .then(r => {
      if (!r.ok) return r.json().then(e => { alert(e.detail); throw new Error(e.detail); });
      return r.json();
    })
    .then(() => {
      Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
          sourceFiles.push({ name: file.name, content: e.target.result });
          renderChapterList();
        };
        reader.readAsText(file);
      });
    });
}

// ── Chapter List ─────────────────────────────────────────────────────────────

function renderChapterList() {
  const list = document.getElementById('chapterList');
  const allItems = [
    ...sourceFiles.map((f, i) => ({ ...f, type: 'source', idx: i })),
    ...chapters.map((c, i) => ({ ...c, type: 'generated', idx: i })),
  ];

  if (!allItems.length) {
    list.innerHTML = '<div style="font-size:10px;color:var(--text3);padding:8px 4px;text-align:center;">No chapters yet</div>';
    return;
  }

  list.innerHTML = allItems.map((item, i) => `
    <div class="chapter-item ${selectedIdx === i ? 'active' : ''}" data-list-idx="${i}" data-type="${item.type}" data-idx="${item.idx}">
      <div class="chapter-num">${String(i + 1).padStart(2, '0')}</div>
      <div class="chapter-name" style="flex:1;overflow:hidden;min-width:0;">
        <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">Chapter ${i + 1}</div>
        <div style="font-size:9px;color:var(--text3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:2px;">${item.name.replace(/\.[^.]+$/, '')}</div>
      </div>
      <div class="chapter-status ${item.type === 'generated' ? 'done' : ''}"></div>
      <button class="delete-btn" data-type="${item.type}" data-idx="${item.idx}">×</button>
    </div>
  `).join('');

  list.querySelectorAll('.chapter-item').forEach(el => {
    el.addEventListener('click', () => {
      selectChapter(
        parseInt(el.dataset.listIdx),
        el.dataset.type,
        parseInt(el.dataset.idx)
      );
    });
  });

  list.querySelectorAll('.delete-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const type = btn.dataset.type;
      const idx = parseInt(btn.dataset.idx);
      const item = type === 'source' ? sourceFiles[idx] : chapters[idx];
      apiFetch(`/api/chapters/${encodeURIComponent(item.name)}`, { method: 'DELETE' });
      if (type === 'source') sourceFiles.splice(idx, 1);
      else chapters.splice(idx, 1);
      selectedIdx = -1;
      renderChapterList();
    });
  });

  Sortable.create(list, {
    animation: 150,
    onEnd(evt) {
      const allItems = [
        ...sourceFiles.map((f, i) => ({ ...f, type: 'source', idx: i })),
        ...chapters.map((c, i) => ({ ...c, type: 'generated', idx: i })),
      ];
      const moved = allItems.splice(evt.oldIndex, 1)[0];
      allItems.splice(evt.newIndex, 0, moved);

      sourceFiles = allItems.filter(i => i.type === 'source');
      chapters = allItems.filter(i => i.type === 'generated');

      apiFetch('/api/chapters/order', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ order: allItems.map(i => i.name) })
      });

      selectedIdx = -1;
      renderChapterList();
    }
  });
}

function selectChapter(listIdx, type, dataIdx) {
  selectedIdx = listIdx;
  renderChapterList();

  const data = type === 'source' ? sourceFiles[dataIdx] : chapters[dataIdx];
  const title = data.name.replace(/\.[^.]+$/, '').replace(/_/g, ' ');

  document.getElementById('chapterTitleDisplay').textContent = title;
  document.getElementById('chapterMetaDisplay').textContent =
    type === 'source' ? 'source chapter' : `generated · ${data.words || '—'} words`;

  document.getElementById('emptyState').style.display = 'none';
  const cc = document.getElementById('chapterContent');
  cc.style.display = 'block';

  const text = data.content || data.text || '';
  cc.innerHTML = text.split('\n').filter(Boolean).map(p =>
    `<p>${p.trim()}</p>`
  ).join('');
}

// ── Generation ───────────────────────────────────────────────────────────────


document.querySelectorAll('.theme-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.body.setAttribute('data-theme', btn.dataset.theme);
  });
});

const resizeHandle = document.getElementById('resizeHandle');
const sidebar = document.querySelector('.sidebar');
let sidebarExpandedWidth = 280;

function toggleSidebar() {
  if (sidebar.classList.contains('collapsed')) {
    sidebar.classList.remove('collapsed');
    sidebar.style.width = sidebarExpandedWidth + 'px';
  } else {
    sidebarExpandedWidth = sidebar.offsetWidth;
    sidebar.classList.add('collapsed');
  }
}

document.getElementById('sidebarToggle').addEventListener('click', toggleSidebar);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') toggleSidebar();

  const allItems = [
    ...sourceFiles.map((f, i) => ({ ...f, type: 'source', idx: i })),
    ...chapters.map((c, i) => ({ ...c, type: 'generated', idx: i })),
  ];

  if (e.key === 'ArrowRight' && selectedIdx < allItems.length - 1) {
    const next = allItems[selectedIdx + 1];
    selectChapter(selectedIdx + 1, next.type, next.idx);
    document.getElementById('contentArea').scrollTop = 0;
  } else if (e.key === 'ArrowLeft' && selectedIdx > 0) {
    const prev = allItems[selectedIdx - 1];
    selectChapter(selectedIdx - 1, prev.type, prev.idx);
    document.getElementById('contentArea').scrollTop = 0;
  }
});

resizeHandle.addEventListener('mousedown', (e) => {
  if (sidebar.classList.contains('collapsed')) return;
  e.preventDefault();
  resizeHandle.classList.add('dragging');

  const startX = e.clientX;
  const startWidth = sidebar.offsetWidth;
  let rafId = null;

  function onMouseMove(e) {
    if (rafId) return;
    rafId = requestAnimationFrame(() => {
      const newWidth = Math.max(185, Math.min(500, startWidth + e.clientX - startX));
      sidebar.style.width = newWidth + 'px';
      rafId = null;
    });
  }

  function onMouseUp() {
    resizeHandle.classList.remove('dragging');
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
  }

  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', onMouseUp);
});

document.getElementById('generateBtn').addEventListener('click', startGeneration);

async function startGeneration() {
  if (isGenerating) return;
  if (!sourceFiles.length) {
    alert('Upload at least one source chapter first.');
    return;
  }

  isGenerating = true;
  totalTokens = 0;
  document.getElementById('tokenCount').textContent = '— tokens';
  const btn = document.getElementById('generateBtn');
  btn.classList.add('generating');
  btn.innerHTML = '<span>Generating...</span>';
  document.getElementById('statusBadge').textContent = 'generating';

  const count = parseInt(document.getElementById('genCount').value);

  for (let i = 0; i < count; i++) {
    await generateChapter();
  }

  isGenerating = false;
  btn.classList.remove('generating');
  btn.innerHTML = '<span>Generate</span><span>→</span>';
  document.getElementById('statusBadge').textContent = 'ready';
  resetAgents();
}

async function generateChapter() {
  const chapterName = `Chapter ${sourceFiles.length + chapters.length + 1}`;
  const newChapter = { name: chapterName, text: '', words: 0 };
  chapters.push(newChapter);
  const chIdx = chapters.length - 1;
  const listIdx = sourceFiles.length + chIdx;

  selectedIdx = listIdx;
  renderChapterList();
  document.getElementById('chapterTitleDisplay').textContent = chapterName;
  document.getElementById('chapterMetaDisplay').textContent = 'generating…';
  document.getElementById('emptyState').style.display = 'none';
  const cc = document.getElementById('chapterContent');
  cc.style.display = 'block';
  cc.innerHTML = '<p><span class="cursor"></span></p>';

  resetAgents();

  const check = await apiFetch('/api/generate/check');
  if (!check.ok) {
    const e = await check.json();
    alert(e.detail);
    chapters.pop();
    renderChapterList();
    return;
  }

  await new Promise((resolve, reject) => {
    const es = new EventSource(`/api/generate?session_id=${sessionId}`);

    es.onmessage = (e) => {
      const event = JSON.parse(e.data);

      if (event.type === 'agent') {
        updateAgent(event.agent, event.status);
      } else if (event.type === 'result') {
        es.close();
        newChapter.text = event.text;
        if (event.title) newChapter.name = event.title;
        newChapter.words = event.text.split(/\s+/).filter(Boolean).length;
        cc.innerHTML = newChapter.text.split('\n').filter(Boolean).map(p =>
          `<p>${p.trim()}</p>`
        ).join('');
        document.getElementById('chapterMetaDisplay').textContent = `generated · ${newChapter.words} words`;
        totalTokens += event.tokens || 0;
        document.getElementById('tokenCount').textContent = totalTokens.toLocaleString() + ' tokens';
        renderChapterList();
        resolve();
      }
    };

    es.onerror = () => { es.close(); reject(new Error('SSE error')); };
  });
}

// ── Agent Bar ────────────────────────────────────────────────────────────────

const agentIdMap = {
  world_builder: 'agentWorldBuilder',
  cleaner: 'agentCleaner',
  plot_planner: 'agentPlotPlanner',
  analyzer: 'agentAnalyzer',
  predictor: 'agentPredictor',
  writer: 'agentWriter',
  critic: 'agentCritic',
};

function updateAgent(agent, status) {
  const id = agentIdMap[agent];
  if (!id) return;
  const el = document.getElementById(id);
  if (status === 'active') el.className = 'agent-step active';
  else if (status === 'done') el.className = 'agent-step done';
  else if (status.startsWith('error:')) {
    el.className = 'agent-step error';
    const msg = status.slice(6);
    document.getElementById('chapterMetaDisplay').textContent = `⚠ ${msg}`;
    document.getElementById('statusBadge').textContent = 'error';
  }
}

async function animateAgents() {
  const agents = ['agentPlanner', 'agentPredictor', 'agentWriter', 'agentCritic', 'agentState'];
  const durations = [600, 600, 800, 1800, 1400];

  agents.forEach(id => { document.getElementById(id).className = 'agent-step'; });

  for (let i = 0; i < agents.length; i++) {
    if (i > 0) document.getElementById(agents[i - 1]).className = 'agent-step done';
    document.getElementById(agents[i]).className = 'agent-step active';
    await sleep(durations[i]);
  }
  document.getElementById(agents[agents.length - 1]).className = 'agent-step done';
}

function resetAgents() {
  ['agentWorldBuilder', 'agentCleaner', 'agentPlotPlanner', 'agentAnalyzer', 'agentPredictor', 'agentWriter', 'agentCritic'].forEach(id => {
    document.getElementById(id).className = 'agent-step';
  });
}

// ── Actions ──────────────────────────────────────────────────────────────────

document.getElementById('copyBtn').addEventListener('click', () => {
  if (selectedIdx < 0) return;
  const isSource = selectedIdx < sourceFiles.length;
  const data = isSource ? sourceFiles[selectedIdx] : chapters[selectedIdx - sourceFiles.length];
  navigator.clipboard.writeText(data.content || data.text || '');
});

document.getElementById('exportBtn').addEventListener('click', () => {
  if (selectedIdx < 0) return;
  const isSource = selectedIdx < sourceFiles.length;
  const data = isSource ? sourceFiles[selectedIdx] : chapters[selectedIdx - sourceFiles.length];
  const blob = new Blob([data.content || data.text || ''], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (data.name || 'chapter') + '.txt';
  a.click();
});

const modalOverlay = document.getElementById('modalOverlay');
const modalInput = document.getElementById('modalInput');

document.getElementById('exportEpubBtn').addEventListener('click', () => {
  modalInput.value = '';
  modalOverlay.classList.add('open');
  modalInput.focus();
});

document.getElementById('modalCancel').addEventListener('click', () => {
  modalOverlay.classList.remove('open');
});

modalOverlay.addEventListener('click', (e) => {
  if (e.target === modalOverlay) modalOverlay.classList.remove('open');
});

modalInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') doExport();
  if (e.key === 'Escape') modalOverlay.classList.remove('open');
});

document.getElementById('modalConfirm').addEventListener('click', doExport);

async function doExport() {
  const title = modalInput.value.trim() || 'Untitled';
  modalOverlay.classList.remove('open');

  const allItems = [
    ...sourceFiles.map(f => ({ text: f.content })),
    ...chapters.map(c => ({ text: c.text })),
  ];

  const res = await apiFetch('/api/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, chapters: allItems })
  });

  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${title}.epub`;
  a.click();
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

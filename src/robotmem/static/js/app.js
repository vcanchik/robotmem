/* robotmem Web UI — 全局初始化 + Tab 切换 + 主题 */

// ── API 封装 ──

const API = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
        return res.json();
    },
    async del(url, body) {
        const res = await fetch(url, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
        return res.json();
    },
    async put(url, body) {
        const res = await fetch(url, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
        return res.json();
    },
};

// ── Tab 切换 ──

function switchTab(target) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    var el = document.getElementById('tab-' + target);
    if (el) el.classList.add('active');

    var tabBtn = document.querySelector('.tab[data-tab="' + target + '"]');
    if (tabBtn) tabBtn.classList.add('active');

    if (target === 'dashboard') loadDashboard();
    else if (target === 'memories') loadMemories();
    else if (target === 'sessions') loadSessions();
    else if (target === 'doctor') loadDoctor();
}

function initTabs() {
    document.querySelectorAll('.tab[data-tab]').forEach(function(tab) {
        tab.addEventListener('click', function() {
            switchTab(tab.dataset.tab);
        });
    });
}

// ── 主题切换 ──

function initTheme() {
    const btn = document.getElementById('theme-toggle');
    const saved = localStorage.getItem('robotmem-theme');
    if (saved === 'dark') setTheme('dark');

    btn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        setTheme(current === 'dark' ? 'light' : 'dark');
    });
}

function setTheme(theme) {
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        document.querySelector('.icon-sun').style.display = 'none';
        document.querySelector('.icon-moon').style.display = '';
    } else {
        document.documentElement.removeAttribute('data-theme');
        document.querySelector('.icon-sun').style.display = '';
        document.querySelector('.icon-moon').style.display = 'none';
    }
    localStorage.setItem('robotmem-theme', theme);
}

// ── Dashboard ──

async function loadDashboard() {
    try {
        const [stats, colls, failures] = await Promise.all([
            API.get('/api/stats'),
            API.get('/api/collections'),
            API.get('/api/recent-failures?limit=5'),
        ]);

        document.getElementById('stat-total').textContent = stats.total || 0;
        document.getElementById('stat-facts').textContent = stats.by_type?.fact || 0;
        document.getElementById('stat-perceptions').textContent = stats.by_type?.perception || 0;
        document.getElementById('stat-recent').textContent = stats.recent_24h || 0;

        // Category distribution
        const catEl = document.getElementById('category-list');
        catEl.innerHTML = '';
        if (stats.by_category) {
            for (const [cat, count] of Object.entries(stats.by_category)) {
                catEl.innerHTML += `<div class="kv-item">
                    <span class="kv-key">${esc(cat)}</span>
                    <span class="kv-value">${count}</span>
                </div>`;
            }
        }
        if (!catEl.innerHTML) catEl.innerHTML = '<div class="kv-item"><span class="kv-key">No data</span></div>';

        // Collections
        const collEl = document.getElementById('collection-list');
        collEl.innerHTML = '';
        if (colls.collections?.length) {
            for (const c of colls.collections) {
                collEl.innerHTML += `<div class="kv-item">
                    <span class="kv-key">${esc(c.name)}</span>
                    <span class="kv-value">${c.count}</span>
                </div>`;
            }
        }
        if (!collEl.innerHTML) collEl.innerHTML = '<div class="kv-item"><span class="kv-key">No data</span></div>';

        // Recent Failures
        const failEl = document.getElementById('recent-failures');
        failEl.innerHTML = '';
        if (failures.failures?.length) {
            for (const f of failures.failures) {
                const display = f.human_summary || f.content || '';
                const badgeClass = f.category === 'postmortem' ? 'badge-postmortem' : 'badge-gotcha';
                failEl.innerHTML += `<div class="failure-item failure-item-${safeCssClass(f.category)}" onclick="showMemoryDetail(${f.id})">
                    <div class="failure-item-header">
                        <span class="memory-item-id">#${f.id}</span>
                        <span class="badge ${badgeClass}">${esc(f.category)}</span>
                        <span class="failure-time">${formatTime(f.created_at)}</span>
                    </div>
                    <div class="failure-item-content">${esc(display)}</div>
                </div>`;
            }
        } else {
            failEl.innerHTML = '<div class="kv-item"><span class="kv-key">No recent failures</span></div>';
        }
    } catch (e) {
        console.error('loadDashboard:', e);
    }
}

// ── Doctor ──

async function loadDoctor() {
    try {
        const d = await API.get('/api/doctor');
        if (!d?.fts5 || !d?.vec0 || !d?.zero_hit) return;

        // FTS5
        setHealthIcon('health-fts5-icon', 'health-fts5-detail', d.fts5);
        // vec0
        setHealthIcon('health-vec0-icon', 'health-vec0-detail', d.vec0);

        // Zero-hit rate
        const zeroIcon = document.getElementById('health-zero-icon');
        const zeroDetail = document.getElementById('health-zero-detail');
        zeroDetail.textContent = d.zero_hit.rate + '% (' + d.zero_hit.count + '/' + d.zero_hit.total + ')';
        if (d.zero_hit.rate > 80) {
            zeroIcon.textContent = '\u26A0';
            zeroIcon.className = 'health-icon health-icon-warn';
        } else {
            zeroIcon.textContent = '\u2705';
            zeroIcon.className = 'health-icon health-icon-ok';
        }

        // Storage
        setText('doctor-memories', d.memories.total);
        setText('doctor-facts', d.memories.by_type?.fact || 0);
        setText('doctor-perceptions', d.memories.by_type?.perception || 0);
        const sessTotal = d.sessions.total;
        const sessActive = d.sessions.by_status?.active || 0;
        const sessEnded = d.sessions.by_status?.ended || 0;
        setText('doctor-sessions', sessTotal + ' (active: ' + sessActive + ', ended: ' + sessEnded + ')');
        setText('health-db-size', formatBytes(d.db_size_bytes));
    } catch (e) {
        console.error('loadDoctor:', e);
    }
}

function setHealthIcon(iconId, detailId, data) {
    const icon = document.getElementById(iconId);
    const detail = document.getElementById(detailId);
    if (data.ok === true) {
        icon.textContent = '\u2705';
        icon.className = 'health-icon health-icon-ok';
        detail.textContent = data.indexed + '/' + data.expected;
    } else if (data.ok === false) {
        icon.textContent = '\u274C';
        icon.className = 'health-icon health-icon-err';
        detail.textContent = data.indexed + '/' + data.expected;
    } else {
        icon.textContent = '\u2796';
        icon.className = 'health-icon health-icon-neutral';
        detail.textContent = 'N/A';
    }
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function formatBytes(b) {
    if (b === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(b) / Math.log(1024));
    return (b / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
}

// ── Sessions ──

let sessionPage = 0;
let openSessionId = null;

async function loadSessions(page) {
    if (page !== undefined) sessionPage = page;
    openSessionId = null;
    const el = document.getElementById('session-list');
    el.innerHTML = '<div class="loading">Loading...</div>';

    try {
        const data = await API.get(`/api/sessions?page=${sessionPage}&limit=20`);

        if (!data.sessions?.length) {
            el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">&#128221;</div><div class="empty-state-text">No sessions yet</div></div>';
            document.getElementById('session-pagination').innerHTML = '';
            return;
        }

        el.innerHTML = data.sessions.map(s => `
            <div class="session-item" data-external-id="${esc(s.external_id)}">
                <div class="session-item-header">
                    <span class="session-id">${esc(s.external_id || '-')}</span>
                    <span class="badge ${s.status === 'active' ? 'badge-fact' : ''}">${esc(s.status)}</span>
                </div>
                <div class="session-meta">
                    <span>Collection: ${esc(s.collection)}</span>
                    <span>Memories: ${s.memory_count}</span>
                    <span>Sessions: ${s.session_count}</span>
                    <span>${formatTime(s.created_at)}</span>
                </div>
            </div>
        `).join('');

        renderPagination('session-pagination', sessionPage, data.total, 20, loadSessions);
    } catch (e) {
        el.innerHTML = '<div class="empty-state"><div class="empty-state-text">Failed to load sessions</div></div>';
    }
}

// ── Session Timeline ──

async function toggleSessionTimeline(externalId, sessionEl) {
    // 关闭已打开的
    const existing = document.querySelector('.session-timeline');
    if (existing) existing.remove();

    if (openSessionId === externalId) {
        openSessionId = null;
        return;
    }
    openSessionId = externalId;

    const timeline = document.createElement('div');
    timeline.className = 'session-timeline';
    timeline.innerHTML = '<div class="loading">Loading timeline...</div>';
    sessionEl.after(timeline);

    try {
        const data = await API.get(`/api/sessions/${encodeURIComponent(externalId)}/memories`);
        if (!data.memories?.length) {
            timeline.innerHTML = '<div class="timeline-empty">No memories in this session</div>';
            return;
        }
        timeline.innerHTML = data.memories.map(m => {
            const display = m.human_summary || (m.content || '').slice(0, 120);
            const dotClass = m.type === 'fact' ? 'timeline-dot-fact' : 'timeline-dot-perception';
            const typeClass = m.type === 'fact' ? 'badge-fact' : 'badge-perception';
            const time = m.created_at ? new Date(m.created_at + 'Z').toLocaleTimeString() : '';
            return `<div class="timeline-item" onclick="event.stopPropagation(); showMemoryDetail(${m.id})">
                <div class="timeline-dot ${dotClass}"></div>
                <div class="timeline-content">
                    <span class="timeline-time">${time}</span>
                    <span class="badge ${typeClass}">${esc(m.type)}</span>
                    ${m.category ? `<span class="badge">${esc(m.category)}</span>` : ''}
                    <div class="timeline-text">${esc(display)}</div>
                </div>
            </div>`;
        }).join('');
        if (data.total > 20) {
            const moreBtn = document.createElement('div');
            moreBtn.className = 'timeline-more';
            moreBtn.textContent = `Show all ${data.total} memories`;
            moreBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                loadFullTimeline(externalId);
            });
            timeline.appendChild(moreBtn);
        }
    } catch (e) {
        timeline.innerHTML = '<div class="timeline-empty">Failed to load timeline</div>';
    }
}

async function loadFullTimeline(externalId) {
    const timeline = document.querySelector('.session-timeline');
    if (!timeline) return;
    timeline.innerHTML = '<div class="loading">Loading all memories...</div>';
    try {
        const data = await API.get(`/api/sessions/${encodeURIComponent(externalId)}/memories?limit=100`);
        if (!data.memories?.length) {
            timeline.innerHTML = '<div class="timeline-empty">No memories</div>';
            return;
        }
        timeline.innerHTML = data.memories.map(m => {
            const display = m.human_summary || (m.content || '').slice(0, 120);
            const dotClass = m.type === 'fact' ? 'timeline-dot-fact' : 'timeline-dot-perception';
            const typeClass = m.type === 'fact' ? 'badge-fact' : 'badge-perception';
            const time = m.created_at ? new Date(m.created_at + 'Z').toLocaleTimeString() : '';
            return `<div class="timeline-item" onclick="event.stopPropagation(); showMemoryDetail(${m.id})">
                <div class="timeline-dot ${dotClass}"></div>
                <div class="timeline-content">
                    <span class="timeline-time">${time}</span>
                    <span class="badge ${typeClass}">${esc(m.type)}</span>
                    ${m.category ? `<span class="badge">${esc(m.category)}</span>` : ''}
                    <div class="timeline-text">${esc(display)}</div>
                </div>
            </div>`;
        }).join('');
    } catch (e) {
        timeline.innerHTML = '<div class="timeline-empty">Failed to load timeline</div>';
    }
}

// ── Helpers ──

function esc(s) {
    if (s == null) return '';
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}

function safeCssClass(s) {
    if (s == null) return '';
    return String(s).replace(/[^a-z0-9_-]/gi, '-').toLowerCase();
}

function formatTime(t) {
    if (!t) return '-';
    try {
        const d = new Date(t + 'Z');
        return d.toLocaleString();
    } catch { return t; }
}

function renderPagination(elId, currentPage, total, limit, loadFn) {
    const pages = Math.ceil(total / limit);
    const el = document.getElementById(elId);
    if (pages <= 1) { el.innerHTML = ''; return; }

    el.innerHTML = '';
    const prev = document.createElement('button');
    prev.disabled = currentPage === 0;
    prev.textContent = 'Prev';
    prev.addEventListener('click', () => loadFn(currentPage - 1));

    const info = document.createElement('span');
    info.className = 'page-info';
    info.textContent = `${currentPage + 1} / ${pages}`;

    const next = document.createElement('button');
    next.disabled = currentPage >= pages - 1;
    next.textContent = 'Next';
    next.addEventListener('click', () => loadFn(currentPage + 1));

    el.append(prev, info, next);
}

// ── Init ──

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initTheme();
    loadDashboard();

    // Session 点击事件委托（避免 inline onclick XSS）
    const sessionList = document.getElementById('session-list');
    if (sessionList) {
        sessionList.addEventListener('click', (e) => {
            const item = e.target.closest('.session-item');
            if (item && item.dataset.externalId) {
                toggleSessionTimeline(item.dataset.externalId, item);
            }
        });
    }
});

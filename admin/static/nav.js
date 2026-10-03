/*
 * Shared persistent navbar for every admin screen.
 *
 * Single source of truth for the page list: to add a new agent page, add one
 * entry to PAGES below (and its route in admin/app.py). The bar scrolls
 * horizontally when there are more items than fit, so it scales to many agents.
 *
 * Each page includes:  <nav class="main-nav" id="main-nav"></nav>
 *                      <script src="/static/nav.js" defer></script>
 */
(function () {
  'use strict';

  var PAGES = [
    { path: '/',            label: 'Inicio' },
    { path: '/propositor',  label: 'Propositor' },
    { path: '/archivero',   label: 'Archivero' },
    { path: '/verificator',  label: 'Verificator' },
    { path: '/mapper',       label: 'Mapper' },
    { path: '/cast-manager', label: 'Cast Manager' },
    { path: '/cast-director', label: 'Cast Director' },
    { path: '/locations',    label: 'Lugares' },
    { path: '/performance',  label: 'Performance' }
    /* Agentes futuros: añadir aquí. */
  ];

  var css = [
    /* Right-aligned via margin-left:auto on the first item — unlike
       justify-content:flex-end, this keeps the overflow start reachable, so
       the first links can always be scrolled back into view. */
    '.main-nav { display: flex; gap: 4px; align-items: center; flex: 1; min-width: 0;',
    '  overflow-x: auto; scrollbar-width: thin; padding: 2px 0 4px;',
    '  overscroll-behavior-x: contain; }',
    '.main-nav a:first-child { margin-left: auto; }',
    '.main-nav::-webkit-scrollbar { height: 6px; }',
    '.main-nav::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 3px; }',
    '.main-nav::-webkit-scrollbar-thumb:hover { background: #555; }',
    '.main-nav a { color: var(--text-mid, #888); font-size: 10px; letter-spacing: 1px;',
    '  text-decoration: none; padding: 3px 10px; border: 1px solid var(--border, #252525);',
    '  border-radius: 2px; white-space: nowrap; flex-shrink: 0;',
    '  transition: color .15s, border-color .15s; }',
    '.main-nav a:hover { color: var(--accent, #f59e0b); border-color: var(--accent, #f59e0b); }',
    '.main-nav a.active { color: var(--accent, #f59e0b); border-color: var(--accent-dim, #78350f);',
    '  background: rgba(245, 158, 11, .08); }'
  ].join('\n');

  var style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  var nav = document.getElementById('main-nav');
  if (!nav) return;

  var here = location.pathname.replace(/\/+$/, '') || '/';
  PAGES.forEach(function (page) {
    var a = document.createElement('a');
    a.href = page.path;
    a.textContent = page.label;
    if (page.path === here) a.className = 'active';
    nav.appendChild(a);
  });

  // Mouse wheel scrolls the bar horizontally (a vertical wheel does nothing
  // on an overflow-x container by default, leaving mouse users stranded).
  nav.addEventListener('wheel', function (ev) {
    if (nav.scrollWidth <= nav.clientWidth) return;
    if (Math.abs(ev.deltaY) > Math.abs(ev.deltaX)) {
      nav.scrollLeft += ev.deltaY;
      ev.preventDefault();
    }
  }, { passive: false });

  // Land with the current page's link in view.
  var active = nav.querySelector('a.active');
  if (active) active.scrollIntoView({ block: 'nearest', inline: 'center' });
})();

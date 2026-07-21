/*
 * Mold Studio icon registry — line icons on a 24-unit grid, 1.7–1.8 stroke,
 * round caps, drawn with currentColor (spec §02). Inner-SVG markup only;
 * rendered by components/Icon.vue. Add missing glyphs at the same weight
 * (Lucide-compatible geometry) — never mix fills or weights.
 */

export const ICONS = {
  // Workspaces
  create:
    '<path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9z"/><path d="M19 4.5v3M20.5 6h-3"/>',
  sparkle:
    '<path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9z"/>',
  library:
    '<rect x="3" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5"/>',
  models:
    '<path d="M12 3l8.5 4.3-8.5 4.3L3.5 7.3z"/><path d="M3.5 12L12 16.3 20.5 12M3.5 16.5L12 20.8l8.5-4.3"/>',
  machines:
    '<rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><circle cx="7" cy="7.5" r="1"/><circle cx="7" cy="16.5" r="1"/>',
  settings: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9M14 4v6M8 14v6"/>',

  // Navigation / chrome
  sidebar:
    '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  sliders: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  "chevron-down": '<path d="M6 9l6 6 6-6"/>',
  "chevron-up": '<path d="M6 15l6-6 6 6"/>',
  "chevron-right": '<path d="M9 6l6 6-6 6"/>',
  "chevron-left": '<path d="M15 6l-6 6 6 6"/>',
  "arrow-right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  check: '<path d="M4 12.5l5 5 11-11"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',

  // Media / actions
  image:
    '<rect x="3" y="3" width="18" height="18" rx="2.5"/><circle cx="8.5" cy="8.5" r="1.8"/><path d="M21 15l-5-5L5 21"/>',
  video:
    '<rect x="3" y="5" width="13" height="14" rx="2.5"/><path d="M16 10.5l5-3v9l-5-3z"/>',
  play: '<path d="M7 5l12 7-12 7z"/>',
  download: '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
  upload:
    '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/>',
  save: '<path d="M4 12v7a1 1 0 001 1h14a1 1 0 001-1v-7"/><path d="M12 15V3M8 7l4-4 4 4"/>',
  reuse:
    '<path d="M3 12a9 9 0 019-9 9 9 0 016.7 3M21 3v6h-6"/><path d="M21 12a9 9 0 01-9 9 9 9 0 01-6.7-3M3 21v-6h6"/>',
  reroll: '<path d="M3 12a9 9 0 0114-7l3 3M21 3v6h-6"/>',
  refresh:
    '<path d="M3 12a9 9 0 0 1 15.5-6.3L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.3L3 16"/><path d="M3 21v-5h5"/>',
  trash:
    '<path d="M4 7h16M10 11v6M14 11v6"/><path d="M6 7l1 13a1 1 0 001 1h8a1 1 0 001-1l1-13"/><path d="M9 7V4a1 1 0 011-1h4a1 1 0 011 1v3"/>',
  star: '<path d="M12 3l2.7 5.9 6.3.7-4.7 4.3 1.3 6.1L12 16.9 6.4 20l1.3-6.1L3 9.6l6.3-.7z"/>',

  // Generate / advanced
  scheduler:
    '<path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/>',
  negative: '<circle cx="12" cy="12" r="9"/><path d="M6 6l12 12"/>',
  layers: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  upscale: '<path d="M4 14v6h6M20 10V4h-6"/><path d="M14 10l6-6M10 14l-6 6"/>',
  output: '<path d="M4 4h16v16H4z"/><path d="M4 15l4-4 4 4 3-3 5 5"/>',
  mask: '<path d="M12 3s7 3 7 9-7 9-7 9-7-3-7-9 7-9 7-9z"/>',

  // Machines / connect
  cloud: '<path d="M17 18a4 4 0 000-8 6 6 0 00-11.6 1.5A3.5 3.5 0 006 18z"/>',
  wifi: '<path d="M5 13a10 10 0 0114 0M8.5 16.5a5 5 0 017 0M2 9.5a15 15 0 0120 0"/><circle cx="12" cy="20" r="1"/>',
  lock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 018 0v4"/>',
} as const;

export type IconName = keyof typeof ICONS;

export const ICON_NAMES = Object.keys(ICONS) as IconName[];

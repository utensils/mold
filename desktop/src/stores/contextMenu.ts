import { defineStore } from "pinia";

export interface MenuItem {
  label: string;
  /** Styled with Stop — destructive actions. */
  danger?: boolean;
  disabled?: boolean;
  action?: () => void;
  /** Checkable item (`menuitemcheckbox`): renders a ✓ column. `undefined`
   *  means "not checkable", which is different from `false`. */
  checked?: boolean;
  /** Submenu entries — opened on hover / ArrowRight / click, closed by
   *  ArrowLeft / Escape. An item with children never runs `action`. */
  children?: MenuEntry[];
}

export interface MenuEntrySeparator {
  separator: true;
}

export type MenuEntry = MenuItem | MenuEntrySeparator;

export const isSeparator = (e: MenuEntry): e is MenuEntrySeparator => "separator" in e;

export const hasChildren = (e: MenuEntry): e is MenuItem & { children: MenuEntry[] } =>
  !isSeparator(e) && Array.isArray(e.children) && e.children.length > 0;

/** True when any entry of a menu is checkable — the ✓ column is reserved
 *  for the whole menu so labels stay aligned. */
export const hasCheckColumn = (entries: MenuEntry[]): boolean =>
  entries.some((e) => !isSeparator(e) && e.checked !== undefined);

/** Estimated geometry used to keep the menu inside the window. */
export const ITEM_HEIGHT = 26;
export const SEPARATOR_HEIGHT = 9;
export const MENU_WIDTH = 224;
/** A submenu overlaps its parent's edge by this much so the pointer can
 *  cross without a gap. */
export const SUBMENU_OVERLAP = 2;
/** Vertical padding of the menu panel (`py-1`): a submenu's first item lines
 *  up with its parent item when the panel starts this far above it. */
const MENU_PADDING_Y = 4;
const MARGIN = 6;

function estimatedHeight(entries: MenuEntry[]): number {
  return (
    entries.reduce((h, e) => h + (isSeparator(e) ? SEPARATOR_HEIGHT : ITEM_HEIGHT), 0) +
    MENU_PADDING_Y * 2 +
    2
  );
}

/** Pure position clamp — exported for tests. */
export function clampPosition(
  x: number,
  y: number,
  entries: MenuEntry[],
  viewport: { width: number; height: number },
): { x: number; y: number } {
  const width = Math.min(MENU_WIDTH, Math.max(0, viewport.width - MARGIN * 2));
  const height = Math.min(estimatedHeight(entries), Math.max(0, viewport.height - MARGIN * 2));
  return {
    x: Math.max(MARGIN, Math.min(x, viewport.width - width - MARGIN)),
    y: Math.max(MARGIN, Math.min(y, viewport.height - height - MARGIN)),
  };
}

/** Where a parent item sits on screen — what a submenu anchors to. */
export interface SubmenuAnchor {
  left: number;
  right: number;
  top: number;
}

/**
 * Pure submenu placement — exported for tests. Opens to the right of the
 * parent item (overlapping by `SUBMENU_OVERLAP`), flips to the left when the
 * right side cannot fit a full menu, and shares `clampPosition`'s vertical
 * rule so a tall submenu near the bottom edge slides up.
 */
export function submenuPosition(
  anchor: SubmenuAnchor,
  children: MenuEntry[],
  viewport: { width: number; height: number },
): { x: number; y: number } {
  const width = Math.min(MENU_WIDTH, Math.max(0, viewport.width - MARGIN * 2));
  const rightX = anchor.right - SUBMENU_OVERLAP;
  const fitsRight = rightX + width + MARGIN <= viewport.width;
  const x = fitsRight ? rightX : Math.max(MARGIN, anchor.left - width + SUBMENU_OVERLAP);
  const { y } = clampPosition(x, anchor.top - MENU_PADDING_Y, children, viewport);
  return { x: Math.max(MARGIN, Math.min(x, viewport.width - width - MARGIN)), y };
}

export interface OpenSubmenu {
  /** Index of the parent entry in `entries`. */
  parentIndex: number;
  entries: MenuEntry[];
  x: number;
  y: number;
  /** Keyboard cursor inside the submenu; -1 = none. */
  highlighted: number;
}

function firstActionable(entries: MenuEntry[]): number {
  return entries.findIndex((e) => !isSeparator(e) && !e.disabled);
}

function nextActionable(entries: MenuEntry[], current: number, delta: 1 | -1): number {
  const actionable = entries
    .map((e, i) => ({ e, i }))
    .filter(({ e }) => !isSeparator(e) && !e.disabled);
  if (actionable.length === 0) return -1;
  const at = actionable.findIndex(({ i }) => i === current);
  return actionable[(at + delta + actionable.length) % actionable.length]!.i;
}

/** One app-wide custom context menu (replaces the WebKit default). */
export const useContextMenuStore = defineStore("contextMenu", {
  state: () => ({
    visible: false,
    x: 0,
    y: 0,
    entries: [] as MenuEntry[],
    /** Keyboard cursor over actionable items; -1 = none. */
    highlighted: -1,
    /** The one open submenu (nesting is one level deep by design). */
    submenu: null as OpenSubmenu | null,
  }),
  actions: {
    open(event: MouseEvent, entries: MenuEntry[]) {
      event.preventDefault();
      event.stopPropagation();
      if (entries.length === 0) return;
      const pos = clampPosition(event.clientX, event.clientY, entries, {
        width: window.innerWidth,
        height: window.innerHeight,
      });
      this.entries = entries;
      this.x = pos.x;
      this.y = pos.y;
      this.highlighted = -1;
      this.submenu = null;
      this.visible = true;
    },
    close() {
      this.visible = false;
      this.entries = [];
      this.highlighted = -1;
      this.submenu = null;
    },
    /** Run an item. A parent item opens its submenu instead of closing the
     *  menu (it needs an anchor, so the component passes one when it has
     *  it; keyboard callers reach `enterSubmenu`). */
    activate(entry: MenuEntry, anchor?: SubmenuAnchor) {
      if (isSeparator(entry) || entry.disabled) return;
      if (hasChildren(entry)) {
        const index = this.entries.indexOf(entry);
        if (index !== -1) this.openSubmenu(index, anchor);
        return;
      }
      // Close first: actions may open dialogs or push routes.
      this.close();
      entry.action?.();
    },
    /** Open the submenu of the root entry at `index`. Without an anchor
     *  (keyboard before the DOM measured) it sits beside the parent menu at
     *  the estimated item offset. */
    openSubmenu(index: number, anchor?: SubmenuAnchor) {
      const entry = this.entries[index];
      if (!entry || !hasChildren(entry) || entry.disabled) return;
      if (this.submenu?.parentIndex === index) return;
      const resolved: SubmenuAnchor = anchor ?? {
        left: this.x,
        right: this.x + MENU_WIDTH,
        top: this.y + MENU_PADDING_Y + this.estimatedItemOffset(index),
      };
      const pos = submenuPosition(resolved, entry.children, {
        width: window.innerWidth,
        height: window.innerHeight,
      });
      this.highlighted = index;
      this.submenu = {
        parentIndex: index,
        entries: entry.children,
        x: pos.x,
        y: pos.y,
        highlighted: -1,
      };
    },
    closeSubmenu() {
      this.submenu = null;
    },
    /** Keyboard: ArrowRight on a parent item opens it and lands on its first
     *  enabled child. */
    enterSubmenu() {
      if (this.highlighted < 0) return;
      this.openSubmenu(this.highlighted);
      if (this.submenu) this.submenu.highlighted = firstActionable(this.submenu.entries);
    },
    /** Keyboard: ArrowLeft / Escape inside a submenu returns to its parent. */
    leaveSubmenu() {
      if (!this.submenu) return;
      this.highlighted = this.submenu.parentIndex;
      this.submenu = null;
    },
    /** Activate the keyboard-highlighted item (Enter). */
    activateHighlighted() {
      if (this.submenu) {
        const child = this.submenu.entries[this.submenu.highlighted];
        if (child) this.activate(child);
        return;
      }
      const entry = this.entries[this.highlighted];
      if (!entry) return;
      if (hasChildren(entry)) this.enterSubmenu();
      else this.activate(entry);
    },
    /** Move the keyboard cursor over enabled items, wrapping — inside the
     *  open submenu when there is one, else over the root entries. */
    move(delta: 1 | -1) {
      if (this.submenu) {
        const next = nextActionable(this.submenu.entries, this.submenu.highlighted, delta);
        if (next !== -1) this.submenu.highlighted = next;
        return;
      }
      const next = nextActionable(this.entries, this.highlighted, delta);
      if (next !== -1) this.highlighted = next;
    },
    /** Estimated pixel offset of entry `index` from the top of the root
     *  panel's content (the keyboard fallback anchor). */
    estimatedItemOffset(index: number): number {
      let offset = 0;
      for (let i = 0; i < index && i < this.entries.length; i++) {
        offset += isSeparator(this.entries[i]!) ? SEPARATOR_HEIGHT : ITEM_HEIGHT;
      }
      return offset;
    },
  },
});

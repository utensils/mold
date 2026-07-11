import { defineStore } from "pinia";

export interface MenuItem {
  label: string;
  /** Styled with Stop — destructive actions. */
  danger?: boolean;
  disabled?: boolean;
  action?: () => void;
}

export interface MenuEntrySeparator {
  separator: true;
}

export type MenuEntry = MenuItem | MenuEntrySeparator;

export const isSeparator = (e: MenuEntry): e is MenuEntrySeparator => "separator" in e;

/** Estimated geometry used to keep the menu inside the window. */
export const ITEM_HEIGHT = 26;
export const SEPARATOR_HEIGHT = 9;
export const MENU_WIDTH = 224;
const MARGIN = 6;

/** Pure position clamp — exported for tests. */
export function clampPosition(
  x: number,
  y: number,
  entries: MenuEntry[],
  viewport: { width: number; height: number },
): { x: number; y: number } {
  const width = Math.min(MENU_WIDTH, Math.max(0, viewport.width - MARGIN * 2));
  const estimatedHeight =
    entries.reduce((h, e) => h + (isSeparator(e) ? SEPARATOR_HEIGHT : ITEM_HEIGHT), 0) + 10;
  const height = Math.min(estimatedHeight, Math.max(0, viewport.height - MARGIN * 2));
  return {
    x: Math.max(MARGIN, Math.min(x, viewport.width - width - MARGIN)),
    y: Math.max(MARGIN, Math.min(y, viewport.height - height - MARGIN)),
  };
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
      this.visible = true;
    },
    close() {
      this.visible = false;
      this.entries = [];
      this.highlighted = -1;
    },
    activate(entry: MenuEntry) {
      if (isSeparator(entry) || entry.disabled) return;
      // Close first: actions may open dialogs or push routes.
      this.close();
      entry.action?.();
    },
    /** Move the keyboard cursor over enabled items, wrapping. */
    move(delta: 1 | -1) {
      const actionable = this.entries
        .map((e, i) => ({ e, i }))
        .filter(({ e }) => !isSeparator(e) && !e.disabled);
      if (actionable.length === 0) return;
      const current = actionable.findIndex(({ i }) => i === this.highlighted);
      const next = actionable[(current + delta + actionable.length) % actionable.length]!;
      this.highlighted = next.i;
    },
  },
});

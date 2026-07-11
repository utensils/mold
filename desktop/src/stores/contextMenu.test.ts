import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import {
  clampPosition,
  ITEM_HEIGHT,
  MENU_WIDTH,
  useContextMenuStore,
  type MenuEntry,
} from "./contextMenu";

const items = (n: number, overrides: Partial<MenuEntry>[] = []): MenuEntry[] =>
  Array.from({ length: n }, (_, i) => ({
    label: `item ${i}`,
    action: vi.fn(),
    ...(overrides[i] ?? {}),
  }));

describe("clampPosition", () => {
  const viewport = { width: 1000, height: 600 };

  it("keeps the menu inside the right and bottom edges", () => {
    const pos = clampPosition(990, 590, items(4), viewport);
    expect(pos.x).toBeLessThanOrEqual(1000 - MENU_WIDTH);
    expect(pos.y).toBeLessThanOrEqual(600 - 4 * ITEM_HEIGHT);
  });

  it("passes through positions that already fit", () => {
    expect(clampPosition(100, 100, items(2), viewport)).toEqual({ x: 100, y: 100 });
  });

  it("stays usable when the scaled viewport is narrower than the normal menu", () => {
    expect(clampPosition(300, 300, items(20), { width: 180, height: 140 })).toEqual({
      x: 6,
      y: 6,
    });
  });
});

describe("context menu store", () => {
  beforeEach(() => setActivePinia(createPinia()));

  const openEvent = () =>
    ({
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
      clientX: 100,
      clientY: 100,
    }) as unknown as MouseEvent;

  it("open shows entries and prevents the native menu", () => {
    const menu = useContextMenuStore();
    const e = openEvent();
    menu.open(e, items(3));
    expect(menu.visible).toBe(true);
    expect(menu.entries).toHaveLength(3);
    expect(e.preventDefault).toHaveBeenCalled();
  });

  it("activate runs the action and closes; disabled items are inert", () => {
    const menu = useContextMenuStore();
    const entries = items(2, [{}, { disabled: true }]);
    menu.open(openEvent(), entries);
    menu.activate(entries[1]!);
    expect(menu.visible).toBe(true); // disabled: still open, nothing ran
    menu.activate(entries[0]!);
    expect(
      (entries[0] as unknown as { action: ReturnType<typeof vi.fn> }).action,
    ).toHaveBeenCalled();
    expect(menu.visible).toBe(false);
  });

  it("keyboard cursor skips separators and disabled items, wrapping", () => {
    const menu = useContextMenuStore();
    const entries: MenuEntry[] = [
      { label: "a", action: vi.fn() },
      { separator: true },
      { label: "b", disabled: true },
      { label: "c", action: vi.fn() },
    ];
    menu.open(openEvent(), entries);
    menu.move(1);
    expect(menu.highlighted).toBe(0);
    menu.move(1);
    expect(menu.highlighted).toBe(3);
    menu.move(1); // wraps
    expect(menu.highlighted).toBe(0);
    menu.move(-1);
    expect(menu.highlighted).toBe(3);
  });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import {
  clampPosition,
  hasCheckColumn,
  hasChildren,
  ITEM_HEIGHT,
  MENU_WIDTH,
  SUBMENU_OVERLAP,
  submenuPosition,
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

describe("submenuPosition", () => {
  const viewport = { width: 1000, height: 600 };
  const anchor = { left: 100, right: 324, top: 140 };

  it("opens to the right of the parent item, overlapping its edge, aligned with the item", () => {
    expect(submenuPosition(anchor, items(3), viewport)).toEqual({
      x: 324 - SUBMENU_OVERLAP,
      y: 140 - 4,
    });
  });

  it("flips to the left when the right side cannot fit a full menu", () => {
    const nearRight = { left: 820, right: 1044 - 100, top: 140 };
    const pos = submenuPosition(nearRight, items(3), viewport);
    expect(pos.x).toBe(820 - MENU_WIDTH + SUBMENU_OVERLAP);
  });

  it("slides a tall submenu up so it stays inside the bottom edge", () => {
    const pos = submenuPosition({ left: 100, right: 324, top: 560 }, items(10), viewport);
    expect(pos.y + 10 * ITEM_HEIGHT).toBeLessThanOrEqual(600);
  });
});

describe("submenus + checkable items", () => {
  beforeEach(() => setActivePinia(createPinia()));

  const openEvent = () =>
    ({
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
      clientX: 100,
      clientY: 100,
    }) as unknown as MouseEvent;

  const childAction = vi.fn();
  const tree = (): MenuEntry[] => [
    { label: "Reuse settings", action: vi.fn() },
    { separator: true },
    {
      label: "Tags",
      children: [
        { label: "blue", checked: true, action: childAction },
        { label: "keep", checked: false, action: vi.fn() },
        { separator: true },
        { label: "New tag…", action: vi.fn() },
      ],
    },
    {
      label: "Add to collection",
      children: [{ label: "Smurfs", checked: false, action: vi.fn() }],
    },
    { label: "Delete", danger: true, action: vi.fn() },
  ];

  it("hasChildren / hasCheckColumn classify entries", () => {
    const entries = tree();
    expect(hasChildren(entries[2]!)).toBe(true);
    expect(hasChildren(entries[0]!)).toBe(false);
    expect(hasChildren({ label: "x", children: [] })).toBe(false);
    expect(hasCheckColumn(entries)).toBe(false);
    expect(hasCheckColumn((entries[2] as { children: MenuEntry[] }).children)).toBe(true);
  });

  it("activating a parent item opens its submenu instead of closing the menu", () => {
    const menu = useContextMenuStore();
    const entries = tree();
    menu.open(openEvent(), entries);
    menu.activate(entries[2]!, { left: 100, right: 324, top: 160 });
    expect(menu.visible).toBe(true);
    expect(menu.submenu).toMatchObject({ parentIndex: 2, x: 324 - SUBMENU_OVERLAP });
    expect(menu.submenu!.entries).toHaveLength(4);
    expect(menu.highlighted).toBe(2);
  });

  it("ArrowRight enters the highlighted parent's submenu on its first enabled child; ArrowLeft leaves", () => {
    const menu = useContextMenuStore();
    menu.open(openEvent(), tree());
    menu.move(1); // Reuse settings
    menu.move(1); // Tags
    expect(menu.highlighted).toBe(2);
    menu.enterSubmenu();
    expect(menu.submenu?.parentIndex).toBe(2);
    expect(menu.submenu?.highlighted).toBe(0);
    // Movement is scoped to the submenu while it is open.
    menu.move(1);
    expect(menu.submenu?.highlighted).toBe(1);
    menu.move(1); // skips the separator
    expect(menu.submenu?.highlighted).toBe(3);
    expect(menu.highlighted).toBe(2);
    menu.leaveSubmenu();
    expect(menu.submenu).toBeNull();
    expect(menu.highlighted).toBe(2);
  });

  it("Enter on a child runs it and closes everything; Enter on a parent opens it", () => {
    const menu = useContextMenuStore();
    menu.open(openEvent(), tree());
    menu.move(1);
    menu.move(1);
    menu.activateHighlighted(); // parent → opens
    expect(menu.submenu).not.toBeNull();
    menu.activateHighlighted(); // first child
    expect(childAction).toHaveBeenCalled();
    expect(menu.visible).toBe(false);
    expect(menu.submenu).toBeNull();
  });

  it("hovering a different parent swaps the submenu; a leaf closes it", () => {
    const menu = useContextMenuStore();
    menu.open(openEvent(), tree());
    menu.openSubmenu(2);
    menu.openSubmenu(3);
    expect(menu.submenu?.parentIndex).toBe(3);
    menu.closeSubmenu();
    expect(menu.submenu).toBeNull();
  });

  it("a disabled parent never opens", () => {
    const menu = useContextMenuStore();
    menu.open(openEvent(), [{ label: "Tags", disabled: true, children: [{ label: "x" }] }]);
    menu.openSubmenu(0);
    expect(menu.submenu).toBeNull();
  });

  it("open() and close() reset the submenu", () => {
    const menu = useContextMenuStore();
    menu.open(openEvent(), tree());
    menu.openSubmenu(2);
    menu.open(openEvent(), tree());
    expect(menu.submenu).toBeNull();
    menu.openSubmenu(2);
    menu.close();
    expect(menu.submenu).toBeNull();
  });
});

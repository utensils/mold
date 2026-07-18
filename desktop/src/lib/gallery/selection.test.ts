import { describe, expect, it } from "vitest";
import { applySelectionClick } from "./selection";

const order = ["a.png", "b.png", "c.png", "d.png", "e.png"];

describe("applySelectionClick", () => {
  it("a plain click toggles the item on and sets the anchor", () => {
    const next = applySelectionClick(new Set(), null, order, "b.png", {});
    expect([...next.selection]).toEqual(["b.png"]);
    expect(next.anchor).toBe("b.png");
  });

  it("a plain click toggles an already-selected item off", () => {
    const next = applySelectionClick(new Set(["b.png"]), "b.png", order, "b.png", {});
    expect(next.selection.size).toBe(0);
    expect(next.anchor).toBe("b.png");
  });

  it("shift-click selects the inclusive range from the anchor, keeping the anchor", () => {
    const next = applySelectionClick(new Set(["b.png"]), "b.png", order, "d.png", {
      shift: true,
    });
    expect([...next.selection].sort()).toEqual(["b.png", "c.png", "d.png"]);
    expect(next.anchor).toBe("b.png");
  });

  it("shift-click works backwards (clicked before the anchor)", () => {
    const next = applySelectionClick(new Set(["d.png"]), "d.png", order, "a.png", {
      shift: true,
    });
    expect([...next.selection].sort()).toEqual(["a.png", "b.png", "c.png", "d.png"]);
  });

  it("shift-click never clears selections outside the range (Finder-style)", () => {
    const next = applySelectionClick(new Set(["e.png", "a.png"]), "a.png", order, "b.png", {
      shift: true,
    });
    expect([...next.selection].sort()).toEqual(["a.png", "b.png", "e.png"]);
  });

  it("shift-click without a usable anchor falls back to a single toggle", () => {
    // Anchor vanished from the filtered order (filter changed underneath).
    const next = applySelectionClick(new Set(), "gone.png", order, "c.png", {
      shift: true,
    });
    expect([...next.selection]).toEqual(["c.png"]);
    expect(next.anchor).toBe("c.png");
  });

  it("meta-click toggles a single item without collapsing the rest", () => {
    const next = applySelectionClick(new Set(["a.png", "b.png"]), "a.png", order, "d.png", {
      meta: true,
    });
    expect([...next.selection].sort()).toEqual(["a.png", "b.png", "d.png"]);
    expect(next.anchor).toBe("d.png");
  });

  it("never mutates the incoming selection set", () => {
    const current = new Set(["a.png"]);
    applySelectionClick(current, "a.png", order, "c.png", { shift: true });
    expect([...current]).toEqual(["a.png"]);
  });
});

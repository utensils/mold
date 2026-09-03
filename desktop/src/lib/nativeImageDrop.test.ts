import { describe, expect, it } from "vitest";
import { dropTargetAtPosition, reduceNativeImageDrag } from "./nativeImageDrop";

describe("native image drop overlay", () => {
  it("does not arm for an internal filmstrip drag", () => {
    let state = { candidate: false, visible: false };
    state = reduceNativeImageDrag(state, { type: "over" });
    expect(state).toEqual({ candidate: false, visible: false });

    state = reduceNativeImageDrag(state, { type: "enter", paths: [] });
    state = reduceNativeImageDrag(state, { type: "over" });
    expect(state).toEqual({ candidate: false, visible: false });
  });

  it("stays visible for a native image drag until it leaves", () => {
    let state = reduceNativeImageDrag(
      { candidate: false, visible: false },
      { type: "enter", paths: ["/tmp/source.png"] },
    );
    state = reduceNativeImageDrag(state, { type: "over" });
    expect(state.visible).toBe(true);

    expect(reduceNativeImageDrag(state, { type: "leave" })).toEqual({
      candidate: false,
      visible: false,
    });
  });
});

describe("dropTargetAtPosition", () => {
  it("names the well under the cursor, in CSS pixels", () => {
    document.body.innerHTML = `
      <div id="well" data-drop-target="references">
        <img id="tile" />
      </div>`;
    const tile = document.getElementById("tile")!;
    const ratio = 2;
    Object.defineProperty(window, "devicePixelRatio", {
      configurable: true,
      value: ratio,
    });
    const seen: number[][] = [];
    document.elementFromPoint = (x: number, y: number) => {
      seen.push([x, y]);
      return tile;
    };

    // Tauri reports PHYSICAL pixels; the hit test happens in CSS pixels, and
    // a hit on a child resolves to its `data-drop-target` ancestor.
    expect(dropTargetAtPosition({ x: 240, y: 100 })).toBe("references");
    expect(seen).toEqual([[120, 50]]);
  });

  it("answers null for a drop on chrome or with no position at all", () => {
    document.body.innerHTML = `<div id="plain"></div>`;
    document.elementFromPoint = () => document.getElementById("plain");
    expect(dropTargetAtPosition({ x: 10, y: 10 })).toBeNull();
    expect(dropTargetAtPosition(null)).toBeNull();
    expect(dropTargetAtPosition(undefined)).toBeNull();
  });

  it("carries the position through the overlay reducer untouched", () => {
    // The reducer only decides overlay visibility; the position rides the
    // same payload so the bridge can hit-test the drop.
    const payload = {
      type: "drop" as const,
      paths: ["/tmp/a.png"],
      position: { x: 4, y: 5 },
    };
    expect(reduceNativeImageDrag({ candidate: true, visible: true }, payload)).toEqual({
      candidate: false,
      visible: false,
    });
    expect(payload.position).toEqual({ x: 4, y: 5 });
  });
});

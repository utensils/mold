import { describe, expect, it } from "vitest";
import { reduceNativeImageDrag } from "./nativeImageDrop";

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

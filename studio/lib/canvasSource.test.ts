import { describe, expect, it } from "vitest";
import { canvasSource } from "./canvasSource";

const first = { base64: "FIRST", width: 1280, height: 720 };
const last = { base64: "LAST", width: 720, height: 1280 };
describe("canvas source authority", () => {
  it("uses first, falls back to last on removal, and observes last replacement", () => {
    expect(
      canvasSource({
        mode: "single",
        supportsEndFrame: true,
        source: first,
        end: last,
      }),
    ).toBe(first);
    expect(
      canvasSource({
        mode: "single",
        supportsEndFrame: true,
        source: null,
        end: last,
      }),
    ).toBe(last);
    const replacement = { ...last, base64: "NEW", width: 1024, height: 1024 };
    expect(
      canvasSource({
        mode: "single",
        supportsEndFrame: true,
        source: null,
        end: replacement,
      }),
    ).toBe(replacement);
  });
  it("uses dedicated H3 boundaries and ignores parked endpoints on other recipes", () => {
    const h3 = {
      firstFrame: { data: "FIRST", width: 1280, height: 720 },
      lastFrame: { data: "LAST", width: 720, height: 1280 },
    };
    expect(
      canvasSource({ mode: "h3-boundaries", supportsEndFrame: false, h3 }),
    ).toEqual(first);
    expect(
      canvasSource({
        mode: "h3-boundaries",
        supportsEndFrame: false,
        h3: { ...h3, firstFrame: null },
      }),
    ).toEqual(last);
    expect(
      canvasSource({
        mode: "single",
        supportsEndFrame: false,
        source: null,
        end: last,
        h3,
      }),
    ).toBeNull();
    expect(
      canvasSource({
        mode: "h3-boundaries",
        supportsEndFrame: false,
        h3: { firstFrame: { ...h3.firstFrame, data: "" }, lastFrame: null },
      }),
    ).toBeNull();
  });
});

import { describe, expect, it } from "vitest";
import { autoGrowRows } from "./autogrow";

function textarea(scrollHeight: number, lineHeight = "20px"): HTMLTextAreaElement {
  const el = document.createElement("textarea");
  el.style.lineHeight = lineHeight;
  Object.defineProperty(el, "scrollHeight", { value: scrollHeight, configurable: true });
  document.body.appendChild(el);
  return el;
}

describe("autoGrowRows", () => {
  it("tracks content height below the cap without an internal scrollbar", () => {
    const el = textarea(60);
    autoGrowRows(el, 8);
    expect(el.style.height).toBe("60px");
    expect(el.style.overflowY).toBe("hidden");
  });

  it("caps at maxRows and flips to internal scrolling", () => {
    const el = textarea(400);
    autoGrowRows(el, 8); // 8 × 20px = 160px cap
    expect(el.style.height).toBe("160px");
    expect(el.style.overflowY).toBe("auto");
  });

  it("always clips horizontal overflow (WebKit UA default is auto)", () => {
    // Subpixel width rounding under webview zoom can otherwise surface a
    // phantom horizontal scrollbar even on an empty textarea.
    const short = textarea(60);
    autoGrowRows(short, 8);
    expect(short.style.overflowX).toBe("hidden");

    const tall = textarea(400);
    autoGrowRows(tall, 8);
    expect(tall.style.overflowX).toBe("hidden");
  });
});

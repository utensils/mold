import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ToastShelf from "./ToastShelf.vue";
import shelfSource from "./ToastShelf.vue?raw";
import { SEVERITY_MARKS } from "../lib/notificationSeverity";
import type { Toast } from "./types";

const TOASTS: readonly Toast[] = [
  { id: "t1", kind: "success", text: "Saved to library" },
  { id: "t2", kind: "info", text: "Model pull started" },
  { id: "t3", kind: "error", text: "Host unreachable" },
];

function make(toasts: readonly Toast[] = TOASTS) {
  return mount(ToastShelf, { props: { toasts } });
}

describe("ToastShelf", () => {
  it("renders one toast per entry", () => {
    const wrapper = make();
    expect(wrapper.findAll(".ms-toast")).toHaveLength(3);
  });

  it("renders an empty shelf without toasts", () => {
    const wrapper = make([]);
    expect(wrapper.findAll(".ms-toast")).toHaveLength(0);
  });

  it("uses role=status for success and info, role=alert for error and warning", () => {
    const wrapper = make();
    expect(wrapper.findAll("[role=status]")).toHaveLength(2);
    const alerts = wrapper.findAll("[role=alert]");
    expect(alerts).toHaveLength(1);
    expect(alerts[0]!.text()).toContain("Host unreachable");

    // A warning here is the sticky "your machine is gone" — as time-sensitive
    // as an error, so it must not wait for a polite region to be read.
    const warned = make([
      { id: "w", kind: "warning", text: "Can't reach plato" },
    ]);
    expect(warned.findAll("[role=alert]")).toHaveLength(1);
  });

  it("renders the newest toast first", () => {
    const wrapper = make();
    const texts = wrapper.findAll(".ms-toast__text").map((t) => t.text());
    expect(texts).toEqual([
      "Host unreachable",
      "Model pull started",
      "Saved to library",
    ]);
  });

  it("anchors the shelf to the top of the app frame", () => {
    const rule = /\.ms-toasts\s*\{([^}]*)\}/.exec(shelfSource)?.[1] ?? "";
    expect(rule).toMatch(/top:/);
    expect(rule).not.toMatch(/bottom:/);
  });

  it("renders the kind glyphs", () => {
    const wrapper = make();
    const glyphs = wrapper.findAll(".ms-toast__glyph").map((g) => g.text());
    expect(glyphs).toEqual(["✕", "•", "✓"]);
  });

  it("marks error toasts with the error class", () => {
    const wrapper = make();
    const toasts = wrapper.findAll(".ms-toast");
    expect(toasts[0]!.classes()).toContain("ms-toast--error");
    expect(toasts[2]!.classes()).not.toContain("ms-toast--error");
  });

  it("renders an action button only when actionLabel is set", () => {
    const wrapper = make([
      { id: "a", kind: "info", text: "Print deleted", actionLabel: "Undo" },
      { id: "b", kind: "info", text: "Plain" },
    ]);
    const actions = wrapper.findAll(".ms-toast__action");
    expect(actions).toHaveLength(1);
    expect(actions[0]!.text()).toBe("Undo");
  });

  it("emits action with the toast id", async () => {
    const wrapper = make([
      {
        id: "undo-me",
        kind: "info",
        text: "Print deleted",
        actionLabel: "Undo",
      },
    ]);
    await wrapper.find(".ms-toast__action").trigger("click");
    expect(wrapper.emitted("action")).toEqual([["undo-me"]]);
  });

  it("emits dismiss with the toast id", async () => {
    const wrapper = make();
    await wrapper.findAll(".ms-toast__dismiss")[1]!.trigger("click");
    expect(wrapper.emitted("dismiss")).toEqual([["t2"]]);
  });

  it("renders warnings with the yellow tone class, glyph, and label", () => {
    const wrapper = make([
      { id: "w", kind: "warning", text: "Can't reach plato" },
    ]);
    const toast = wrapper.get(".ms-toast");
    expect(toast.classes()).toContain("ms-toast--warning");
    expect(toast.classes()).not.toContain("ms-toast--error");
    expect(wrapper.get(".ms-toast__glyph").text()).toBe("!");
    expect(wrapper.get(".ms-toast__tone").text()).toBe("Warning");
  });

  it("tints each severity from the shared table, never a local copy", () => {
    const wrapper = make([
      { id: "i", kind: "info", text: "Queued" },
      { id: "s", kind: "success", text: "Saved" },
      { id: "w", kind: "warning", text: "Can't reach plato" },
      { id: "e", kind: "error", text: "Failed" },
    ]);
    // Newest first.
    expect(
      wrapper.findAll(".ms-toast__glyph").map((g) => g.attributes("style")),
    ).toEqual([
      `color: ${SEVERITY_MARKS.error.color};`,
      `color: ${SEVERITY_MARKS.warning.color};`,
      `color: ${SEVERITY_MARKS.success.color};`,
      `color: ${SEVERITY_MARKS.info.color};`,
    ]);
    // No stylesheet rule may restate a hue — that is how the surfaces drifted.
    expect(shelfSource).not.toMatch(/\.ms-toast--\w+ \.ms-toast__glyph/);
  });

  it("labels every dismiss button for screen readers", () => {
    const wrapper = make();
    for (const btn of wrapper.findAll(".ms-toast__dismiss")) {
      expect(btn.attributes("aria-label")).toBe("Dismiss");
    }
  });
});

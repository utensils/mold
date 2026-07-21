import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ToastShelf, { type Toast } from "./ToastShelf.vue";

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

  it("uses role=status for success and info, role=alert for error", () => {
    const wrapper = make();
    expect(wrapper.findAll("[role=status]")).toHaveLength(2);
    const alerts = wrapper.findAll("[role=alert]");
    expect(alerts).toHaveLength(1);
    expect(alerts[0]!.text()).toContain("Host unreachable");
  });

  it("renders the kind glyphs", () => {
    const wrapper = make();
    const glyphs = wrapper.findAll(".ms-toast__glyph").map((g) => g.text());
    expect(glyphs).toEqual(["✓", "•", "✕"]);
  });

  it("marks error toasts with the error class", () => {
    const wrapper = make();
    const toasts = wrapper.findAll(".ms-toast");
    expect(toasts[2]!.classes()).toContain("ms-toast--error");
    expect(toasts[0]!.classes()).not.toContain("ms-toast--error");
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

  it("labels every dismiss button for screen readers", () => {
    const wrapper = make();
    for (const btn of wrapper.findAll(".ms-toast__dismiss")) {
      expect(btn.attributes("aria-label")).toBe("Dismiss");
    }
  });
});

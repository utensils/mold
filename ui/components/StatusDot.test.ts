/**
 * The machine status dot was web-local while four desktop and phone surfaces
 * hand-rolled the same 8px circle. It is one component now; these pin the
 * contract those callers depend on — the state attribute, the decorative
 * role, and that it paints from `--mold-*` rather than a shell's alias.
 */
import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import StatusDot from "./StatusDot.vue";

describe("StatusDot", () => {
  it("is unknown before anything has answered", () => {
    const dot = mount(StatusDot);
    expect(dot.attributes("data-state")).toBe("unknown");
  });

  it("carries the state it is given", () => {
    for (const state of ["online", "offline", "unknown"] as const) {
      expect(
        mount(StatusDot, { props: { state } }).attributes("data-state"),
      ).toBe(state);
    }
  });

  it("is decoration — the state is said in words beside it", () => {
    const dot = mount(StatusDot, { props: { state: "offline" } });
    expect(dot.attributes("aria-hidden")).toBe("true");
    expect(dot.text()).toBe("");
  });
});

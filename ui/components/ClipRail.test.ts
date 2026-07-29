import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ClipRail from "./ClipRail.vue";
import railSource from "./ClipRail.vue?raw";
import pillSource from "./ClipPill.vue?raw";
import type { RailClip } from "./types";

function clips(count: number): RailClip[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `c${i}`,
    prompt: `clip ${i}`,
    frames: 97,
    transition: "smooth",
    fadeFrames: 8,
  }));
}

function make(props: Record<string, unknown> = {}) {
  return mount(ClipRail, {
    props: {
      clips: clips(2),
      activeId: "c0",
      motionTail: 17,
      ...props,
    },
  });
}

describe("ClipRail", () => {
  it("renders a pill per clip with one seam per join", () => {
    const wrapper = make({ clips: clips(3) });
    expect(wrapper.findAll(".ms-clip")).toHaveLength(3);
    expect(wrapper.findAll(".ms-seam")).toHaveLength(2);
    expect(wrapper.text()).toContain("Opening clip");
    expect(wrapper.text()).toContain("Clip 2");
    expect(wrapper.text()).toContain("Clip 3");
  });

  it("emits select and seam-click with the owning clip id", async () => {
    const wrapper = make({ clips: clips(2) });
    await wrapper.findAll(".ms-clip__body")[1]?.trigger("click");
    expect(wrapper.emitted("select")?.[0]).toEqual(["c1"]);

    await wrapper.find(".ms-seam").trigger("click");
    expect(wrapper.emitted("seam-click")?.[0]).toEqual(["c1"]);
  });

  it("gates the add pill on the stage cap", async () => {
    const open = make({ clips: clips(2), maxStages: 3 });
    expect(open.find(".ms-rail__add").exists()).toBe(true);
    await open.find(".ms-rail__add").trigger("click");
    expect(open.emitted("add")).toHaveLength(1);

    const full = make({ clips: clips(3), maxStages: 3 });
    expect(full.find(".ms-rail__add").exists()).toBe(false);
  });

  it("only offers removal above the two-clip floor", () => {
    expect(
      make({ clips: clips(2) })
        .find(".ms-clip__remove")
        .exists(),
    ).toBe(false);
    const three = make({ clips: clips(3) });
    expect(three.find(".ms-clip__remove").exists()).toBe(true);
  });

  it("surfaces the edit-session render plan on the pills", () => {
    const wrapper = make({
      clips: clips(3),
      plans: ["cached", "rerender", "new"],
    });
    expect(wrapper.find(".ms-clip__plan--cached").exists()).toBe(true);
    expect(wrapper.find(".ms-clip__plan--rerender").exists()).toBe(true);
  });

  it("keeps long rails scrollable without exposing a desktop scrollbar", () => {
    expect(railSource).toMatch(/\.ms-rail__clips\s*\{[^}]*flex:\s*0 0 auto/s);
    expect(railSource).toMatch(/\.ms-rail\s*\{[^}]*scrollbar-width:\s*none/s);
    expect(railSource).toMatch(
      /\.ms-rail::-webkit-scrollbar\s*\{[^}]*display:\s*none/s,
    );
  });

  it("reserves an in-pill slot for remove so it cannot cover the frame count", () => {
    expect(pillSource).toMatch(
      /\.ms-clip:has\(\.ms-clip__remove\) \.ms-clip__body\s*\{[^}]*padding-right:\s*36px/s,
    );
    expect(pillSource).toMatch(
      /\.ms-clip__remove\s*\{[^}]*top:\s*50%[^}]*right:\s*7px[^}]*opacity:\s*0/s,
    );
    expect(pillSource).toMatch(
      /\.ms-clip:hover \.ms-clip__remove,[\s\S]*\.ms-clip:focus-within \.ms-clip__remove\s*\{[^}]*opacity:\s*1/s,
    );
  });
});

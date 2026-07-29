import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { VueDraggable } from "vue-draggable-plus";
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

  it("emits the completed drag order and keeps tile controls out of the drag handle", async () => {
    const wrapper = make({ clips: clips(3) });
    wrapper
      .findComponent(VueDraggable)
      .vm.$emit("update:modelValue", [clips(3)[2], clips(3)[0], clips(3)[1]]);
    await wrapper.vm.$nextTick();

    expect(wrapper.emitted("reorder")?.[0]).toEqual([["c2", "c0", "c1"]]);
    expect(railSource).toContain(".ms-clip__resize,.ms-clip__resize *");
  });

  it("resizes a clip onto valid frame counts without starting a reorder", async () => {
    const timelineClips = clips(2);
    timelineClips[0]!.frames = 25;
    const wrapper = make({
      clips: timelineClips,
      frameOptions: [9, 17, 25, 33, 41],
    });

    await wrapper.findAll(".ms-clip__resize")[0]!.trigger("pointerdown", {
      clientX: 0,
    });
    document.dispatchEvent(new MouseEvent("pointermove", { clientX: 112 }));
    document.dispatchEvent(new MouseEvent("pointerup"));

    expect(wrapper.emitted("resize")?.at(-1)).toEqual(["c0", 41]);
    expect(wrapper.emitted("reorder")).toBeUndefined();
  });

  it("surfaces the edit-session render plan on filmstrip tiles", () => {
    const wrapper = make({
      clips: clips(3),
      plans: ["cached", "rerender", "new"],
    });
    expect(wrapper.text()).toContain("Cached");
    expect(wrapper.text()).toContain("Re-render");
  });

  it("renders durable stage posters and progress separately from draft clips", () => {
    const wrapper = make({
      mediaByClipId: {
        c0: {
          status: "ready",
          posterUrl: "https://mold.test/stage-0.jpg",
          hasMedia: true,
          cacheReady: true,
        },
        c1: {
          status: "running",
          progressPercent: 42.4,
        },
      },
    });
    expect(
      wrapper.get('img[src="https://mold.test/stage-0.jpg"]').attributes("src"),
    ).toBe("https://mold.test/stage-0.jpg");
    expect(wrapper.get(".ms-clip__status").text()).toContain("Cached");
    expect(wrapper.get(".ms-clip__rendering").text()).toContain("42%");
    expect(
      wrapper.get(".ms-clip__progress > span").attributes("style"),
    ).toContain("42.4%");
  });

  it("emits play for playable scenes and exposes duration in accessible labels", async () => {
    const wrapper = make({
      fps: 24,
      playingId: "c0",
      mediaByClipId: {
        c0: { status: "ready", hasMedia: true, cacheReady: true },
      },
    });
    const play = wrapper.get(".ms-clip__play");
    expect(play.attributes("aria-label")).toBe(
      "Pause Opening clip, 97f · 4.0s",
    );
    expect(play.attributes("aria-pressed")).toBe("true");
    expect(wrapper.get(".ms-clip__frames").text()).toBe("97f · 4.0s");
    await play.trigger("click");
    expect(wrapper.emitted("play")?.[0]).toEqual(["c0"]);
  });

  it("keeps long rails scrollable without exposing a desktop scrollbar", () => {
    expect(railSource).toMatch(/\.ms-rail__clips\s*\{[^}]*flex:\s*0 0 auto/s);
    expect(railSource).toMatch(/\.ms-rail\s*\{[^}]*scrollbar-width:\s*none/s);
    expect(railSource).toMatch(
      /\.ms-rail::-webkit-scrollbar\s*\{[^}]*display:\s*none/s,
    );
  });

  it("keeps playback and removal as separate, keyboard-focusable controls", () => {
    expect(pillSource).toMatch(/class="ms-clip__play"[\s\S]*:aria-label=/);
    expect(pillSource).toMatch(
      /\.ms-clip__remove\s*\{[^}]*position:\s*absolute[^}]*opacity:\s*0/s,
    );
    expect(pillSource).toMatch(
      /\.ms-clip:hover \.ms-clip__remove,[\s\S]*\.ms-clip:focus-within \.ms-clip__remove\s*\{[^}]*opacity:\s*1/s,
    );
  });

  it("uses a cinematic 16:9 filmstrip treatment", () => {
    expect(pillSource).toMatch(
      /\.ms-clip__thumb\s*\{[^}]*aspect-ratio:\s*16 \/ 9/s,
    );
    expect(railSource).toMatch(/\.ms-rail__perfs/);
    expect(railSource).toMatch(
      /\.ms-rail\s*\{[^}]*background:[\s\S]*color-mix\(in srgb, var\(--print\)/s,
    );
  });
});

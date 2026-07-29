import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { VueDraggable } from "vue-draggable-plus";
import ClipRail from "./ClipRail.vue";
import railSource from "./ClipRail.vue?raw";
import pillSource from "./ClipPill.vue?raw";
import seamSource from "./SeamPill.vue?raw";
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
    const draggable = wrapper.findComponent(VueDraggable);
    expect(draggable.props("forceFallback")).toBe(true);
    expect(draggable.props("fallbackTolerance")).toBe(3);
    draggable.vm.$emit("update:modelValue", [
      clips(3)[2],
      clips(3)[0],
      clips(3)[1],
    ]);
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

  it("makes overflow intentional with snap points, an end gutter, and a visible scrollbar", () => {
    expect(railSource).toMatch(/\.ms-rail__clips\s*\{[^}]*flex:\s*0 0 auto/s);
    expect(railSource).toMatch(
      /\.ms-rail\s*\{[^}]*scroll-snap-type:\s*x proximity/s,
    );
    expect(railSource).toMatch(
      /\.ms-rail__item\s*\{[^}]*scroll-snap-align:\s*start/s,
    );
    expect(railSource).toContain('class="ms-rail__end-space"');
    expect(railSource).toContain('aria-label="Scroll filmstrip right"');
  });

  it("reveals overflow controls and scrolls the filmstrip by a useful page", async () => {
    const wrapper = make({ clips: clips(6) });
    const rail = wrapper.get(".ms-rail").element as HTMLElement;
    Object.defineProperties(rail, {
      clientWidth: { configurable: true, value: 400 },
      scrollWidth: { configurable: true, value: 1200 },
      scrollLeft: { configurable: true, writable: true, value: 0 },
    });
    const scrollTo = vi.fn();
    rail.scrollTo = scrollTo;
    rail.dispatchEvent(new Event("scroll"));
    await wrapper.vm.$nextTick();

    expect(wrapper.attributes("data-scroll-end")).toBe("true");
    await wrapper.get('[aria-label="Scroll filmstrip right"]').trigger("click");
    expect(scrollTo).toHaveBeenLastCalledWith({
      left: 288,
      behavior: "smooth",
    });
  });

  it("auto-scrolls a newly selected tile fully inside the viewport", async () => {
    const wrapper = make({ clips: clips(6), activeId: "c0" });
    const rail = wrapper.get(".ms-rail").element as HTMLElement;
    Object.defineProperties(rail, {
      clientWidth: { configurable: true, value: 400 },
      scrollWidth: { configurable: true, value: 1400 },
      scrollLeft: { configurable: true, writable: true, value: 0 },
    });
    rail.getBoundingClientRect = () => ({ left: 0, right: 400 }) as DOMRect;
    const selected = wrapper.get('[data-clip-id="c5"]').element as HTMLElement;
    selected.getBoundingClientRect = () =>
      ({ left: 900, right: 1096 }) as DOMRect;
    const scrollTo = vi.fn();
    rail.scrollTo = scrollTo;

    await wrapper.setProps({ activeId: "c5" });
    await wrapper.vm.$nextTick();
    expect(scrollTo).toHaveBeenLastCalledWith({
      left: 742,
      behavior: "smooth",
    });
  });

  it("defines tall, default, short, and compact filmstrip density tiers", () => {
    expect(railSource).toMatch(
      /\.ms-rail-frame\s*\{[^}]*--filmstrip-tile-max:\s*184px[^}]*height:\s*188px/s,
    );
    expect(railSource).toMatch(
      /@container create-bench \(min-height: 620px\)[\s\S]*--filmstrip-tile-max:\s*208px[\s\S]*height:\s*204px/,
    );
    expect(railSource).toMatch(
      /@container create-bench \(max-height: 430px\)[\s\S]*--filmstrip-tile-max:\s*156px[\s\S]*height:\s*160px/,
    );
    expect(railSource).toMatch(
      /@container create-bench \(max-height: 340px\)[\s\S]*--filmstrip-tile-max:\s*134px[\s\S]*height:\s*138px/,
    );
    expect(pillSource).toContain(
      "width: min(var(--clip-width), var(--filmstrip-tile-max, 196px))",
    );
  });

  it("changes duration on the x axis while every tile keeps the track height", () => {
    const wrapper = make({
      clips: [
        { ...clips(2)[0]!, frames: 25 },
        { ...clips(2)[1]!, frames: 97 },
      ],
      frameOptions: [25, 33, 41, 49, 57, 65, 73, 81, 89, 97],
    });
    const tiles = wrapper.findAll(".ms-clip");
    expect(tiles[0]!.attributes("style")).not.toBe(
      tiles[1]!.attributes("style"),
    );
    expect(pillSource).toMatch(
      /\.ms-clip__body\s*\{[^}]*height:\s*var\(--filmstrip-scene-height,\s*140px\)/s,
    );
    expect(pillSource).toMatch(
      /\.ms-clip__thumb\s*\{[^}]*height:\s*var\(--filmstrip-thumb-height,\s*104px\)/s,
    );
    expect(pillSource).toMatch(
      /\.ms-clip__thumb :deep\(img\),[\s\S]*object-fit:\s*cover/s,
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

  it("uses a cinematic fixed-track filmstrip treatment", () => {
    expect(railSource).toContain("--filmstrip-thumb-height: 104px");
    expect(railSource).toMatch(/\.ms-rail__perfs/);
    expect(railSource).toMatch(
      /\.ms-rail-frame\s*\{[^}]*background:[\s\S]*color-mix\(in srgb, var\(--print\)/s,
    );
  });

  it("keeps seams as compact connectors instead of thumbnail-height capsules", () => {
    expect(seamSource).toContain('class="ms-seam__caption"');
    expect(railSource).toMatch(
      /\.ms-rail__item\s*\{[^}]*align-items:\s*flex-start/s,
    );
    expect(railSource).toMatch(
      /\.ms-rail-frame :deep\(\.ms-seam\)\s*\{[^}]*grid-template-rows:\s*32px 16px[^}]*margin-top:\s*calc\(\(var\(--filmstrip-thumb-height\) - 34px\) \/ 2\)[^}]*min-width:\s*44px[^}]*height:\s*54px[^}]*border:\s*0[^}]*background:\s*transparent/s,
    );
    expect(railSource).toMatch(
      /\.ms-rail-frame :deep\(\.ms-seam::before\)\s*\{[^}]*height:\s*1px/s,
    );
    expect(railSource).toMatch(
      /\.ms-rail-frame :deep\(\.ms-seam__diagram\)\s*\{[^}]*width:\s*30px[^}]*height:\s*30px[^}]*border-radius:\s*50%/s,
    );
    expect(railSource).toContain(
      ':deep(.ms-seam[data-transition="fade"] .ms-seam__diagram)',
    );
  });
});

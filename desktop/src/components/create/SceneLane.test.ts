import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SceneLane from "./SceneLane.vue";
import type { RailClip } from "@ui/components/types";

function clip(id: string, prompt: string, frames: number): RailClip {
  return { id, prompt, frames, transition: "smooth", fadeFrames: 8 };
}

function make(props: Record<string, unknown> = {}) {
  return mount(SceneLane, {
    props: {
      clips: [clip("c0", "rain on the gutter", 50), clip("c1", "the boat sets off", 100)],
      activeId: "c0",
      motionTail: 0,
      fps: 25,
      ...props,
    },
  });
}

/** Widths come from the lane's own box, so the drag needs a real one. */
function sizeLane(wrapper: ReturnType<typeof make>, width: number) {
  const lane = wrapper.get("[data-test='scene-lane']").element as HTMLElement;
  lane.getBoundingClientRect = () => ({ width, left: 0, right: width }) as DOMRect;
}

describe("SceneLane — the lane is the clip", () => {
  it("gives every block a flex grow equal to the seconds it plays", () => {
    const blocks = make().findAll("[data-test='scene-block']");
    expect(blocks.map((block) => (block.element as HTMLElement).style.flexGrow)).toEqual([
      "2",
      "4",
    ]);
    expect(blocks.map((block) => (block.element as HTMLElement).style.flexBasis)).toEqual([
      "0px",
      "0px",
    ]);
  });

  /*
   * The block's width and its caption are the same number. Sized by the
   * frames it PLAYS but labelled with `clip.frames`, a smooth-seamed block
   * was narrower than its own caption claimed — the lane is the ruler, so the
   * two cannot disagree. The authored length stays one hover away.
   */
  it("captions every block with the time it plays, not its authored length", () => {
    const feet = make({ motionTail: 25 }).findAll("[data-test='scene-block'] .ms-lane__length");
    // 50f and 100f at 25fps, the second handing 25f back across a smooth seam.
    expect(feet.map((foot) => foot.text())).toEqual(["50f · 2.0s", "75f · 3.0s"]);
  });

  it("names the authored length on the block whose seam takes frames from it", () => {
    const feet = make({ motionTail: 25 }).findAll("[data-test='scene-block'] .ms-lane__length");
    expect(feet[0]!.attributes("title")).toBeUndefined();
    expect(feet[1]!.attributes("title")).toContain("Plays 75f · 3.0s of 100f · 4.0s");
    expect(feet[1]!.attributes("title")).toContain("25 frames");
  });

  it("says one number when no seam takes anything", () => {
    const feet = make({ motionTail: 0 }).findAll("[data-test='scene-block'] .ms-lane__length");
    expect(feet.map((foot) => foot.text())).toEqual(["50f · 2.0s", "100f · 4.0s"]);
    expect(feet.map((foot) => foot.attributes("title"))).toEqual([undefined, undefined]);
  });

  it("counts a smooth seam's carried frames once, so the lane sums to the clip", () => {
    const blocks = make({ motionTail: 25 }).findAll("[data-test='scene-block']");
    expect(blocks.map((block) => (block.element as HTMLElement).style.flexGrow)).toEqual([
      "2",
      "3",
    ]);
  });

  it("titles a block with the scene's own words and names an undescribed one", () => {
    const wrapper = make({ clips: [clip("c0", "", 50), clip("c1", "the boat sets off", 50)] });
    const titles = wrapper.findAll("[data-test='scene-title']").map((title) => title.text());
    expect(titles).toEqual(["Opening scene", "the boat sets off"]);
  });

  it("says scenes to assistive tech", () => {
    expect(make().get("[data-test='scene-lane']").attributes("aria-label")).toBe("Scenes lane");
  });
});

describe("SceneLane — trimming", () => {
  it("puts the trim grip on the selected block alone", () => {
    const wrapper = make({ frameOptions: [25, 50, 75, 100] });
    expect(wrapper.findAll("[data-test='scene-grip']")).toHaveLength(1);
    expect(
      wrapper
        .get("[data-test='scene-grip']")
        .element.closest("[data-clip-id]")
        ?.getAttribute("data-clip-id"),
    ).toBe("c0");
  });

  it("snaps a grip drag onto the frame grid", async () => {
    const wrapper = make({ frameOptions: [25, 50, 75, 100] });
    sizeLane(wrapper, 300);
    await wrapper.get("[data-test='scene-grip']").trigger("pointerdown", { clientX: 0 });
    document.dispatchEvent(new MouseEvent("pointermove", { clientX: 55 }));
    document.dispatchEvent(new MouseEvent("pointerup"));

    expect(wrapper.emitted("resize")?.at(-1)).toEqual(["c0", 75]);
  });

  it("offers no grip when the model pins the scene length", () => {
    expect(
      make({ frameOptions: [50] })
        .find("[data-test='scene-grip']")
        .exists(),
    ).toBe(false);
  });
});

describe("SceneLane — seams and keys", () => {
  it("rides a seam chip on every join but the opening", async () => {
    // A zero motion tail is LTX-Video's independent-clip join.
    const wrapper = make();
    const seams = wrapper.findAll("[data-test='scene-seam']");
    expect(seams).toHaveLength(1);
    expect(seams[0]!.text()).toContain("Join");
    expect(seams[0]!.attributes("aria-label")).toBe(
      "How rain on the gutter meets the boat sets off: Join",
    );

    await seams[0]!.trigger("click");
    expect(wrapper.emitted("seam-click")?.[0]).toEqual(["c1"]);
  });

  it("names a carried seam Smooth and says a fade's length", () => {
    const smooth = make({ motionTail: 25 });
    expect(smooth.get("[data-test='scene-seam']").text()).toContain("Smooth");

    const fading = clip("c1", "the boat sets off", 100);
    fading.transition = "fade";
    const wrapper = make({
      motionTail: 25,
      clips: [clip("c0", "rain on the gutter", 50), fading],
    });
    const chip = wrapper.get("[data-test='scene-seam']");
    expect(chip.text()).toContain("Fade");
    expect(chip.text()).toContain("8f");
  });

  it("moves the selection with the arrows and opens a seam with Enter", async () => {
    const wrapper = make();
    const first = wrapper.findAll("[data-test='scene-block']")[0]!;
    await first.trigger("keydown", { key: "ArrowRight" });
    expect(wrapper.emitted("select")?.at(-1)).toEqual(["c1"]);

    const second = wrapper.findAll("[data-test='scene-block']")[1]!;
    await second.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("seam-click")?.at(-1)).toEqual(["c1"]);
  });

  it("removes with Delete, but never below the two-scene floor", async () => {
    const two = make();
    await two.findAll("[data-test='scene-block']")[1]!.trigger("keydown", { key: "Delete" });
    expect(two.emitted("remove")).toBeUndefined();

    const three = make({
      clips: [clip("c0", "one", 50), clip("c1", "two", 50), clip("c2", "three", 50)],
    });
    await three.findAll("[data-test='scene-block']")[1]!.trigger("keydown", { key: "Delete" });
    expect(three.emitted("remove")?.at(-1)).toEqual(["c1"]);
  });

  it("leaves the opening scene no seam to open", async () => {
    const wrapper = make();
    await wrapper.findAll("[data-test='scene-block']")[0]!.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("seam-click")).toBeUndefined();
  });
});

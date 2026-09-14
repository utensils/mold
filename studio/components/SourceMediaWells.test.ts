import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SourceMediaWells from "./SourceMediaWells.vue";
import {
  EXCLUSIVE_WELLS_NOTE,
  type SourceMediaPlan,
} from "../lib/sourceMediaPlan";

function factory(plan: SourceMediaPlan, extra: Record<string, unknown> = {}) {
  return mount(SourceMediaWells, { props: { plan, ...extra } });
}

describe("SourceMediaWells", () => {
  it("renders nothing for plans it does not own", () => {
    for (const plan of [
      { kind: "none" },
      { kind: "attachments", max: null, required: false, primary: null },
      { kind: "h3-references" },
    ] satisfies SourceMediaPlan[]) {
      const wrapper = factory(plan);
      expect(wrapper.find("[data-test='source-media-wells']").exists()).toBe(
        false,
      );
    }
  });

  it("renders the shared primary well as a Qwen edit Target", () => {
    const wrapper = factory({
      kind: "attachments",
      max: null,
      required: true,
      primary: "target",
    });
    expect(wrapper.text()).toContain("Target");
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      true,
    );
  });

  it("renders one optional image well for a plain image model", () => {
    const wrapper = factory({
      kind: "single",
      required: false,
      endFrame: false,
      video: false,
    });
    expect(wrapper.text()).toContain("Source");
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      false,
    );
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);
  });

  it("adds the required badge and end-frame well per the advertised contract", async () => {
    const wrapper = factory({
      kind: "single",
      required: true,
      endFrame: true,
      video: true,
    });
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(true);
    await wrapper.get("[data-test='end-frame-gallery']").trigger("click");
    expect(wrapper.emitted("gallery")).toEqual([["end"]]);
  });

  it("routes files, gallery, and clear per slot", async () => {
    const wrapper = factory(
      {
        kind: "single",
        required: false,
        endFrame: true,
        video: true,
      },
      { source: { data: "QUJD", filename: "still.png" } },
    );
    await wrapper.get("[data-test='source-remove']").trigger("click");
    expect(wrapper.emitted("clear")).toEqual([["source"]]);
    await wrapper.get("[data-test='source-replace']").trigger("click");
    expect(wrapper.emitted("gallery")).toEqual([["source"]]);
    const file = new File(["png"], "closing.png", { type: "image/png" });
    await wrapper
      .get("[data-test='end-frame-well']")
      .trigger("drop", { dataTransfer: { files: [file] } });
    expect(wrapper.emitted("file")).toEqual([["end", file]]);
  });

  it("prefixes surface hooks without duplicating the well implementation", () => {
    const wrapper = factory(
      { kind: "single", required: false, endFrame: false, video: false },
      { source: { data: "QUJD" }, testIdPrefix: "mobile-" },
    );
    expect(wrapper.find("[data-test='mobile-source-preview']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='mobile-source-replace']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='mobile-source-remove']").exists()).toBe(
      true,
    );
  });

  it("renders H3 boundaries with frame wording and hides the empty last frame when only first is reviewed", () => {
    const required = factory({
      kind: "h3-boundaries",
      requiredEndpoint: "first",
    });
    expect(required.text()).toContain("First frame");
    expect(required.find("[data-test='source-required-badge']").exists()).toBe(
      true,
    );
    expect(required.find("[data-test='end-frame-well']").exists()).toBe(false);

    const open = factory({ kind: "h3-boundaries", requiredEndpoint: null });
    expect(open.text()).toContain("Last frame");
    expect(open.find("[data-test='end-frame-well']").exists()).toBe(true);
  });

  it("keeps a restored incompatible H3 last frame removable but never re-acquirable", async () => {
    const wrapper = factory(
      { kind: "h3-boundaries", requiredEndpoint: "first" },
      { endFrame: { data: "TEFTVA==", filename: "old-last.png" } },
    );
    expect(wrapper.text()).toContain("Incompatible");
    expect(wrapper.text()).toContain("first frame only");
    const replace = wrapper.get("[data-test='end-frame-replace']");
    expect(replace.attributes("disabled")).toBeDefined();
    await wrapper.get("[data-test='end-frame-remove']").trigger("click");
    expect(wrapper.emitted("clear")).toEqual([["end"]]);
  });

  it("surfaces the conditioning error under the source well", () => {
    const wrapper = factory(
      { kind: "single", required: true, endFrame: false, video: true },
      { error: "This checkpoint renders from an image — attach one." },
    );
    expect(
      wrapper.get("[data-test='source-conditioning-error']").text(),
    ).toContain("attach one");
  });
});

describe("SourceMediaWells for an exclusive (klein) plan", () => {
  const klein: SourceMediaPlan = {
    kind: "single-or-references",
    single: { required: false, endFrame: false, video: false },
    references: { max: 4, maxPixelsSingle: null, maxPixelsMulti: null },
  };

  it("renders the same Source well the single plan does", () => {
    const wrapper = factory(klein);
    expect(wrapper.find("[data-test='source-media-wells']").exists()).toBe(
      true,
    );
    expect(wrapper.text()).toContain("Source");
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);
    expect(wrapper.find("[data-test='source-parked-note']").exists()).toBe(
      false,
    );
  });

  it("parks with an inline note while keeping the well interactive", async () => {
    const wrapper = factory(klein, {
      source: { data: "QUJD", filename: "still.png" },
      parked: true,
      note: EXCLUSIVE_WELLS_NOTE,
    });
    expect(wrapper.get("[data-test='source-parked-note']").text()).toBe(
      EXCLUSIVE_WELLS_NOTE,
    );
    expect(wrapper.attributes("data-parked")).toBe("true");
    // Parked is not disabled: attaching here makes this the active well again
    // (last write wins), and the parked media is never discarded.
    expect(
      wrapper.get("[data-test='source-replace']").attributes("disabled"),
    ).toBeUndefined();
    await wrapper.get("[data-test='source-remove']").trigger("click");
    expect(wrapper.emitted("clear")).toEqual([["source"]]);
  });

  it("names its drop target so an OS drag can reach it", () => {
    expect(
      factory(klein)
        .get("[data-test='source-well']")
        .element.closest("[data-drop-target]")
        ?.getAttribute("data-drop-target"),
    ).toBe("source");
    // H3's boundaries are their own targets, not the generic source well.
    const h3 = factory({ kind: "h3-boundaries", requiredEndpoint: null });
    expect(
      h3
        .get("[data-test='source-well']")
        .element.closest("[data-drop-target]")
        ?.getAttribute("data-drop-target"),
    ).toBe("h3-first");
    expect(
      h3
        .get("[data-test='end-frame-well']")
        .element.closest("[data-drop-target]")
        ?.getAttribute("data-drop-target"),
    ).toBe("h3-last");
  });
});

describe("SourceMediaWells — a caller that already titled the group", () => {
  /*
   * The desktop Create rail heads this group with its own plain-language
   * label ("Start from a photo"), so the well's terse "SOURCE" legend under
   * it rendered the same heading twice:
   *
   *     START FROM A PHOTO
   *     SOURCE ─────────────
   *
   * `titled` lets that caller say so. It suppresses ONLY the sole generic
   * "Source" legend — the case where the legend carries no information the
   * caller's heading does not. A legend that distinguishes one well from
   * another (First/Last frame, Target, or any layout with a second well)
   * is doing real work and stays regardless.
   */
  const soleSource = {
    kind: "single",
    required: false,
    endFrame: false,
    video: false,
  } satisfies SourceMediaPlan;

  it("drops the sole Source legend when the caller titled the group", () => {
    const wrapper = factory(soleSource, { titled: true });
    expect(wrapper.find("[data-test='source-media-wells']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
    expect(wrapper.text()).not.toContain("Source");
  });

  it("keeps the legend when the caller did not title the group", () => {
    expect(factory(soleSource).text()).toContain("Source");
  });

  it("keeps a required badge the legend would otherwise have carried", () => {
    const wrapper = factory(
      { kind: "single", required: true, endFrame: false, video: false },
      { titled: true },
    );
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      true,
    );
  });

  it("keeps both legends when a second well makes them tell wells apart", () => {
    const wrapper = factory(
      { kind: "single", required: false, endFrame: true, video: true },
      { titled: true },
    );
    expect(wrapper.text()).toContain("Source");
    expect(wrapper.text()).toContain("End frame");
  });

  it("keeps a legend that names something the caller's heading cannot", () => {
    const h3 = factory(
      { kind: "h3-boundaries", requiredEndpoint: "first" },
      { titled: true },
    );
    expect(h3.text()).toContain("First frame");

    const qwen = factory(
      { kind: "attachments", max: null, required: true, primary: "target" },
      { titled: true },
    );
    expect(qwen.text()).toContain("Target");
  });
});

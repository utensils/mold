import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SourceMediaWells from "./SourceMediaWells.vue";
import type { SourceMediaPlan } from "../lib/sourceMediaPlan";

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

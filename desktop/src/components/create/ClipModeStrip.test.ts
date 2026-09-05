import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import ClipModeStrip from "./ClipModeStrip.vue";
import clipModeStripSource from "./ClipModeStrip.vue?raw";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    appSettingsGet: vi.fn().mockResolvedValue({}),
  },
}));

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

function stillForm(): GenerateForm {
  return reactive({ ...newGenerateForm(), family: "flux" });
}

function clipForm(): GenerateForm {
  return reactive({ ...newGenerateForm(), model: "ltx-video", family: "ltx-video" });
}

function meshForm(): GenerateForm {
  return reactive({ ...newGenerateForm(), family: "hunyuan3d" });
}

function segments(wrapper: ReturnType<typeof mount>) {
  return wrapper.get("[data-test='clip-mode']").findAll("button");
}

function checked(wrapper: ReturnType<typeof mount>) {
  return segments(wrapper)
    .find((b) => b.attributes("aria-checked") === "true")
    ?.text();
}

/*
 * Simple | Scenes — how the clip gets made — is a row of its own UNDER the
 * toolbar rather than a second control beside Still picture | Short clip | 3-D
 * object. On the toolbar it sat between that control and the doors, so
 * choosing Short clip pushed the whole right-hand cluster left by its width and the
 * control a person had just clicked jumped away from the pointer. The strip
 * exists only while the kind is a clip, and what is on screen answers it, so
 * the toggle and the view can never disagree.
 */
describe("ClipModeStrip", () => {
  it("is present only while the kind is a clip", () => {
    expect(
      mount(ClipModeStrip, { props: { form: stillForm() } })
        .find("[data-test='clip-mode-strip']")
        .exists(),
    ).toBe(false);
    expect(
      mount(ClipModeStrip, { props: { form: meshForm() } })
        .find("[data-test='clip-mode-strip']")
        .exists(),
    ).toBe(false);

    const clip = mount(ClipModeStrip, { props: { form: clipForm() } });
    expect(clip.find("[data-test='clip-mode-strip']").exists()).toBe(true);
    expect(segments(clip).map((b) => b.text())).toEqual(["Simple", "Scenes"]);
    expect(clip.get("[data-test='clip-mode']").attributes("aria-label")).toBe(
      "How to make the clip",
    );
  });

  it("appears for a sequence draft even before the form holds a clip style", () => {
    // Scenes on a machine with no clip style: the draft is the sequence and
    // the timeline owns the empty state, so the way back to Simple must stay.
    useSequenceDraftStore().output = "sequence";
    const wrapper = mount(ClipModeStrip, { props: { form: stillForm() } });
    expect(wrapper.find("[data-test='clip-mode-strip']").exists()).toBe(true);
    expect(checked(wrapper)).toBe("Scenes");
  });

  it("reads Simple for a fresh draft and Scenes for a sequence", () => {
    expect(checked(mount(ClipModeStrip, { props: { form: clipForm() } }))).toBe("Simple");
    useSequenceDraftStore().output = "sequence";
    expect(checked(mount(ClipModeStrip, { props: { form: clipForm() } }))).toBe("Scenes");
  });

  it("says in one sentence what the chosen way does", () => {
    const simple = mount(ClipModeStrip, { props: { form: clipForm() } });
    expect(simple.get("[data-test='clip-mode-hint']").text()).toBe("One prompt, one clip.");
    useSequenceDraftStore().output = "sequence";
    const scenes = mount(ClipModeStrip, { props: { form: clipForm() } });
    expect(scenes.get("[data-test='clip-mode-hint']").text()).toBe(
      "Scene by scene, joined into one clip.",
    );
  });

  it("hands the switch to the view, which seeds and parks the draft", async () => {
    const wrapper = mount(ClipModeStrip, { props: { form: clipForm() } });
    await segments(wrapper)[1]!.trigger("click");
    expect(wrapper.emitted("set-clip-mode")).toEqual([["scenes"]]);

    // Picking what is already on screen changes nothing.
    await segments(wrapper)[0]!.trigger("click");
    expect(wrapper.emitted("set-clip-mode")).toEqual([["scenes"]]);
  });

  it("keeps the tool's own words off the screen", () => {
    const template = clipModeStripSource
      .replace(/<script[\s\S]*?<\/script>/g, "")
      .replace(/<[^>]*>/g, " ")
      .replace(/\{\{[\s\S]*?\}\}/g, " ");
    for (const banned of [/\beditor\b/i, /\btimeline\b/i, /\bmode\b/i]) {
      expect(banned.test(template), `${banned}`).toBe(false);
    }
  });
});

import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it } from "vitest";
import { reactive } from "vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import SequenceAdvancedSettings from "./SequenceAdvancedSettings.vue";
import { newGenerateForm } from "../../lib/generateForm";

const cameraControls = [
  {
    id: "dolly-in",
    label: "Dolly in",
    size_bytes: 327_309_208,
    installed: false,
    download_model: "ltx2-camera-control-dolly-in-19b",
    download_repo: "Lightricks/camera",
    download_filename: "dolly-in.safetensors",
    download_sha256: "a".repeat(64),
  },
];

beforeEach(() => {
  setActivePinia(createPinia());
  useSequenceDraftStore().ensureClips(97);
});

describe("SequenceAdvancedSettings camera motion", () => {
  it("leaves the opening image to the primary form and out of its active count", async () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "PARKED" };
    const form = reactive(newGenerateForm());
    const wrapper = mount(SequenceAdvancedSettings, { props: { form } });

    expect(wrapper.find("[data-test='sequence-section-opening-image']").exists()).toBe(false);
    expect(wrapper.find("[data-test='sequence-opening-image-well']").exists()).toBe(false);
    expect(wrapper.find("[data-test='sequence-source-strength']").exists()).toBe(false);
    expect(wrapper.find("[data-test='sequence-source-fit']").exists()).toBe(false);
    expect(wrapper.findComponent({ name: "ImagePickerModal" }).exists()).toBe(false);
    // An attached opening image alone is not a sequence-Advanced control.
    expect(wrapper.get(".ms-adv__summary").text()).toBe("Sequence controls");
    expect(draft.openingImage?.base64).toBe("PARKED");

    draft.clips[0]!.negativePrompt = "blurry";
    await flushPromises();
    expect(wrapper.get(".ms-adv__summary").text()).toBe("1 active");
    expect(draft.openingImage?.base64).toBe("PARKED");
  });

  it("preserves but disables clip negatives from an opaque model's advertised recipe", () => {
    const form = newGenerateForm();
    form.family = "ltx2";
    form.model = "hf:opaque/checkpoint";
    form.guidanceCapabilities = {
      adjustable: false,
      supports_negative_prompt: false,
      fixed_scale: 1,
    };
    useSequenceDraftStore().clips[0]!.negativePrompt = "flicker";
    const wrapper = mount(SequenceAdvancedSettings, { props: { form } });
    const input = wrapper.get(
      "textarea[aria-label='What the selected scene should steer away from']",
    );
    expect(input.attributes("disabled")).toBeDefined();
    expect(useSequenceDraftStore().clips[0]!.negativePrompt).toBe("flicker");
    expect(wrapper.get("[data-test='sequence-negative-unavailable-hint']").text()).toContain(
      "does not use negative-prompt guidance",
    );
  });

  it("edits the active clip and labels a first-use download", async () => {
    const draft = useSequenceDraftStore();
    const wrapper = mount(SequenceAdvancedSettings, {
      props: { form: newGenerateForm(), cameraControls, cameraControlsEnabled: true },
    });

    const select = wrapper.get("[data-test='sequence-camera-motion']");
    expect(select.text()).toContain("downloads on first use");
    await select.setValue("dolly-in");
    expect(draft.clips[0]?.cameraControl).toBe("dolly-in");
  });

  it("resets camera motion across the sequence", async () => {
    const draft = useSequenceDraftStore();
    draft.enableAudio = true;
    draft.clips[0]!.cameraControl = "dolly-in";
    draft.clips[1]!.cameraControl = "/models/camera/custom.safetensors";
    const wrapper = mount(SequenceAdvancedSettings, {
      props: { form: newGenerateForm(), cameraControls, cameraControlsEnabled: true },
    });
    await wrapper.get("[data-test='sequence-advanced-reset']").trigger("click");
    expect(draft.clips.map((clip) => clip.cameraControl)).toEqual([null, null]);
    expect(draft.enableAudio).toBe(true);
  });

  it("preserves the opening image and source conditioning across Reset", async () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    const form = newGenerateForm();
    form.strength = 0.55;
    form.sourceFit = { mode: "pad-repaint" };
    const wrapper = mount(SequenceAdvancedSettings, {
      props: { form, cameraControls, cameraControlsEnabled: true },
    });
    await wrapper.get("[data-test='sequence-advanced-reset']").trigger("click");
    expect(draft.openingImage?.filename).toBe("opening.png");
    expect(form.strength).toBe(0.55);
    expect(form.sourceFit).toEqual({ mode: "pad-repaint" });
  });

  it("hides camera motion outside the LTX-2 family", () => {
    const wrapper = mount(SequenceAdvancedSettings, {
      props: { form: newGenerateForm(), cameraControls, cameraControlsEnabled: false },
    });
    expect(wrapper.find("[data-test='sequence-section-camera']").exists()).toBe(false);
  });
});

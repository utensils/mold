import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it } from "vitest";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import SequenceAdvancedSettings from "./SequenceAdvancedSettings.vue";

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
  it("edits the active clip and labels a first-use download", async () => {
    const draft = useSequenceDraftStore();
    const wrapper = mount(SequenceAdvancedSettings, {
      props: { cameraControls, cameraControlsEnabled: true },
    });
    const select = wrapper.get("[data-test='sequence-camera-motion']");
    expect(select.text()).toContain("downloads on first use");
    await select.setValue("dolly-in");
    expect(draft.clips[0]?.cameraControl).toBe("dolly-in");
  });

  it("resets camera motion across the sequence", async () => {
    const draft = useSequenceDraftStore();
    draft.clips[0]!.cameraControl = "dolly-in";
    draft.clips[1]!.cameraControl = "/models/camera/custom.safetensors";
    const wrapper = mount(SequenceAdvancedSettings, {
      props: { cameraControls, cameraControlsEnabled: true },
    });
    await wrapper.get("[data-test='sequence-advanced-reset']").trigger("click");
    expect(draft.clips.map((clip) => clip.cameraControl)).toEqual([null, null]);
  });

  it("hides camera motion outside the LTX-2 family", () => {
    const wrapper = mount(SequenceAdvancedSettings, {
      props: { cameraControls, cameraControlsEnabled: false },
    });
    expect(wrapper.find("[data-test='sequence-section-camera']").exists()).toBe(false);
  });
});

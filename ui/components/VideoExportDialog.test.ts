import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import VideoExportDialog from "./VideoExportDialog.vue";

describe("VideoExportDialog", () => {
  it("offers supported formats and emits GIF bounce loop choices", async () => {
    const wrapper = mount(VideoExportDialog, {
      props: {
        open: true,
        filename: "rain.mp4",
        formats: ["gif", "apng", "webp"],
      },
    });

    expect(wrapper.text()).toContain("GIF");
    expect(wrapper.text()).toContain("APNG");
    expect(wrapper.text()).toContain("WEBP");
    await wrapper
      .get('input[name="gif-playback"][value="bounce"]')
      .setValue(true);
    await wrapper.get('input[name="gif-repeat"][value="once"]').setValue(true);
    await wrapper.get("form").trigger("submit");

    expect(wrapper.emitted("export")?.[0]).toEqual([
      {
        format: "gif",
        playback: "bounce",
        repeat: "once",
        max_dimension: 720,
        fps: 12,
      },
    ]);
  });

  it("hides GIF-only controls for APNG", async () => {
    const wrapper = mount(VideoExportDialog, {
      props: { open: true, filename: "rain.mp4", formats: ["gif", "apng"] },
    });
    await wrapper
      .get('input[name="export-format"][value="apng"]')
      .setValue(true);
    expect(wrapper.find('input[name="gif-playback"]').exists()).toBe(false);
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")?.[0]).toEqual([
      {
        format: "apng",
        playback: "loop",
        repeat: "forever",
        max_dimension: 720,
        fps: 12,
      },
    ]);
  });
});

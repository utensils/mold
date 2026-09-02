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

  /**
   * A caller that offers more than one place for the export (the phone's
   * share sheet or its Mold folder) lists them; the choice rides beside the
   * options rather than inside them, so the request body posted to the host
   * never carries a client-side destination.
   */
  it("offers destinations only when asked, and emits the chosen one beside the options", async () => {
    const wrapper = mount(VideoExportDialog, {
      props: {
        open: true,
        filename: "armchair.glb",
        formats: ["gif", "apng"],
        destinations: [
          { value: "share", label: "Share…" },
          { value: "folder", label: "Save to Mold folder" },
        ],
      },
    });

    expect(
      wrapper
        .findAll('input[name="export-destination"]')
        .map((radio) => (radio.element as HTMLInputElement).value),
    ).toEqual(["share", "folder"]);
    expect(wrapper.text()).toContain("Save to Mold folder");
    await wrapper
      .get('input[name="export-destination"][value="folder"]')
      .setValue(true);
    await wrapper.get("form").trigger("submit");

    expect(wrapper.emitted("export")?.[0]).toEqual([
      {
        format: "gif",
        playback: "loop",
        repeat: "forever",
        max_dimension: 720,
        fps: 12,
      },
      "folder",
    ]);

    const plain = mount(VideoExportDialog, {
      props: { open: true, filename: "rain.mp4", formats: ["gif"] },
    });
    expect(plain.find('input[name="export-destination"]').exists()).toBe(false);
    await plain.get("form").trigger("submit");
    expect(plain.emitted("export")?.[0]).toHaveLength(1);
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

import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
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
   * Transparency belongs to a mesh TURNTABLE, which is rendered on request
   * and so can leave its backdrop out. A video re-encode has frames that
   * already exist, and the host refuses `transparent` on one — so the key
   * must be absent from that request, not merely false.
   */
  it("offers a transparent backdrop only for a turntable, and remembers the choice", async () => {
    const values: Record<string, string> = {};
    const storage = {
      getItem: (key: string) => values[key] ?? null,
      setItem: (key: string, value: string) => {
        values[key] = value;
      },
      removeItem: (key: string) => {
        delete values[key];
      },
    };
    vi.stubGlobal("localStorage", storage);

    const video = mount(VideoExportDialog, {
      props: { open: true, filename: "rain.mp4", formats: ["gif"] },
    });
    expect(video.find('[data-test="export-transparent"]').exists()).toBe(false);
    await video.get("form").trigger("submit");
    expect(video.emitted("export")?.[0]?.[0]).not.toHaveProperty("transparent");

    const turntable = () =>
      mount(VideoExportDialog, {
        props: {
          open: true,
          filename: "armchair.glb",
          formats: ["gif", "apng"],
          transparency: true,
        },
      });

    const first = turntable();
    const checkbox = first.get('[data-test="export-transparent"]');
    expect((checkbox.element as HTMLInputElement).checked).toBe(false);
    // Off sends nothing at all, so an untouched turntable posts the body it
    // always did.
    await first.get("form").trigger("submit");
    expect(first.emitted("export")?.[0]?.[0]).not.toHaveProperty("transparent");

    await checkbox.setValue(true);
    await first.get("form").trigger("submit");
    expect(first.emitted("export")?.[1]?.[0]).toMatchObject({
      transparent: true,
    });

    // The next export opens already checked: someone who wants their
    // turntables cut out wants that every time.
    const second = turntable();
    expect(
      (
        second.get('[data-test="export-transparent"]')
          .element as HTMLInputElement
      ).checked,
    ).toBe(true);
    await second.get("form").trigger("submit");
    expect(second.emitted("export")?.[0]?.[0]).toMatchObject({
      transparent: true,
    });

    vi.unstubAllGlobals();
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

  it("shows a failed export in the shared error notice so a long host URL wraps inside the card", () => {
    const message =
      "Couldn't reach the media host: error sending request for url (http://192.168.1.114:7680/api/gallery/export/mold%2Dhunyuan3d%2Dfp16%2D1788359192104.glb)";
    const wrapper = mount(VideoExportDialog, {
      props: {
        open: true,
        filename: "mold-hunyuan3d-fp16-1788359192104.glb",
        formats: ["gif", "apng", "webp"],
        error: message,
      },
    });

    const notice = wrapper.get("[data-test='video-export-error']");
    expect(notice.attributes("role")).toBe("alert");
    expect(notice.get("[data-test='error-notice-message']").text()).toBe(
      message,
    );
    expect(
      notice.get("[data-test='error-notice-message']").classes(),
    ).toContain("ms-error__message");
    expect(notice.find("[data-test='copy-error-notice']").exists()).toBe(true);
  });
  it("preserves zero, parks pause across formats and refuses invalid edits", async () => {
    const wrapper = mount(VideoExportDialog, {
      props: {
        open: true,
        filename: "loop.mp4",
        formats: ["gif", "apng"],
        pauseControl: { min: 0, max: 5000, step: 10, default: 0 },
      },
    });
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")?.[0]?.[0]).toHaveProperty("pause_ms", 0);
    await wrapper.get('[data-test="export-pause"]').setValue(250);
    await wrapper
      .get('input[name="export-format"][value="apng"]')
      .setValue(true);
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")?.[1]?.[0]).not.toHaveProperty("pause_ms");
    await wrapper
      .get('input[name="export-format"][value="gif"]')
      .setValue(true);
    await wrapper.get('input[name="gif-repeat"][value="once"]').setValue(true);
    expect(wrapper.find('[data-test="export-pause"]').exists()).toBe(false);
    await wrapper
      .get('input[name="gif-playback"][value="bounce"]')
      .setValue(true);
    expect(
      (wrapper.get('[data-test="export-pause"]').element as HTMLInputElement)
        .value,
    ).toBe("250");
    await wrapper.get('[data-test="export-pause"]').setValue(11);
    expect(
      wrapper.get('button[type="submit"]').attributes("disabled"),
    ).toBeDefined();
  });

  it("does not submit invalid turntable views or while converting", async () => {
    const wrapper = mount(VideoExportDialog, {
      props: {
        open: true,
        filename: "object.glb",
        formats: ["gif"],
        transparency: true,
      },
    });
    await wrapper.get('input[aria-label="Views per turn"]').setValue(181);
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")).toBeUndefined();
    await wrapper.get('input[aria-label="Views per turn"]').setValue(36);
    await wrapper.setProps({ busy: true });
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")).toBeUndefined();
  });
  it("restores visible video defaults when a mesh viewer becomes a clip", async () => {
    const wrapper = mount(VideoExportDialog, {
      props: {
        open: true,
        filename: "object.glb",
        formats: ["gif"],
        transparency: true,
      },
    });
    await wrapper.setProps({ open: false });
    await wrapper.setProps({
      open: true,
      filename: "loop.mp4",
      transparency: false,
    });
    expect(
      (
        wrapper.get('input[name="export-size"][value="720"]')
          .element as HTMLInputElement
      ).checked,
    ).toBe(true);
    expect(
      (
        wrapper.get('input[name="export-fps"][value="12"]')
          .element as HTMLInputElement
      ).checked,
    ).toBe(true);
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")?.[0]?.[0]).toMatchObject({
      max_dimension: 720,
      fps: 12,
    });
  });
  it("offers only the advertised animation intersection and disables geometry-only hosts", async () => {
    const wrapper = mount(VideoExportDialog, {
      props: {
        open: true,
        filename: "loop.mp4",
        formats: ["gif", "apng", "webp", "obj", "stl"],
      },
    });
    expect(
      wrapper
        .findAll('input[name="export-format"]')
        .map((input) => input.attributes("value")),
    ).toEqual(["gif", "apng", "webp"]);
    await wrapper.setProps({ formats: ["obj", "stl"] });
    expect(wrapper.findAll('input[name="export-format"]')).toHaveLength(0);
    expect(
      wrapper.get('button[type="submit"]').attributes("disabled"),
    ).toBeDefined();
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")).toBeUndefined();
  });
});

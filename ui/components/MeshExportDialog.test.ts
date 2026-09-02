import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MeshExportDialog from "./MeshExportDialog.vue";
import type { MeshExportGeometryCapabilities } from "@studio/lib/meshExport";

/**
 * The chrome around `MeshGeometryFields`. Like the video sheet it is
 * transport-free: it emits the geometry the user settled on, and a
 * destination BESIDE it when the caller offered a choice.
 */

const capabilities: MeshExportGeometryCapabilities = {
  size_mm: { min: 1, max: 1000, default: 100 },
  up_axes: ["y", "z"],
  origins: ["center", "floor"],
  defaults: {
    obj: { size_mm: null, up_axis: "y", origin: "floor" },
    stl: { size_mm: 100, up_axis: "z", origin: "floor" },
  },
};

function mountDialog(props: Record<string, unknown> = {}) {
  return mount(MeshExportDialog, {
    props: {
      open: true,
      filename: "chair.glb",
      format: "stl",
      capabilities,
      ...props,
    },
  });
}

describe("MeshExportDialog", () => {
  it("renders nothing while closed", () => {
    const wrapper = mountDialog({ open: false });
    expect(wrapper.find("[data-test='mesh-export-dialog']").exists()).toBe(
      false,
    );
  });

  it("names the container it is about to write", () => {
    const wrapper = mountDialog();
    expect(wrapper.get("#mesh-export-title").text()).toBe("Export as STL");
    expect(wrapper.text()).toContain("Mesh export");
    expect(wrapper.text()).toContain("chair.glb");
  });

  it("starts from the host's defaults for this format", () => {
    const wrapper = mountDialog();
    const fields = wrapper.getComponent({ name: "MeshGeometryFields" });
    expect(fields.props("modelValue")).toEqual({
      size_mm: 100,
      up_axis: "z",
      origin: "floor",
    });
  });

  it("emits the geometry the user settled on", async () => {
    const wrapper = mountDialog();
    await wrapper.get("[data-test='mesh-geometry-size-200']").setValue();
    await wrapper.get("[data-test='mesh-geometry-origin-center']").setValue();
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")).toEqual([
      [{ size_mm: 200, up_axis: "z", origin: "center" }],
    ]);
  });

  // A size typed for an STL must not ride along into the next OBJ.
  it("resets the draft to the new format's defaults on each opening", async () => {
    const wrapper = mountDialog();
    await wrapper.get("[data-test='mesh-geometry-size-200']").setValue();
    await wrapper.setProps({ open: false });
    await wrapper.setProps({ open: true, format: "obj" });
    const fields = wrapper.getComponent({ name: "MeshGeometryFields" });
    expect(fields.props("modelValue")).toEqual({
      size_mm: null,
      up_axis: "y",
      origin: "floor",
    });
  });

  // The options are the request body; where the file goes is the client's
  // own business, so it rides beside them and never inside them.
  it("carries a chosen destination beside the options", async () => {
    const wrapper = mountDialog({
      destinations: [
        { value: "share", label: "Share…" },
        { value: "folder", label: "Save to Mold folder" },
      ],
    });
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")?.[0]).toEqual([
      { size_mm: 100, up_axis: "z", origin: "floor" },
      "share",
    ]);
  });

  it("omits the destination entirely when there is no choice to make", async () => {
    const wrapper = mountDialog({
      destinations: [{ value: "share", label: "Share…" }],
    });
    await wrapper.get("form").trigger("submit");
    expect(wrapper.emitted("export")?.[0]).toHaveLength(1);
  });

  it("shows the host's own refusal", () => {
    const wrapper = mountDialog({ error: "That size is out of range." });
    expect(wrapper.get("[role='alert']").text()).toBe(
      "That size is out of range.",
    );
  });

  it("locks the sheet while the export runs", () => {
    const wrapper = mountDialog({ busy: true });
    expect(
      wrapper.get("[data-test='mesh-export-submit']").attributes("disabled"),
    ).toBeDefined();
    expect(
      wrapper.getComponent({ name: "MeshGeometryFields" }).props("disabled"),
    ).toBe(true);
  });

  it("closes on Cancel", async () => {
    const wrapper = mountDialog();
    const cancel = wrapper
      .findAll("button")
      .find((button) => button.text() === "Cancel")!;
    await cancel.trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });
});

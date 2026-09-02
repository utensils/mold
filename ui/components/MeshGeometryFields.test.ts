import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MeshGeometryFields from "./MeshGeometryFields.vue";
import type {
  MeshExportGeometryCapabilities,
  MeshGeometryOptions,
} from "@studio/lib/meshExport";

/**
 * The geometry knobs, rendered from the HOST's own contract. Everything here
 * is about not inventing a choice the host never advertised, and about never
 * offering "As stored" for a format whose default is a size — the wire has no
 * key that asks such a format to skip scaling.
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

const stlDefaults: MeshGeometryOptions = {
  size_mm: 100,
  up_axis: "z",
  origin: "floor",
};

/** 1 wide, 0.4286 tall, 0.6857 deep, in the stored Y-up frame. */
const bounds = { min: [-0.5, -0.2143, -0.3429], max: [0.5, 0.2143, 0.3428] };

function mountFields(props: Record<string, unknown> = {}) {
  return mount(MeshGeometryFields, {
    props: {
      modelValue: stlDefaults,
      capabilities,
      format: "stl",
      ...props,
    },
  });
}

describe("MeshGeometryFields", () => {
  it("offers the millimetre presets and the host's own default", () => {
    const wrapper = mountFields();
    const labels = wrapper
      .findAll("[data-test^='mesh-geometry-size-']")
      .map((input) => input.attributes("data-test"));
    expect(labels).toContain("mesh-geometry-size-50");
    expect(labels).toContain("mesh-geometry-size-100");
    expect(labels).toContain("mesh-geometry-size-200");
  });

  it("clamps the presets into the host's range and drops the duplicates", () => {
    const wrapper = mountFields({
      capabilities: {
        ...capabilities,
        size_mm: { min: 10, max: 60, default: 60 },
      },
      modelValue: { ...stlDefaults, size_mm: 60 },
    });
    const chips = wrapper
      .findAll("[data-test^='mesh-geometry-size-']")
      .map((input) => input.attributes("data-test"))
      .filter((name) => name?.match(/^mesh-geometry-size-\d+$/));
    expect(chips).toEqual(["mesh-geometry-size-50", "mesh-geometry-size-60"]);
  });

  // A size-defaulting format cannot ask the host to skip scaling.
  it("offers As stored only where the format's own default is as stored", () => {
    expect(
      mountFields().find("[data-test='mesh-geometry-size-stored']").exists(),
    ).toBe(false);
    const asStored = mountFields({
      format: "obj",
      modelValue: { size_mm: null, up_axis: "y", origin: "floor" },
    });
    expect(
      asStored.find("[data-test='mesh-geometry-size-stored']").exists(),
    ).toBe(true);
  });

  it("emits the picked preset without touching the other knobs", async () => {
    const wrapper = mountFields();
    await wrapper.get("[data-test='mesh-geometry-size-200']").setValue();
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([
      { size_mm: 200, up_axis: "z", origin: "floor" },
    ]);
  });

  it("clamps a typed size into the host's range and writes it back", async () => {
    const wrapper = mountFields();
    const field = wrapper.get<HTMLInputElement>(
      "[data-test='mesh-geometry-size-input']",
    );
    await field.setValue("4000");
    await field.trigger("change");
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([
      { size_mm: 1000, up_axis: "z", origin: "floor" },
    ]);
    expect(field.element.value).toBe("1000");
  });

  it("keeps the current size when the field is left empty", async () => {
    const wrapper = mountFields();
    const field = wrapper.get<HTMLInputElement>(
      "[data-test='mesh-geometry-size-input']",
    );
    field.element.value = "";
    await field.trigger("change");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    expect(field.element.value).toBe("100");
  });

  it("says what the exported file will measure", () => {
    const wrapper = mountFields({ bounds });
    expect(wrapper.get("[data-test='mesh-geometry-size-label']").text()).toBe(
      "100.0 × 68.6 × 42.9 mm",
    );
  });

  it("names the knob when the viewer has reported no box", () => {
    expect(
      mountFields().get("[data-test='mesh-geometry-size-label']").text(),
    ).toBe("longest side 100 mm");
  });

  it("names both axes in the words the destination tools use", () => {
    const wrapper = mountFields();
    expect(wrapper.get("[data-test='mesh-geometry-up-y'] + span").text()).toBe(
      "Y-up · as stored (glTF, Blender OBJ)",
    );
    expect(wrapper.get("[data-test='mesh-geometry-up-z'] + span").text()).toBe(
      "Z-up · slicers, CAD, Blender STL/PLY",
    );
  });

  it("emits the picked axis and origin", async () => {
    const wrapper = mountFields();
    await wrapper.get("[data-test='mesh-geometry-up-y']").setValue();
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([
      { size_mm: 100, up_axis: "y", origin: "floor" },
    ]);
    await wrapper.get("[data-test='mesh-geometry-origin-center']").setValue();
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([
      { size_mm: 100, up_axis: "z", origin: "center" },
    ]);
  });

  // The host is the authority: an axis it does not advertise is not a choice.
  it("renders only the axes and origins the host advertises", () => {
    const wrapper = mountFields({
      capabilities: { ...capabilities, up_axes: ["z"], origins: ["floor"] },
    });
    expect(wrapper.find("[data-test='mesh-geometry-up-y']").exists()).toBe(
      false,
    );
    expect(
      wrapper.find("[data-test='mesh-geometry-origin-center']").exists(),
    ).toBe(false);
  });

  it("disables every control while the export is running", () => {
    const wrapper = mountFields({ disabled: true });
    expect(
      wrapper
        .findAll("fieldset")
        .every((set) => set.attributes("disabled") !== undefined),
    ).toBe(true);
  });
});

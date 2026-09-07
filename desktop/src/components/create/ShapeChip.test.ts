/*
 * The composer's canvas control. It was a caret advertising a menu, wired to
 * an emit whose only handler set `inspectorTab = "settings"` — already the
 * value on mount — so clicking it did nothing; and its label came from
 * `outputFamilyLabel(width, height)` while the inspector's came from
 * `resolveOutputShape`, so the two could state different things about one
 * canvas. What these tests pin is that it now WRITES, and that it says what
 * the resolver says.
 */
import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import ShapeChip from "./ShapeChip.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import { generationCapabilitiesForFamily } from "../../lib/capabilities";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";

function mountChip(form: GenerateForm, family = form.family) {
  setActivePinia(createPinia());
  return mount(ShapeChip, {
    attachTo: document.body,
    props: { form, caps: generationCapabilitiesForFamily(family, form.model) },
  });
}

function squareForm(): GenerateForm {
  const form = reactive(newGenerateForm()) as GenerateForm;
  form.family = "flux";
  form.model = "flux-dev:q4";
  form.width = 1024;
  form.height = 1024;
  return form;
}

async function openMenu(wrapper: ReturnType<typeof mountChip>) {
  await wrapper.get("[data-test='shape-chip']").trigger("click");
  return wrapper;
}

describe("ShapeChip", () => {
  it("labels the canvas from the resolver, not from the raw pixels", async () => {
    expect(mountChip(squareForm()).get("[data-test='shape-chip']").text()).toContain(
      "Square · 1024",
    );

    // An off-ladder size is exactly where the two used to part: the rail
    // marked it approximate and the chip stated it flatly. The chip now
    // carries the same `≈` the rail's picker does, from the same object.
    const wide = squareForm();
    wide.width = 1216;
    wide.height = 704;
    const wrapper = await openMenu(mountChip(wide));
    expect(wrapper.get("[data-test='shape-chip']").text()).toContain("16:9 · ≈1216×704");
    expect(wrapper.findComponent(ShapePicker).props("approximate")).toBe(true);
    wrapper.unmount();
  });

  it("opens a real picker rather than a door to somewhere else", async () => {
    const wrapper = await openMenu(mountChip(squareForm()));
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(true);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(true);
    wrapper.unmount();
  });

  it("writes the form's canvas when a shape is picked, and says what it meant", async () => {
    const form = squareForm();
    const wrapper = await openMenu(mountChip(form));
    const picker = wrapper.findComponent(ShapePicker);
    const wide = (picker.props("options") ?? []).find(
      (option) => option.id !== picker.props("modelValue"),
    );
    if (!wide) throw new Error("the ladder offers only one shape");
    picker.vm.$emit("update:modelValue", wide.id);
    await wrapper.vm.$nextTick();

    expect(form.width / form.height).not.toBe(1);
    expect(wrapper.emitted("canvas-intent")?.[0]).toEqual(["manual"]);
    // And the chip immediately reads back what it just wrote.
    expect(wrapper.get("[data-test='shape-chip']").text()).not.toContain("Square · 1024");
    wrapper.unmount();
  });

  it("writes the form's canvas when a resolution is picked", async () => {
    const form = squareForm();
    const wrapper = await openMenu(mountChip(form));
    const selector = wrapper.findComponent(ResolutionSelector);
    const other = (selector.props("options") ?? []).find(
      (option) => option.id !== selector.props("modelValue"),
    );
    if (!other) throw new Error("the ladder offers only one size");
    selector.vm.$emit("update:modelValue", other.id);
    await wrapper.vm.$nextTick();

    expect(form.width).toBe(other.width);
    expect(form.height).toBe(other.height);
    wrapper.unmount();
  });

  it("shows no chip for a canvasless (3-D) recipe", () => {
    const form = squareForm();
    form.family = "hunyuan3d";
    form.model = "hunyuan3d-2.1:fp16";
    form.width = 0;
    form.height = 0;
    const wrapper = mountChip(form, "hunyuan3d");
    expect(wrapper.find("[data-test='shape-chip']").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("0×0");
  });
});

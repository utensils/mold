import { mount } from "@vue/test-utils";
import { reactive } from "vue";
import { beforeEach, describe, expect, it } from "vitest";
import {
  GENERATION_TEMPLATES_STORAGE_KEY,
  saveGenerationTemplate,
} from "../lib/generationTemplates";
import { newGenerateForm } from "../lib/generateForm";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import MobileTemplates from "./MobileTemplates.vue";
import { MOBILE_GENERATION_TEMPLATES_STORAGE_KEY } from "./mobileTemplateStorage";

installMemoryLocalStorage();

describe("MobileTemplates", () => {
  beforeEach(() => localStorage.clear());

  it("shows saved templates before the disclosure is opened", async () => {
    const form = reactive(newGenerateForm());
    saveGenerationTemplate(
      "Saved earlier",
      form,
      MOBILE_GENERATION_TEMPLATES_STORAGE_KEY,
      "studio-id",
    );

    const wrapper = mount(MobileTemplates, { props: { form, hostId: "studio-id" } });
    await wrapper.vm.$nextTick();

    expect(wrapper.get("[data-test='mobile-template-disclosure']").text()).toContain("1 saved");
  });

  it("saves in an iPhone-only namespace and emits a selected template", async () => {
    const form = reactive(newGenerateForm());
    form.prompt = "violet lightning over the sea";
    form.model = "flux:fp16";
    const wrapper = mount(MobileTemplates, { props: { form, hostId: "studio-id" } });

    await wrapper.get("[data-test='mobile-template-disclosure']").trigger("click");
    await wrapper.get("[data-test='mobile-template-name']").setValue("Storm");
    await wrapper.get("[data-test='mobile-template-save']").trigger("click");

    expect(localStorage.getItem(GENERATION_TEMPLATES_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(MOBILE_GENERATION_TEMPLATES_STORAGE_KEY)).toContain("Storm");
    expect(wrapper.get("[data-test='mobile-template-disclosure']").text()).toContain("1 saved");

    await wrapper.get("[data-test='mobile-template-load']").trigger("click");
    expect(wrapper.emitted("load")?.[0]?.[0]).toMatchObject({
      name: "Storm",
      scopeId: "studio-id",
      form: { prompt: "violet lightning over the sea", model: "flux:fp16" },
    });
  });

  it("searches, renames, and confirms deletion", async () => {
    const form = reactive(newGenerateForm());
    form.prompt = "first prompt";
    const wrapper = mount(MobileTemplates, { props: { form, hostId: "studio-id" } });
    await wrapper.get("[data-test='mobile-template-disclosure']").trigger("click");
    await wrapper.get("[data-test='mobile-template-name']").setValue("First");
    await wrapper.get("[data-test='mobile-template-save']").trigger("click");
    form.prompt = "second prompt";
    await wrapper.get("[data-test='mobile-template-name']").setValue("Second");
    await wrapper.get("[data-test='mobile-template-save']").trigger("click");

    await wrapper.get("[data-test='mobile-template-search']").setValue("first");
    expect(wrapper.findAll("[data-test='mobile-template-row']")).toHaveLength(1);
    await wrapper.get("[data-test='mobile-template-rename']").trigger("click");
    await wrapper.get("[data-test='mobile-template-rename-input']").setValue("Favorite");
    await wrapper.get("[data-test='mobile-template-rename-input']").trigger("keydown.enter");
    expect(wrapper.get("[data-test='mobile-template-row']").text()).toContain("Favorite");

    await wrapper.get("[data-test='mobile-template-delete']").trigger("click");
    await wrapper.get("[data-test='mobile-template-delete']").trigger("click");
    expect(wrapper.findAll("[data-test='mobile-template-row']")).toHaveLength(0);
  });
});

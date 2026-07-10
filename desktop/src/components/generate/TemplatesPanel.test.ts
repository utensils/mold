import { beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import TemplatesPanel from "./TemplatesPanel.vue";
import { installMemoryLocalStorage } from "../../lib/testSupport/memoryLocalStorage";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import {
  loadGenerationTemplates,
  saveGenerationTemplate,
  type GenerationTemplate,
} from "../../lib/generationTemplates";

installMemoryLocalStorage();

function formWith(overrides: Partial<GenerateForm> = {}): GenerateForm {
  return { ...newGenerateForm(), ...overrides };
}

function mountPanel(form: GenerateForm) {
  return mount(TemplatesPanel, { props: { form } });
}

beforeEach(() => {
  localStorage.clear();
});

describe("TemplatesPanel", () => {
  it("shows the empty state when there are no templates", () => {
    const wrapper = mountPanel(formWith());
    expect(wrapper.get('[data-test="template-empty"]').text()).toContain("No templates yet");
    expect(wrapper.find('[data-test="template-row"]').exists()).toBe(false);
  });

  it("saves the current form with a prompt-slice fallback name", async () => {
    const wrapper = mountPanel(
      formWith({ prompt: "a misty harbour at dawn", model: "flux-dev:q8" }),
    );
    await wrapper.get('[data-test="template-save"]').trigger("click");

    const stored = loadGenerationTemplates();
    expect(stored.length).toBe(1);
    expect(stored[0]?.name).toBe("a misty harbour at dawn");
    expect(wrapper.get('[data-test="template-row-name"]').text()).toBe("a misty harbour at dawn");
  });

  it("uses the typed name over the fallback", async () => {
    const wrapper = mountPanel(formWith({ prompt: "whatever" }));
    await wrapper.get('[data-test="template-name"]').setValue("My preset");
    await wrapper.get('[data-test="template-save"]').trigger("click");
    expect(loadGenerationTemplates()[0]?.name).toBe("My preset");
  });

  it("emits load with the clicked template", async () => {
    saveGenerationTemplate("Preset A", formWith({ prompt: "x", model: "sdxl:fp16" }));
    const wrapper = mountPanel(formWith());
    await wrapper.get('[data-test="template-load"]').trigger("click");

    const events = wrapper.emitted("load");
    expect(events).toHaveLength(1);
    const emitted = events?.[0]?.[0] as GenerationTemplate;
    expect(emitted.name).toBe("Preset A");
    expect(emitted.form.prompt).toBe("x");
  });

  it("renames inline via the rename input", async () => {
    saveGenerationTemplate("Old", formWith({ prompt: "x" }));
    const wrapper = mountPanel(formWith());

    await wrapper.get('[data-test="template-rename"]').trigger("click");
    const input = wrapper.get('[data-test="template-rename-input"]');
    await input.setValue("Renamed");
    await input.trigger("keydown.enter");

    expect(loadGenerationTemplates()[0]?.name).toBe("Renamed");
    expect(wrapper.get('[data-test="template-row-name"]').text()).toBe("Renamed");
  });

  it("requires two clicks to delete (confirm affordance)", async () => {
    saveGenerationTemplate("Doomed", formWith({ prompt: "x" }));
    const wrapper = mountPanel(formWith());

    const deleteBtn = wrapper.get('[data-test="template-delete"]');
    await deleteBtn.trigger("click");
    // First click arms the confirm; nothing deleted yet.
    expect(deleteBtn.text()).toContain("Confirm");
    expect(loadGenerationTemplates().length).toBe(1);

    await wrapper.get('[data-test="template-delete"]').trigger("click");
    expect(loadGenerationTemplates().length).toBe(0);
    expect(wrapper.find('[data-test="template-empty"]').exists()).toBe(true);
  });

  it("filters the list by search query", async () => {
    saveGenerationTemplate("Portrait", formWith({ prompt: "a face", model: "flux-dev:q8" }));
    saveGenerationTemplate("Landscape", formWith({ prompt: "hills", model: "sdxl:fp16" }));
    const wrapper = mountPanel(formWith());

    expect(wrapper.findAll('[data-test="template-row"]').length).toBe(2);
    await wrapper.get('[data-test="template-search"]').setValue("hills");
    const rows = wrapper.findAll('[data-test="template-row-name"]');
    expect(rows.length).toBe(1);
    expect(rows[0]?.text()).toBe("Landscape");
  });
});

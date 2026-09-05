/**
 * Starting points as picture cards. A saved starting point has no picture of
 * its own — its media is conditioning input, not a result — so the card draws
 * the style's family mark instead of a thumbnail.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import StarterList from "./StarterList.vue";
import { saveGenerationTemplate } from "../../lib/generationTemplates";
import { newGenerateForm } from "../../lib/generateForm";
import { installMemoryLocalStorage } from "../../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

beforeEach(() => window.localStorage.clear());
afterEach(() => (document.body.innerHTML = ""));

describe("StarterList", () => {
  it("says what an empty shelf means and how to fill it", () => {
    const wrapper = mount(StarterList);
    expect(wrapper.get("[data-test='starter-empty']").text()).toContain("No starting points yet");
    expect(wrapper.find("[data-test='starter-list']").exists()).toBe(true);
    expect(wrapper.findAll("[data-test='starter-card']")).toHaveLength(0);
  });

  it("draws one card per saved starting point and hands it back on click", async () => {
    const form = newGenerateForm();
    form.model = "sdxl-base:fp16";
    form.family = "sdxl";
    saveGenerationTemplate("Rainy windows", form);

    const wrapper = mount(StarterList);
    expect(wrapper.find("[data-test='starter-empty']").exists()).toBe(false);
    const card = wrapper.get("[data-test='starter-card']");
    expect(card.text()).toContain("Rainy windows");
    expect(card.text()).toContain("1024×1024");

    await card.trigger("click");
    expect(wrapper.emitted("load")?.[0]?.[0]).toMatchObject({ name: "Rainy windows" });
  });
});

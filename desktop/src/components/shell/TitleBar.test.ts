import { createPinia, setActivePinia } from "pinia";
import { mount } from "@vue/test-utils";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import TitleBar from "./TitleBar.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useUiStore } from "../../stores/ui";

const stub = { template: "<div />" };

function makeRouter(): Router {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/generate", meta: { title: "Generate" }, component: stub },
      { path: "/gallery", meta: { title: "Gallery" }, component: stub },
    ],
  });
}

async function mountAt(path: string) {
  const router = makeRouter();
  router.push(path);
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  return mount(TitleBar, { global: { plugins: [pinia, router] } });
}

describe("TitleBar", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("names the view from the active route meta", async () => {
    const gallery = await mountAt("/gallery");
    expect(gallery.get("[data-test='shell-title']").text()).toBe("Gallery");
    const generate = await mountAt("/generate");
    expect(generate.get("[data-test='shell-title']").text()).toBe("Generate");
  });

  it("offers back and forward only where history allows", async () => {
    const wrapper = await mountAt("/generate");
    expect(wrapper.get("[aria-label='Back']").attributes("disabled")).toBeDefined();
    expect(wrapper.get("[aria-label='Forward']").attributes("disabled")).toBeDefined();
  });

  it("toggles the persisted sidebar collapse", async () => {
    const wrapper = await mountAt("/generate");
    const prefs = useAppPrefsStore();
    const update = vi.spyOn(prefs, "update").mockResolvedValue();
    await wrapper.get('[aria-label="Toggle sidebar"]').trigger("click");
    expect(update).toHaveBeenCalledWith({ sidebarCollapsed: true });
  });

  it("opens the command palette from the search chip", async () => {
    const wrapper = await mountAt("/generate");
    const ui = useUiStore();
    expect(ui.paletteOpen).toBe(false);
    await wrapper.get('[aria-label="Open command palette"]').trigger("click");
    expect(ui.paletteOpen).toBe(true);
  });
});

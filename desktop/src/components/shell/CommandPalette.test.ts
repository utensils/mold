import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const routerPush = vi.hoisted(() => vi.fn());
vi.mock("vue-router", () => ({ useRouter: () => ({ push: routerPush }) }));
vi.mock("../../lib/api/history", () => ({ fetchHistory: vi.fn().mockResolvedValue([]) }));
vi.mock("../../lib/api/models", () => ({ loadModel: vi.fn(), unloadModel: vi.fn() }));

import CommandPalette from "./CommandPalette.vue";
import { useGalleryStore } from "../../stores/gallery";
import { useUiStore } from "../../stores/ui";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { GalleryImage } from "../../lib/api/types";

beforeEach(() => {
  setActivePinia(createPinia());
  routerPush.mockClear();
});

async function openPalette() {
  const wrapper = mount(CommandPalette, { attachTo: document.body });
  const ui = useUiStore();
  ui.paletteOpen = true;
  await wrapper.vm.$nextTick();
  await wrapper.vm.$nextTick();
  return wrapper;
}

describe("CommandPalette command registry", () => {
  it("navigates to Models for both 'models' and 'catalog' queries", async () => {
    const wrapper = await openPalette();
    const input = wrapper.get("input");

    await input.setValue("models");
    let texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Go to Models"))).toBe(true);

    // Muscle memory: the old "catalog" name still finds the Models entry.
    await input.setValue("catalog");
    texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Go to Models"))).toBe(true);
    expect(texts.some((t) => t.includes("Go to Catalog"))).toBe(false);
    wrapper.unmount();
  });

  it("offers Compose chain and Open history alongside the five workspaces", async () => {
    const wrapper = await openPalette();
    const input = wrapper.get("input");

    await input.setValue("chain");
    let texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Compose chain"))).toBe(true);

    await input.setValue("history");
    texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Open history"))).toBe(true);
    wrapper.unmount();
  });

  it("offers theme + appearance commands wired to the shared prefs plumbing", async () => {
    const wrapper = await openPalette();
    const prefs = useAppPrefsStore();
    const update = vi.spyOn(prefs, "update").mockResolvedValue();

    await wrapper.get("input").setValue("theme");
    const options = wrapper.findAll("[role='option']");
    const texts = options.map((o) => o.text());
    expect(texts.some((t) => t.includes("Theme: Mold"))).toBe(true);
    expect(texts.some((t) => t.includes("Theme: Safelight"))).toBe(true);
    expect(texts.some((t) => t.includes("Appearance: Dark"))).toBe(true);

    await options.find((o) => o.text().includes("Theme: Safelight"))!.trigger("click");
    expect(update).toHaveBeenCalledWith({ themeFamily: "safelight" });
    wrapper.unmount();
  });

  it("no longer offers the retired 'Switch to built-in engine' command", async () => {
    // The built-in engine is always the primary now — there is no remote
    // primary to switch away from; recovery is "Restart engine".
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("engine");
    const texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Switch to built-in engine"))).toBe(false);
    expect(texts.some((t) => t.includes("Restart engine"))).toBe(true);
    wrapper.unmount();
  });
});

describe("CommandPalette gallery results", () => {
  const print = (filename: string, prompt: string, model: string): GalleryImage =>
    ({ filename, timestamp: 1, metadata: { prompt, model, seed: 1 } }) as never;

  it("surfaces matching prints and deep-links to their lightbox", async () => {
    const wrapper = await openPalette();
    useGalleryStore().buckets["local"] = {
      items: [
        print("mold-flux-1.png", "a paper plane at dawn", "flux-dev:q8"),
        print("other.png", "a cat", "sd15:fp16"),
      ],
      loading: false,
      error: null,
      loaded: true,
    };
    await wrapper.get("input").setValue("plane");
    await wrapper.vm.$nextTick();

    const options = wrapper.findAll("[role='option']");
    const match = options.find((o) => o.text().includes("a paper plane at dawn"));
    expect(match).toBeDefined();
    expect(match!.text()).toContain("library");
    expect(options.some((o) => o.text().includes("a cat"))).toBe(false);

    await match!.trigger("click");
    expect(routerPush).toHaveBeenCalledWith("/library?print=mold-flux-1.png");
    wrapper.unmount();
  });

  it("offers no print rows for a blank query", async () => {
    const wrapper = await openPalette();
    useGalleryStore().buckets["local"] = {
      items: [print("mold-flux-1.png", "a paper plane at dawn", "flux-dev:q8")],
      loading: false,
      error: null,
      loaded: true,
    };
    await wrapper.vm.$nextTick();
    const texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("a paper plane at dawn"))).toBe(false);
    wrapper.unmount();
  });
});

describe("CommandPalette a11y semantics", () => {
  it("is a modal dialog wrapping a combobox and listbox", async () => {
    const wrapper = await openPalette();

    const dialog = wrapper.get("[role='dialog']");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Command palette");

    const combobox = wrapper.get("[role='combobox']");
    expect(combobox.attributes("aria-controls")).toBe("cmd-palette-listbox");
    expect(combobox.attributes("aria-expanded")).toBe("true");

    const listbox = wrapper.get("#cmd-palette-listbox");
    expect(listbox.attributes("role")).toBe("listbox");
    wrapper.unmount();
  });

  it("marks options with role=option and points aria-activedescendant at the selection", async () => {
    const wrapper = await openPalette();

    const options = wrapper.findAll("[role='option']");
    expect(options.length).toBeGreaterThan(0);
    expect(options[0]!.attributes("aria-selected")).toBe("true");
    expect(options[0]!.attributes("id")).toBe("cmd-palette-option-0");

    // The combobox's active descendant tracks the highlighted option.
    expect(wrapper.get("[role='combobox']").attributes("aria-activedescendant")).toBe(
      "cmd-palette-option-0",
    );
    wrapper.unmount();
  });
});

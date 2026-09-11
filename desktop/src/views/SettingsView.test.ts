import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";

// The section bodies pull in stores and IPC that aren't the subject here —
// stub them to identifiable markers. What this suite tests is what the VIEW
// still owns after the frame moved to the shared kit: which sections a desktop
// shell renders, which body each one gets, and the `?section=` deep link.
// The frame's own behaviour (scroll-spy, lazy bodies, search, the settling
// hold) is pinned in `studio/components/settings/SettingsShell.test.ts`.
function stub(marker: string) {
  return { default: { template: `<div data-test="${marker}" />` } };
}
vi.mock("../components/settings/AppearanceCard.vue", () => stub("stub-app"));
vi.mock("../components/settings/UpdatesSection.vue", () => stub("stub-updates"));
vi.mock("../components/settings/AboutSection.vue", () => stub("stub-about"));
vi.mock("../components/settings/HostsSection.vue", () => stub("stub-hosts"));
vi.mock("../components/settings/PerformanceSection.vue", () => stub("stub-performance"));
vi.mock("../components/settings/GenerationSection.vue", () => stub("stub-generation"));
vi.mock("../components/settings/MediaSection.vue", () => stub("stub-media"));
vi.mock("../components/settings/StylesDiskSection.vue", () => stub("stub-styles"));
vi.mock("../components/settings/LibrarySection.vue", () => stub("stub-library"));
vi.mock("../components/settings/ExpansionSection.vue", () => stub("stub-expansion"));
vi.mock("../components/settings/AccountsSection.vue", () => stub("stub-accounts"));
vi.mock("../components/settings/CloudSection.vue", () => stub("stub-cloud"));
vi.mock("../components/settings/PerStyleDefaultsSection.vue", () => stub("stub-styleDefaults"));
vi.mock("../components/settings/ProfilesSection.vue", () => stub("stub-profiles"));
vi.mock("../components/settings/AdvancedSection.vue", () => stub("stub-advanced"));
vi.mock("@studio/components/PairingAccessPanel.vue", () => stub("stub-pairing"));
vi.mock("@studio/components/LicenseSettingsPanel.vue", () => stub("stub-licenses"));

import SettingsView from "./SettingsView.vue";
import SettingsShell from "@studio/components/settings/SettingsShell.vue";
import { sectionsForSurface } from "@studio/lib/settingsSchema";
import { useSettingsConfigStore } from "../stores/settingsConfig";

const DESKTOP_SECTIONS = sectionsForSurface("desktop");

const scrollIntoView = vi.fn();
Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
  configurable: true,
  value: scrollIntoView,
});

/** No observer, so the shell mounts every body eagerly — the desktop idiom for
 *  a suite that cares about the bodies rather than about the scroll. */
function withoutObserver() {
  Object.defineProperty(globalThis, "IntersectionObserver", {
    configurable: true,
    writable: true,
    value: undefined,
  });
}

async function mountView(section?: string) {
  const pinia = createPinia();
  setActivePinia(pinia);
  const plugins: unknown[] = [pinia];
  if (section) {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [{ path: "/settings", component: { template: "<div />" } }],
    });
    await router.push({ path: "/settings", query: { section } });
    plugins.push(router);
  }
  const wrapper = mount(SettingsView, { global: { plugins: plugins as never[] } });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  withoutObserver();
});

describe("SettingsView on the shared shell", () => {
  it("renders the sixteen desktop sections in nav order, each with its body", async () => {
    const wrapper = await mountView();
    const navLabels = wrapper.findAll("nav button").map((b) => b.text());
    expect(navLabels).toEqual(DESKTOP_SECTIONS.map((s) => s.label));
    expect(navLabels).toHaveLength(16);
    expect(navLabels[0]).toBe("Look");
    expect(navLabels.at(-1)).toBe("Updates & about");

    for (const id of [
      "app",
      "generation",
      "expansion",
      "hosts",
      "styles",
      "media",
      "library",
      "licenses",
      "pairing",
      "performance",
      "accounts",
      "cloud",
      "styleDefaults",
      "profiles",
      "advanced",
      "updates",
    ]) {
      expect(wrapper.find(`[data-test="section-${id}"]`).exists(), id).toBe(true);
      expect(wrapper.find(`[data-test="stub-${id}"]`).exists(), id).toBe(true);
    }
    // About folded into Updates & about.
    expect(
      wrapper.get("[data-test='section-updates']").find("[data-test='stub-about']").exists(),
    ).toBe(true);
  });

  it("owns its scroll, so the scroll-spy observes the column that moves", async () => {
    // The desktop pane is a fixed height with its own scroller; with the
    // default `scroll="page"` the observer root and the sections would move
    // together and the nav highlight would never change. No unit test can see
    // that (there is no layout here), which is why the prop itself is pinned.
    const wrapper = await mountView();
    expect(wrapper.findComponent(SettingsShell).props("scroll")).toBe("content");
    expect(wrapper.findComponent(SettingsShell).props("layout")).toBe("scroll");
  });

  it("renders no section the desktop shell does not declare", async () => {
    const wrapper = await mountView();
    const declared = new Set(DESKTOP_SECTIONS.map((s) => s.id));
    for (const section of wrapper.findAll("[data-test^='section-']")) {
      const id = section.attributes("data-test")!.replace("section-", "");
      expect(declared.has(id as never), id).toBe(true);
    }
  });

  it("jumps to the section named by ?section= (the Library trash banner's deep link)", async () => {
    const wrapper = await mountView("library");
    expect(wrapper.get("[data-test='settings-nav-library']").attributes("aria-current")).toBe(
      "true",
    );
    expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "start" });
  });

  it("jumps to Updates & about for the native update-check deep link, old name included", async () => {
    for (const section of ["updates", "about"]) {
      scrollIntoView.mockClear();
      const wrapper = await mountView(section);
      expect(wrapper.get("[data-test='settings-nav-updates']").attributes("aria-current")).toBe(
        "true",
      );
      expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "start" });
    }
  });

  it("ignores a ?section= naming something this shell does not render", async () => {
    const wrapper = await mountView("nonesuch");
    expect(wrapper.get("[data-test='settings-nav-app']").attributes("aria-current")).toBe("true");
    expect(scrollIntoView).not.toHaveBeenCalled();
  });

  it("tells the shell which raw keys each section draws, so search can find them", async () => {
    const wrapper = await mountView();
    const config = useSettingsConfigStore();
    config.rows = [
      {
        key: "models.some-style.default_steps",
        value: 28,
        source: "db",
        env_var: null,
        restart_required: false,
      },
    ];
    await flushPromises();
    await wrapper.get("[data-test='settings-search']").setValue("some-style");
    await flushPromises();
    // The style's rows live in Per-style defaults now, so that is the section
    // its name must reach — not the Advanced list it left.
    expect(wrapper.findAll("nav button").map((b) => b.text())).toEqual(["Per-style defaults"]);
  });

  it("says so when the engine exposes no configuration at all", async () => {
    const wrapper = await mountView();
    expect(wrapper.text()).not.toContain("doesn't expose configuration");
    useSettingsConfigStore().available = false;
    await flushPromises();
    expect(wrapper.text()).toContain("doesn't expose configuration");
  });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";

// The section bodies pull in stores and IPC that aren't the subject here —
// stub them to identifiable markers so this suite tests the SettingsView
// shell: the jump nav, the always-open lexicon sections, deep links, and the
// search that narrows both to the sections that match.
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
vi.mock("../components/settings/ProfilesSection.vue", () => stub("stub-profiles"));
vi.mock("../components/settings/AdvancedSection.vue", () => stub("stub-advanced"));
vi.mock("@studio/components/PairingAccessPanel.vue", () => stub("stub-pairing"));
vi.mock("@studio/components/LicenseSettingsPanel.vue", () => stub("stub-licenses"));

import SettingsView from "./SettingsView.vue";
import { SECTIONS } from "../lib/settingsSchema";

const scrollIntoView = vi.fn();
Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
  configurable: true,
  value: scrollIntoView,
});

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

async function typeSearch(wrapper: Awaited<ReturnType<typeof mountView>>, value: string) {
  await wrapper.get("[data-test='settings-search']").setValue(value);
  await flushPromises();
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("SettingsView shell", () => {
  it("renders every lexicon section open, in nav order, with its body", async () => {
    const wrapper = await mountView();
    const navLabels = wrapper.findAll("nav button").map((b) => b.text());
    expect(navLabels).toEqual(SECTIONS.map((s) => s.label));
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

  it("highlights Look first and jumps to a section from the nav", async () => {
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='settings-nav-app']").attributes("aria-current")).toBe("true");

    await wrapper.get("[data-test='settings-nav-library']").trigger("click");
    expect(wrapper.get("[data-test='settings-nav-library']").attributes("aria-current")).toBe(
      "true",
    );
    expect(
      wrapper.get("[data-test='settings-nav-app']").attributes("aria-current"),
    ).toBeUndefined();
    expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "start" });
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

  it("search narrows the nav and the page to the owning section", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "temperature");

    // Write more for me owns expand.temperature — the one section left.
    expect(wrapper.find("[data-test='section-expansion']").exists()).toBe(true);
    expect(wrapper.find("[data-test='stub-expansion']").exists()).toBe(true);
    expect(wrapper.find("[data-test='section-performance']").exists()).toBe(false);
    expect(wrapper.find("[data-test='section-app']").exists()).toBe(false);
    expect(wrapper.findAll("nav button").map((b) => b.text())).toEqual(["Write more for me"]);
  });

  it("finds keyword-only sections (Accounts, Look, Phone pairing) that carry no curated key", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "civitai");
    expect(wrapper.find("[data-test='stub-accounts']").exists()).toBe(true);
    expect(wrapper.find("[data-test='section-expansion']").exists()).toBe(false);

    await typeSearch(wrapper, "theme");
    expect(wrapper.find("[data-test='section-app']").exists()).toBe(true);

    await typeSearch(wrapper, "phone");
    expect(wrapper.find("[data-test='section-pairing']").exists()).toBe(true);
  });

  it("finds My images & trash for trash / retention searches", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "trash");
    expect(wrapper.find("[data-test='section-library']").exists()).toBe(true);
    expect(wrapper.find("[data-test='section-expansion']").exists()).toBe(false);
    await typeSearch(wrapper, "retention");
    expect(wrapper.find("[data-test='stub-library']").exists()).toBe(true);
  });

  it("reports when a search matches nothing", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "zzznope");
    expect(wrapper.find("[data-test='no-search-results']").exists()).toBe(true);
    expect(wrapper.findAll("nav button")).toHaveLength(0);
  });

  it("finds Styles & disk for a directory setting by key or by label", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "models_dir");
    expect(wrapper.find("[data-test='section-styles']").exists()).toBe(true);
    expect(wrapper.find("[data-test='no-search-results']").exists()).toBe(false);

    await typeSearch(wrapper, "finished pictures");
    expect(wrapper.find("[data-test='section-styles']").exists()).toBe(true);

    // Machines keeps its own doorway, findable by the words it still owns.
    await typeSearch(wrapper, "api key");
    expect(wrapper.find("[data-test='section-hosts']").exists()).toBe(true);
  });
});

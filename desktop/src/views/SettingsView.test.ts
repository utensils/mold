import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";
import AccordionSection from "@ui/components/AccordionSection.vue";

// The top cards and accordion bodies pull in stores and IPC that aren't the
// subject here — stub them to identifiable markers so this suite tests the
// SettingsView shell: the single-column structure, one-open-at-a-time
// accordions, and search that opens the owning accordion.
function stub(marker: string) {
  return { default: { template: `<div data-test="${marker}" />` } };
}
vi.mock("../components/settings/AppearanceCard.vue", () => stub("stub-appearance"));
vi.mock("../components/settings/UpdatesSection.vue", () => stub("stub-updates"));
vi.mock("../components/settings/AboutSection.vue", () => stub("stub-about"));
vi.mock("../components/settings/HostsSection.vue", () => stub("stub-hosts"));
vi.mock("../components/settings/PerformanceSection.vue", () => stub("stub-performance"));
vi.mock("../components/settings/GenerationSection.vue", () => stub("stub-generation"));
vi.mock("../components/settings/MediaSection.vue", () => stub("stub-media"));
vi.mock("../components/settings/LibrarySection.vue", () => stub("stub-library"));
vi.mock("../components/settings/ExpansionSection.vue", () => stub("stub-expansion"));
vi.mock("../components/settings/AccountsSection.vue", () => stub("stub-accounts"));
vi.mock("../components/settings/ProfilesSection.vue", () => stub("stub-profiles"));
vi.mock("../components/settings/AdvancedSection.vue", () => stub("stub-advanced"));
vi.mock("@studio/components/PairingAccessPanel.vue", () => stub("stub-paired-access"));

import SettingsView from "./SettingsView.vue";

async function mountView() {
  const pinia = createPinia();
  setActivePinia(pinia);
  const wrapper = mount(SettingsView, { global: { plugins: [pinia] } });
  await flushPromises();
  return wrapper;
}

/** Click a section accordion's header button. */
async function toggleAccordion(wrapper: Awaited<ReturnType<typeof mountView>>, id: string) {
  await wrapper.get(`[data-test="accordion-${id}"]`).get("button").trigger("click");
}

async function typeSearch(wrapper: Awaited<ReturnType<typeof mountView>>, value: string) {
  await wrapper.get("[data-test='settings-search']").setValue(value);
  await flushPromises();
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("SettingsView shell", () => {
  it("renders the top cards, the hosts doorway, and every accordion", async () => {
    const wrapper = await mountView();
    expect(wrapper.find("[data-test='appearance-card']").exists()).toBe(true);
    expect(wrapper.find("[data-test='updates-card']").exists()).toBe(true);
    expect(wrapper.find("[data-test='about-card']").exists()).toBe(true);
    // Hosts is a doorway to the Machines workspace, not a full section here.
    expect(wrapper.find("[data-test='hosts-region']").exists()).toBe(true);
    expect(wrapper.find("[data-test='stub-hosts']").exists()).toBe(true);

    for (const id of [
      "performance",
      "generation",
      "media",
      "library",
      "expansion",
      "accounts",
      "profiles",
      "advanced",
    ]) {
      expect(wrapper.find(`[data-test="accordion-${id}"]`).exists()).toBe(true);
    }
  });

  it("renders icon-led, accented All settings groups with explanatory summaries", async () => {
    const wrapper = await mountView();
    const accordions = wrapper.findAllComponents(AccordionSection);

    expect(accordions).toHaveLength(8);
    for (const accordion of accordions) {
      expect(accordion.props("icon")).toBeTruthy();
      expect(accordion.props("summary")).toBeTruthy();
      expect(accordion.props("tone")).toBe("halide");
    }
  });

  it("keeps every accordion closed by default — nothing blocks first use", async () => {
    const wrapper = await mountView();
    for (const id of [
      "performance",
      "generation",
      "media",
      "library",
      "expansion",
      "accounts",
      "profiles",
      "advanced",
    ]) {
      expect(wrapper.find(`[data-test="stub-${id}"]`).exists()).toBe(false);
    }
  });

  it("opens one accordion at a time", async () => {
    const wrapper = await mountView();
    await toggleAccordion(wrapper, "performance");
    expect(wrapper.find("[data-test='stub-performance']").exists()).toBe(true);

    await toggleAccordion(wrapper, "generation");
    expect(wrapper.find("[data-test='stub-generation']").exists()).toBe(true);
    // Opening generation closes performance.
    expect(wrapper.find("[data-test='stub-performance']").exists()).toBe(false);

    // Clicking the open one again closes it.
    await toggleAccordion(wrapper, "generation");
    expect(wrapper.find("[data-test='stub-generation']").exists()).toBe(false);
  });

  it("opens the accordion named by ?section= (the Library trash banner's deep link)", async () => {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [{ path: "/settings", component: { template: "<div />" } }],
    });
    await router.push({ path: "/settings", query: { section: "library" } });
    const pinia = createPinia();
    setActivePinia(pinia);
    const wrapper = mount(SettingsView, { global: { plugins: [pinia, router] } });
    await flushPromises();
    expect(wrapper.find("[data-test='stub-library']").exists()).toBe(true);
    expect(wrapper.find("[data-test='stub-performance']").exists()).toBe(false);
  });

  it("search opens the owning accordion and hides the rest", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "temperature");

    // The expansion accordion owns expand.temperature — shown and open.
    expect(wrapper.find("[data-test='accordion-expansion']").exists()).toBe(true);
    expect(wrapper.find("[data-test='stub-expansion']").exists()).toBe(true);
    // Non-matching accordions drop out entirely.
    expect(wrapper.find("[data-test='accordion-performance']").exists()).toBe(false);
    expect(wrapper.find("[data-test='accordion-generation']").exists()).toBe(false);
    // Top cards collapse away while searching to focus the results.
    expect(wrapper.find("[data-test='appearance-card']").exists()).toBe(false);
  });

  it("finds keyword-only sections (Accounts) that carry no curated key", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "civitai");
    expect(wrapper.find("[data-test='stub-accounts']").exists()).toBe(true);
    expect(wrapper.find("[data-test='accordion-expansion']").exists()).toBe(false);
  });

  it("opens the Library accordion for trash / retention searches", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "trash");
    expect(wrapper.find("[data-test='accordion-library']").exists()).toBe(true);
    expect(wrapper.find("[data-test='stub-library']").exists()).toBe(true);
    expect(wrapper.find("[data-test='accordion-expansion']").exists()).toBe(false);
    await typeSearch(wrapper, "retention");
    expect(wrapper.find("[data-test='stub-library']").exists()).toBe(true);
  });

  it("reports when a search matches nothing", async () => {
    const wrapper = await mountView();
    await typeSearch(wrapper, "zzznope");
    expect(wrapper.find("[data-test='no-search-results']").exists()).toBe(true);
  });

  it("surfaces the Hosts doorway when a search matches a host-owned setting", async () => {
    const wrapper = await mountView();
    // models_dir / output_dir belong to the Hosts section, which is the doorway
    // card (excluded from the accordions) and hidden while searching.
    await typeSearch(wrapper, "models_dir");
    expect(wrapper.find("[data-test='hosts-region']").exists()).toBe(true);
    expect(wrapper.find("[data-test='stub-hosts']").exists()).toBe(true);
    // A host-owned match is a real result, not "nothing matches".
    expect(wrapper.find("[data-test='no-search-results']").exists()).toBe(false);

    // The human label works too.
    await typeSearch(wrapper, "Output directory");
    expect(wrapper.find("[data-test='hosts-region']").exists()).toBe(true);
  });
});

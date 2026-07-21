import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import HostRoutingPicker from "./HostRoutingPicker.vue";
import { ORIGIN_HOST_ID } from "../../lib/hostRegistry";
import {
  AUTO_TARGET_ID,
  CAPABLE_TARGET_ID,
  type RoutableHost,
} from "../../lib/hostRouting";

function host(overrides: Partial<RoutableHost> & { id: string }): RoutableHost {
  return {
    label: overrides.id,
    url: `http://${overrides.id}:7680`,
    status: "ready",
    queueDepth: 0,
    gpu: null,
    ...overrides,
  };
}

const origin = host({
  id: ORIGIN_HOST_ID,
  label: "this server",
  url: "http://localhost:7680",
});
const studio = host({ id: "studio", label: "Studio", queueDepth: 3 });

function factory(hosts: RoutableHost[], targetId = AUTO_TARGET_ID) {
  return mount(HostRoutingPicker, { props: { hosts, targetId } });
}

describe("HostRoutingPicker", () => {
  it("collapses to an honest row with only this server", () => {
    const wrapper = factory([origin]);
    const row = wrapper.get("[data-test='controls-host']");
    expect(row.text()).toContain("Run on this server");
    expect(wrapper.find("[data-test='host-chip']").exists()).toBe(false);
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("doorways into Machines from the collapsed row", async () => {
    const wrapper = factory([origin]);
    await wrapper.get("[data-test='controls-host']").trigger("click");
    expect(wrapper.emitted("open-machines")).toHaveLength(1);
  });

  it("offers Auto, Most capable and every host once a remote is registered", async () => {
    const wrapper = factory([origin, studio]);
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
    await wrapper.get("[data-test='host-chip']").trigger("click");

    const menu = wrapper.get("[data-test='host-menu']");
    expect(menu.find("[data-test='host-option-auto']").exists()).toBe(true);
    expect(menu.find("[data-test='host-option-capable']").exists()).toBe(true);
    expect(
      menu.get(`[data-test='host-option-${ORIGIN_HOST_ID}']`).text(),
    ).toContain("this server");
    expect(menu.get("[data-test='host-option-studio']").text()).toContain(
      "Studio",
    );
  });

  it("lists this server first", async () => {
    const wrapper = factory([origin, studio]);
    await wrapper.get("[data-test='host-chip']").trigger("click");
    const rows = wrapper
      .get("[data-test='host-menu']")
      .findAll("[role='menuitemradio']");
    // Auto, Most capable, then the hosts in registry order (origin first).
    expect(rows[2]?.attributes("data-test")).toBe(
      `host-option-${ORIGIN_HOST_ID}`,
    );
    expect(rows[3]?.attributes("data-test")).toBe("host-option-studio");
  });

  it("shows each host's live queue depth", async () => {
    const wrapper = factory([origin, studio]);
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.get("[data-test='host-option-studio']").text()).toContain(
      "queue 3",
    );
  });

  it("emits the pick and closes the menu", async () => {
    const wrapper = factory([origin, studio]);
    await wrapper.get("[data-test='host-chip']").trigger("click");
    await wrapper.get("[data-test='host-option-studio']").trigger("click");
    expect(wrapper.emitted("select")).toEqual([["studio"]]);
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("emits the Most capable sentinel", async () => {
    const wrapper = factory([origin, studio]);
    await wrapper.get("[data-test='host-chip']").trigger("click");
    await wrapper.get("[data-test='host-option-capable']").trigger("click");
    expect(wrapper.emitted("select")).toEqual([[CAPABLE_TARGET_ID]]);
  });

  it("names the current pick on the chip", () => {
    expect(factory([origin, studio], AUTO_TARGET_ID).text()).toContain(
      "Run on Auto",
    );
    expect(factory([origin, studio], CAPABLE_TARGET_ID).text()).toContain(
      "Run on Most capable",
    );
    expect(factory([origin, studio], "studio").text()).toContain(
      "Run on Studio",
    );
  });

  it("checks the current pick in the menu", async () => {
    const wrapper = factory([origin, studio], "studio");
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(
      wrapper
        .get("[data-test='host-option-studio']")
        .attributes("aria-checked"),
    ).toBe("true");
    expect(
      wrapper.get("[data-test='host-option-auto']").attributes("aria-checked"),
    ).toBe("false");
  });

  it("disables an offline host and reports it as offline", async () => {
    const offline = { ...studio, status: "error" as const };
    const wrapper = factory([origin, offline]);
    await wrapper.get("[data-test='host-chip']").trigger("click");
    const row = wrapper.get("[data-test='host-option-studio']");
    expect(row.attributes("disabled")).toBeDefined();
    expect(row.text()).toContain("offline");
  });

  it("says a pinned host is offline on the chip rather than claiming ready", () => {
    const offline = { ...studio, status: "error" as const };
    const wrapper = factory([origin, offline], "studio");
    expect(wrapper.get("[data-test='host-chip']").text()).toContain("offline");
  });
});

import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobileAddMachineSheet from "./MobileAddMachineSheet.vue";
import type { DiscoveredHost } from "./hosts";

const NEARBY: DiscoveredHost = {
  name: "Studio Mac",
  host: "studio.local",
  port: 7680,
  authRequired: true,
};

function sheet(props: Partial<Record<string, unknown>> = {}) {
  return mount(MobileAddMachineSheet, {
    attachTo: document.body,
    props: {
      open: true,
      pairing: false,
      discovering: false,
      discovered: [],
      selectedDiscovered: null,
      name: "",
      address: "",
      apiKey: "",
      ...props,
    },
  });
}

describe("MobileAddMachineSheet", () => {
  it("holds all three ways in, and is closed until asked for", async () => {
    const wrapper = sheet({ open: false });

    expect(wrapper.get("[data-test='mobile-add-machine']").attributes("inert")).toBeDefined();
    expect(wrapper.get("[data-test='mobile-add-machine']").classes()).not.toContain("is-open");

    await wrapper.setProps({ open: true });
    expect(wrapper.get("[data-test='mobile-add-machine']").classes()).toContain("is-open");
    // A pairing code, something nearby, or an address typed by hand.
    expect(wrapper.find("[data-test='mobile-scan-pairing']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-discover-hosts']").exists()).toBe(true);
    expect(wrapper.find(".mobile-host-form").exists()).toBe(true);
    wrapper.unmount();
  });

  it("takes the shared sheet chrome, so it dismisses the way every other sheet does", () => {
    const wrapper = sheet();

    expect(wrapper.get("[data-test='mobile-add-machine']").attributes("aria-modal")).toBe("true");
    expect(wrapper.find(".mobile-sheet-grabber").exists()).toBe(true);
    expect(wrapper.get(".mobile-sheet-title").text()).toBe("Add a machine");

    wrapper.get(".mobile-sheet-scrim").trigger("click");
    wrapper.get("[data-test='mobile-add-machine-done']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(2);
    wrapper.unmount();
  });

  it("names the two remote actions rather than performing them itself", async () => {
    const wrapper = sheet({ discovered: [NEARBY] });

    await wrapper.get("[data-test='mobile-scan-pairing']").trigger("click");
    await wrapper.get("[data-test='mobile-discover-hosts']").trigger("click");
    await wrapper.get("[data-test='mobile-discovered-host'] button").trigger("click");

    expect(wrapper.emitted("scan-pairing")).toHaveLength(1);
    expect(wrapper.emitted("discover")).toHaveLength(1);
    expect(wrapper.emitted("pick-discovered")).toEqual([[NEARBY]]);
    wrapper.unmount();
  });

  it("says what it is doing while the camera and the scan are busy", () => {
    const wrapper = sheet({ pairing: true, discovering: true });

    const scan = wrapper.get("[data-test='mobile-scan-pairing']");
    expect(scan.text()).toContain("Opening camera…");
    expect(scan.attributes("disabled")).toBeDefined();
    const discover = wrapper.get("[data-test='mobile-discover-hosts']");
    expect(discover.text()).toBe("Scanning…");
    expect(discover.attributes("disabled")).toBeDefined();
    wrapper.unmount();
  });

  it("edits the machine fields in place, so the app keeps owning them", async () => {
    const wrapper = sheet();

    await wrapper.get(".mobile-host-form input[autocomplete='url']").setValue("plato:7680");
    expect(wrapper.emitted("update:address")).toEqual([["plato:7680"]]);

    await wrapper.get(".mobile-host-form").trigger("submit");
    expect(wrapper.emitted("connect-manual")).toHaveLength(1);
    wrapper.unmount();
  });

  it("asks for a key when a discovered machine requires one, and can focus it", async () => {
    const wrapper = sheet({ selectedDiscovered: NEARBY });
    await flushPromises();

    expect(wrapper.find(".mobile-host-form").exists()).toBe(false);
    const key = wrapper.get("[data-test='mobile-discovered-api-key']");
    (wrapper.vm as unknown as { focusDiscoveredApiKey: () => void }).focusDiscoveredApiKey();
    expect(document.activeElement).toBe(key.element);

    await wrapper.get("[data-test='mobile-discovered-key-prompt']").trigger("submit");
    expect(wrapper.emitted("connect-discovered")).toHaveLength(1);
    wrapper.unmount();
  });

  it("shows why a machine could not be saved, inside the sheet", async () => {
    const wrapper = sheet();
    expect(wrapper.find("[role='alert']").exists()).toBe(false);

    // A fixed overlay with a scrim renders its own failure behind itself
    // otherwise: the explanation lands on the screen you cannot see.
    await wrapper.setProps({ error: "Couldn't reach plato. Check the connection." });
    const alert = wrapper.get("[data-test='mobile-add-machine-error']");
    expect(alert.attributes("role")).toBe("alert");
    expect(alert.text()).toBe("Couldn't reach plato. Check the connection.");
    expect(alert.element.closest(".mobile-sheet-panel")).not.toBeNull();
    wrapper.unmount();
  });
  it("takes focus on open, traps Tab, and returns focus after Escape", async () => {
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();
    const wrapper = sheet({ open: false });
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-add-machine']").attributes("inert")).toBeDefined();

    await wrapper.setProps({ open: true });
    await flushPromises();
    // aria-modal over a background that is not inert is kept by this alone.
    expect(document.activeElement).toBe(wrapper.get(".mobile-sheet-panel").element);

    // The first Tab always arrives with the panel focused.
    await wrapper.get("[data-test='mobile-add-machine']").trigger("keydown", { key: "Tab" });
    expect(wrapper.get(".mobile-sheet-panel").element.contains(document.activeElement)).toBe(true);

    await wrapper.get("[data-test='mobile-add-machine']").trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("close")).toHaveLength(1);

    await wrapper.setProps({ open: false });
    expect(document.activeElement).toBe(trigger);
    wrapper.unmount();
    trigger.remove();
  });
});

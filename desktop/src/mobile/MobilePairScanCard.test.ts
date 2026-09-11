import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobilePairScanCard from "./MobilePairScanCard.vue";

describe("MobilePairScanCard", () => {
  it("asks to scan the desktop's code and says where to find it", () => {
    const wrapper = mount(MobilePairScanCard);

    expect(wrapper.get("[data-test='mobile-pair-scan-card']").text()).toContain(
      "Pair with your desktop",
    );
    // The phone SCANS; it never displays a code of its own, so the tile is a
    // scanner affordance and carries no credential.
    expect(wrapper.get(".mobile-pair-scan-glyph").text()).toBe("QR");
    expect(wrapper.text()).toContain("Open Settings → Mobile pairing on that machine.");
  });

  it("never claims the two devices must share a network", () => {
    // Pairing works over Tailscale, so "same network" would be a false refusal
    // for the setup this app is most often used in.
    const wrapper = mount(MobilePairScanCard);
    expect(wrapper.text()).not.toMatch(/same (network|Wi-Fi)/i);
  });

  it("asks its host to open the camera, and says so with a finger-sized control", () => {
    const wrapper = mount(MobilePairScanCard);

    const scan = wrapper.get("[data-test='mobile-pair-scan']");
    expect(scan.text()).toBe("Scan pairing code");
    expect(scan.attributes("type")).toBe("button");

    scan.trigger("click");
    expect(wrapper.emitted("scan")).toHaveLength(1);
  });

  it("disables the scan while the camera is already open", async () => {
    const wrapper = mount(MobilePairScanCard, { props: { scanning: true } });

    const scan = wrapper.get("[data-test='mobile-pair-scan']");
    expect(scan.attributes("disabled")).toBeDefined();
    await scan.trigger("click");
    expect(wrapper.emitted("scan")).toBeUndefined();
  });

  it("says why a scan failed, where the scan was started", () => {
    // Pairing failures used to set an error that only the Machines tab
    // rendered, so a scan begun in Settings failed in silence.
    const wrapper = mount(MobilePairScanCard, {
      props: { error: "That pairing code has expired. Show a new one." },
    });

    const alert = wrapper.get("[data-test='mobile-pair-scan-error']");
    expect(alert.attributes("role")).toBe("alert");
    expect(alert.text()).toBe("That pairing code has expired. Show a new one.");
  });

  it("says nothing when there is nothing wrong", () => {
    const wrapper = mount(MobilePairScanCard);
    expect(wrapper.find("[data-test='mobile-pair-scan-error']").exists()).toBe(false);
  });
});

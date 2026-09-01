import { nextTick } from "vue";
import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useLicenseAcceptance } from "../composables/useLicenseAcceptance";
import LicenseAcceptanceDialog from "./LicenseAcceptanceDialog.vue";

const licenses = useLicenseAcceptance();

afterEach(() => licenses.cancel());

describe("LicenseAcceptanceDialog", () => {
  it("shows exact-host generic terms and cancellation resumes nothing", async () => {
    const result = licenses.request({
      hostLabel: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: "secret" },
      requirements: [
        {
          installModel: "future-face-adapter",
          licenses: [
            {
              id: "future-license",
              name: "Future model terms",
              url: "https://example.test/pinned",
              canonical: "https://example.test/project",
              sha256: "b".repeat(64),
              summary: "Non-commercial research only.",
            },
          ],
        },
      ],
    });
    const openExternal = vi.fn();
    const wrapper = mount(LicenseAcceptanceDialog, {
      attachTo: document.body,
      props: { openExternal },
    });

    expect(wrapper.text()).toContain("hal9000");
    expect(wrapper.text()).toContain("future-face-adapter");
    expect(wrapper.text()).toContain("Future model terms");
    expect(wrapper.get('a[href="https://example.test/pinned"]').text()).toBe(
      "Pinned terms",
    );
    expect(wrapper.get('a[href="https://example.test/project"]').text()).toBe(
      "Project terms",
    );
    await wrapper.get('a[href="https://example.test/pinned"]').trigger("click");
    expect(openExternal).toHaveBeenCalledWith("https://example.test/pinned");

    await wrapper.get("button.license-secondary").trigger("click");
    await expect(result).resolves.toEqual({
      accepted: false,
      downloaded: false,
    });
    wrapper.unmount();
  });

  it("never promises a download for a record-only acceptance", async () => {
    const wrapper = mount(LicenseAcceptanceDialog);
    const result = licenses.request({
      hostLabel: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: null },
      intent: "record",
      requirements: [
        {
          installModel: "future-face-adapter",
          licenses: [
            {
              id: "future-license",
              name: "Future model terms",
              url: "https://example.test/pinned",
              canonical: "https://example.test/project",
              sha256: "b".repeat(64),
              summary: "Research only.",
            },
          ],
        },
      ],
    });
    await nextTick();

    // Settings' "Accept terms" records consent and downloads nothing; saying
    // otherwise asks the user to agree to an operation that will not happen.
    expect(wrapper.text()).toContain("Accept terms");
    expect(wrapper.text()).not.toContain("Accept terms and download");
    expect(wrapper.text()).not.toContain("Required download");

    await wrapper.get("button.license-secondary").trigger("click");
    await expect(result).resolves.toEqual({
      accepted: false,
      downloaded: false,
    });
    wrapper.unmount();
  });
});

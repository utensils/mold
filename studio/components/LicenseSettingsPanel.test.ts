import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import LicenseSettingsPanel from "./LicenseSettingsPanel.vue";

const { fetchLicenseListing, request } = vi.hoisted(() => ({
  fetchLicenseListing: vi.fn(),
  request: vi.fn().mockResolvedValue(false),
}));

vi.mock("../api/licenseAcceptance", () => ({ fetchLicenseListing }));
vi.mock("../composables/useLicenseAcceptance", () => ({
  useLicenseAcceptance: () => ({ request }),
}));

const terms = (id: string, requiredBy: string[]) => ({
  id,
  name: `Terms ${id}`,
  url: `https://example.test/${id}/pinned`,
  canonical: `https://example.test/${id}`,
  sha256: id.repeat(64).slice(0, 64),
  summary: "Restricted use.",
  accepted: false,
  required_by: requiredBy,
});

afterEach(() => {
  fetchLicenseListing.mockReset();
  request.mockClear();
});

describe("LicenseSettingsPanel", () => {
  it("groups every outstanding term for a future install bundle", async () => {
    fetchLicenseListing.mockResolvedValue({
      licenses: [terms("a", ["future-bundle"]), terms("b", ["future-bundle"])],
    });
    const wrapper = mount(LicenseSettingsPanel, {
      props: {
        target: { baseUrl: "http://host-b:7680", apiKey: "b" },
        hostLabel: "host-b",
      },
    });
    await flushPromises();

    await wrapper.get(".license-settings__actions button").trigger("click");
    expect(request).toHaveBeenCalledWith(
      expect.objectContaining({
        hostLabel: "host-b",
        target: { baseUrl: "http://host-b:7680", apiKey: "b" },
        requirements: [
          expect.objectContaining({
            installModel: "future-bundle",
            licenses: [
              expect.objectContaining({ id: "a" }),
              expect.objectContaining({ id: "b" }),
            ],
          }),
        ],
      }),
    );
  });

  it("fences a slow old-host response after the selected host changes", async () => {
    let resolveA!: (value: unknown) => void;
    let resolveB!: (value: unknown) => void;
    fetchLicenseListing
      .mockReturnValueOnce(new Promise((resolve) => (resolveA = resolve)))
      .mockReturnValueOnce(new Promise((resolve) => (resolveB = resolve)));
    const wrapper = mount(LicenseSettingsPanel, {
      props: {
        target: { baseUrl: "http://host-a:7680", apiKey: "a" },
        hostLabel: "host-a",
      },
    });
    await wrapper.setProps({
      target: { baseUrl: "http://host-b:7680", apiKey: "b" },
      hostLabel: "host-b",
    });
    resolveB({ licenses: [terms("b", ["bundle-b"])] });
    await flushPromises();
    resolveA({ licenses: [terms("a", ["bundle-a"])] });
    await flushPromises();

    expect(wrapper.text()).toContain("Terms b");
    expect(wrapper.text()).not.toContain("Terms a");
  });

  it("keeps a failed host visible with an explicit retry", async () => {
    fetchLicenseListing
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ licenses: [] });
    const wrapper = mount(LicenseSettingsPanel, {
      props: {
        target: { baseUrl: "http://host-b:7680", apiKey: "b" },
        hostLabel: "host-b",
      },
    });
    await flushPromises();
    expect(wrapper.get("[role='alert']").text()).toContain("host-b");

    await wrapper.get("[role='alert'] button").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("no third-party model licenses");
  });
});

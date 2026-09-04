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

    // Either row opens the ONE dialog covering everything the bundle needs.
    const rows = wrapper.findAll(".license-settings__row");
    const pending = rows.find((row) =>
      row.text().includes("a · Restricted use."),
    )!;
    expect(pending.text()).toContain("Needs your OK");
    await pending.get("button").trigger("click");
    expect(request).toHaveBeenCalledWith(
      expect.objectContaining({
        hostLabel: "host-b",
        target: { baseUrl: "http://host-b:7680", apiKey: "b" },
        intent: "record",
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

  it("renders one row per licence: mono id, a state word, and a two-word action", async () => {
    fetchLicenseListing.mockResolvedValue({
      licenses: [
        { ...terms("a", ["bundle-a"]), accepted: true },
        terms("b", ["bundle-b"]),
      ],
    });
    const wrapper = mount(LicenseSettingsPanel, {
      props: {
        target: { baseUrl: "http://h:7680", apiKey: null },
        hostLabel: "h",
      },
    });
    await flushPromises();

    const rows = wrapper
      .findAll(".license-settings__row")
      .filter((row) => row.find(".license-settings__id").exists());
    expect(rows).toHaveLength(2);
    expect(rows[0]!.get(".license-settings__id").text()).toBe(
      "a · Restricted use.",
    );
    expect(rows[0]!.get(".license-settings__state").text()).toBe("Accepted");
    expect(rows[0]!.get("button").text()).toBe("View terms");
    // Pending is a warning, never an error: nothing has gone wrong yet.
    expect(rows[1]!.get(".license-settings__state").classes()).toContain(
      "license-settings__state--pending",
    );
    expect(rows[1]!.get(".license-settings__state").text()).toBe(
      "Needs your OK",
    );
    expect(rows[1]!.get("button").text()).toBe("Read & accept");
  });

  it("shows the machine the answers belong to in the slot the surface fills", async () => {
    fetchLicenseListing.mockResolvedValue({ licenses: [] });
    const wrapper = mount(LicenseSettingsPanel, {
      props: {
        target: { baseUrl: "http://h:7680", apiKey: null },
        hostLabel: "h",
      },
      slots: { machine: '<span data-test="picker">studio-rack</span>' },
    });
    await flushPromises();
    expect(wrapper.get("[data-test='picker']").text()).toBe("studio-rack");
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

    expect(wrapper.text()).toContain("b · Restricted use.");
    expect(wrapper.text()).not.toContain("a · Restricted use.");
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

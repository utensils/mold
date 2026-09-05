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

  it("renders one row per licence: what it unlocks, the licence in mono, a state word", async () => {
    fetchLicenseListing.mockResolvedValue({
      licenses: [
        {
          ...terms("a", ["bundle-a"]),
          accepted: true,
          required_by_styles: [
            { name: "bundle-a", description: "Faces that stay themselves" },
          ],
        },
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
    // The row leads with what the licence unlocks, in the registry's plain
    // words; the licence's own name and summary sit under it in mono, and
    // the id — the machine's handle — rides the tooltip.
    expect(rows[0]!.get(".license-settings__name").text()).toBe(
      "Faces that stay themselves",
    );
    expect(rows[0]!.get(".license-settings__id").text()).toBe(
      "Terms a · Restricted use.",
    );
    expect(rows[0]!.get(".license-settings__what").attributes("title")).toBe(
      "a",
    );
    // An older host lists no styles: the style ids lead instead, and the
    // licence's name and summary still sit beneath them.
    expect(rows[1]!.get(".license-settings__name").text()).toBe("bundle-b");
    expect(rows[1]!.get(".license-settings__id").text()).toBe(
      "Terms b · Restricted use.",
    );
    expect(rows[0]!.get(".license-settings__state").text()).toBe("Accepted");
    // Nothing to accept — the row is its two links and nothing else.
    expect(rows[0]!.find("button").exists()).toBe(false);
    // Pending is a warning, never an error: nothing has gone wrong yet.
    expect(rows[1]!.get(".license-settings__state").classes()).toContain(
      "license-settings__state--pending",
    );
    expect(rows[1]!.get(".license-settings__state").text()).toBe(
      "Needs your OK",
    );
    expect(rows[1]!.get("button").text()).toBe("Read & accept");
  });

  it("links both the pinned terms and the project's own, through the shell", async () => {
    const openExternal = vi.fn();
    fetchLicenseListing.mockResolvedValue({
      licenses: [{ ...terms("a", ["bundle-a"]), accepted: true }],
    });
    const wrapper = mount(LicenseSettingsPanel, {
      props: {
        target: { baseUrl: "http://h:7680", apiKey: null },
        hostLabel: "h",
        openExternal,
      },
    });
    await flushPromises();

    const links = wrapper.findAll(".license-settings__link");
    expect(links.map((link) => link.text())).toEqual([
      "Pinned terms",
      "Project terms",
    ]);
    expect(links[0]!.attributes("href")).toBe("https://example.test/a/pinned");
    expect(links[1]!.attributes("href")).toBe("https://example.test/a");

    // Real links, but the shell opens them: an in-app navigation would
    // replace the app with a licence page.
    await links[1]!.trigger("click");
    expect(openExternal).toHaveBeenCalledWith("https://example.test/a");
  });

  it("a pending row still records acceptance only, never a download", async () => {
    fetchLicenseListing.mockResolvedValue({
      licenses: [terms("b", ["bundle-b"])],
    });
    const wrapper = mount(LicenseSettingsPanel, {
      props: {
        target: { baseUrl: "http://h:7680", apiKey: null },
        hostLabel: "h",
      },
    });
    await flushPromises();
    await wrapper.get(".license-settings__row button").trigger("click");
    expect(request).toHaveBeenCalledWith(
      expect.objectContaining({ intent: "record" }),
    );
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

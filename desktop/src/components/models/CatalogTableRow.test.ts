import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import CatalogTableRow from "./CatalogTableRow.vue";
import { resolveEntrySize } from "../../lib/catalogSizes";
import type { CatalogEntry } from "../../lib/api/types";

vi.mock("../../lib/catalogSizes", () => ({
  resolveEntrySize: vi.fn(),
}));
vi.mock("../../lib/openExternal", () => ({
  openExternal: vi.fn(),
}));

const resolveEntrySizeMock = vi.mocked(resolveEntrySize);

function entry(part: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    id: "cv:8001",
    source: "civitai",
    source_id: "8001",
    name: "Test Flux LoRA",
    author: "alice",
    family: "flux",
    kind: "lora",
    nsfw: false,
    installed: false,
    size_bytes: null,
    download_count: 4_242,
    likes: 88,
    thumbnail_url: "https://image.civitai.com/token/id/original=true/thumb.jpeg",
    page_url: "https://civitai.com/models/9001?modelVersionId=8001",
    ...part,
  };
}

beforeEach(() => {
  resolveEntrySizeMock.mockReset();
  resolveEntrySizeMock.mockResolvedValue(6_400_000_000);
});

describe("CatalogTableRow", () => {
  it("badges a row this machine cannot run, without disabling its Pull (#1276)", async () => {
    const reason = "This mold build was compiled without the MiniMax H3 engine.";
    const wrapper = mount(CatalogTableRow, {
      props: {
        entry: entry(),
        pulling: false,
        runtimeNotice: { message: reason, fromServer: true },
      },
    });
    await flushPromises();

    const badge = wrapper.get("[data-test='runtime-unavailable-badge']");
    expect(badge.text()).toContain("Download only");
    expect(badge.attributes("title")).toBe(reason);
    expect(wrapper.get("[data-test='pull']").attributes("disabled")).toBeUndefined();
  });

  it("badges nothing without a notice", async () => {
    const wrapper = mount(CatalogTableRow, { props: { entry: entry(), pulling: false } });
    await flushPromises();
    expect(wrapper.find("[data-test='runtime-unavailable-badge']").exists()).toBe(false);
  });

  it("renders the shared row shape with no thumbnail — source glyph, name, family, counts", async () => {
    const wrapper = mount(CatalogTableRow, { props: { entry: entry(), pulling: false } });
    await flushPromises();

    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.find('[data-test="family-placeholder"]').exists()).toBe(false);
    expect(wrapper.find("svg[data-source='civitai']").exists()).toBe(true);
    expect(wrapper.get("[data-test='row-title']").text()).toBe("Test Flux LoRA");
    // The chip shows the family's name, not its wire slug (#806).
    expect(wrapper.get("[data-test='row-family']").text()).toBe("FLUX");
    expect(wrapper.text()).toContain("alice");
    expect(wrapper.text()).toContain("↓ 4.2k · ♥ 88");
  });

  it("shows a friendly kind and explicit mature-content badge in its accessible name", async () => {
    const wrapper = mount(CatalogTableRow, {
      props: { entry: entry({ nsfw: true }), pulling: false },
    });
    await flushPromises();

    expect(wrapper.get('[data-test="model-kind-badge"]').text()).toBe("LoRA");
    expect(wrapper.get('[data-test="model-nsfw-badge"]').text()).toBe("18+ NSFW");
    const row = wrapper.get("[data-test='catalog-table-row']");
    expect(row.attributes("role")).toBeUndefined();
    const title = wrapper.get("[data-test='row-title']");
    expect(title.attributes("aria-label")).toContain("LoRA");
    expect(title.attributes("aria-label")).toContain("18+ NSFW");
  });

  it("shows SIZE and FETCH as the two size lines once resolved", async () => {
    const wrapper = mount(CatalogTableRow, {
      props: {
        entry: entry({
          size_bytes: 9_200_000_000,
          companion_details: [{ name: "t5", size_bytes: 11_000_000_000 }],
        }),
        pulling: false,
      },
    });
    await flushPromises();
    const sizes = wrapper.get("[data-test='row-sizes']");
    expect(sizes.text()).toContain("SIZE 9.2 GB");
    expect(sizes.text()).toContain("FETCH 20.2 GB");
  });

  it("shows a size skeleton while resolving, then the resolved SIZE line", async () => {
    let release: (value: number | null) => void = () => {};
    resolveEntrySizeMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    const wrapper = mount(CatalogTableRow, { props: { entry: entry(), pulling: false } });
    expect(wrapper.get("[data-test='row-sizes']").text()).toContain("SIZE …");

    release(6_400_000_000);
    await flushPromises();
    expect(wrapper.get("[data-test='row-sizes']").text()).toContain("SIZE 6.4 GB");
  });

  it("draws Get it filled, the treatment the catalog card already uses", () => {
    const wrapper = mount(CatalogTableRow, { props: { entry: entry(), pulling: false } });
    const get = wrapper.get("[data-test='pull']");
    expect(get.classes()).toContain("ms-toolbar-button--on");
    expect(get.classes()).not.toContain("ms-toolbar-button--accent");
  });

  it("emits open from the row and pull from the button without opening", async () => {
    const wrapper = mount(CatalogTableRow, { props: { entry: entry(), pulling: false } });
    await flushPromises();

    await wrapper.get("[data-test='catalog-table-row']").trigger("click");
    expect(wrapper.emitted("open")).toHaveLength(1);

    await wrapper.get("[data-test='pull']").trigger("click");
    const pulls = wrapper.emitted("pull");
    expect(pulls).toHaveLength(1);
    expect((pulls![0]![0] as CatalogEntry).id).toBe("cv:8001");
    expect(wrapper.emitted("open")).toHaveLength(1);
  });

  it("marks installed entries instead of offering Pull", async () => {
    const wrapper = mount(CatalogTableRow, {
      props: { entry: entry({ installed: true }), pulling: false },
    });
    await flushPromises();
    expect(wrapper.find("[data-test='pull']").exists()).toBe(false);
    expect(wrapper.text()).toContain("● ready");
  });

  it("still offers Pull for a model another machine is missing", async () => {
    const wrapper = mount(CatalogTableRow, {
      props: { entry: entry({ installed: true }), pulling: false, installable: true },
    });
    await flushPromises();
    // The host chips say where it already is; the button sends it somewhere new.
    expect(wrapper.text()).toContain("● ready");
    await wrapper.get("[data-test='pull']").trigger("click");
    expect(wrapper.emitted("pull")).toHaveLength(1);
  });

  it("drops Pull once every machine has the model", async () => {
    const wrapper = mount(CatalogTableRow, {
      props: { entry: entry({ installed: true }), pulling: false, installable: false },
    });
    await flushPromises();
    expect(wrapper.find("[data-test='pull']").exists()).toBe(false);
  });
});

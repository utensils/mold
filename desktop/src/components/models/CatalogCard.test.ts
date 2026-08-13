import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import CatalogCard from "./CatalogCard.vue";
import { resolveEntrySize } from "../../lib/catalogSizes";
import { openExternal } from "../../lib/openExternal";
import type { CatalogEntry } from "../../lib/api/types";

vi.mock("../../lib/catalogSizes", () => ({
  resolveEntrySize: vi.fn(),
}));
vi.mock("../../lib/openExternal", () => ({
  openExternal: vi.fn(),
}));

const resolveEntrySizeMock = vi.mocked(resolveEntrySize);
const openExternalMock = vi.mocked(openExternal);

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
  openExternalMock.mockReset();
  resolveEntrySizeMock.mockResolvedValue(6_400_000_000);
});

describe("CatalogCard", () => {
  it("renders name, author, family, counts, and the thumbnail", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();

    expect(wrapper.text()).toContain("Test Flux LoRA");
    expect(wrapper.text()).toContain("alice");
    expect(wrapper.text()).toContain("flux");
    expect(wrapper.text()).toContain("↓ 4.2k · ♥ 88");
    expect(wrapper.get("img").attributes("src")).toBe(
      "https://image.civitai.com/token/id/width=512/thumb.jpeg",
    );
    expect(wrapper.get("img").attributes("loading")).toBe("lazy");
    expect(wrapper.get("img").attributes("decoding")).toBe("async");
    expect(wrapper.get("img").attributes("fetchpriority")).toBe("low");
    expect(wrapper.get('[data-test="catalog-card"]').classes()).toContain("catalog-card-contained");
  });

  it("labels the model kind and mature content, includes both in the accessible name", async () => {
    const wrapper = mount(CatalogCard, {
      props: {
        entry: entry({
          nsfw: true,
          description: "  A portrait-focused style adapter.  ",
        }),
        pulling: false,
      },
    });
    await flushPromises();

    expect(wrapper.get('[data-test="model-kind-badge"]').text()).toBe("LoRA");
    expect(wrapper.get('[data-test="model-nsfw-badge"]').text()).toBe("18+ NSFW");
    expect(wrapper.get('[data-test="catalog-description"]').text()).toBe(
      "A portrait-focused style adapter.",
    );
    expect(wrapper.get('[data-test="catalog-description"]').classes()).toContain("line-clamp-2");
    expect(wrapper.get('[data-test="catalog-card"]').attributes("aria-label")).toContain("LoRA");
    expect(wrapper.get('[data-test="catalog-card"]').attributes("aria-label")).toContain(
      "18+ NSFW",
    );
  });

  it("omits blank descriptions and the mature-content badge for safe entries", async () => {
    const wrapper = mount(CatalogCard, {
      props: { entry: entry({ description: "   ", nsfw: false }), pulling: false },
    });
    await flushPromises();

    expect(wrapper.find('[data-test="catalog-description"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="model-nsfw-badge"]').exists()).toBe(false);
  });

  it("replaces a failed thumbnail with the family placeholder", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();
    await wrapper.get("img").trigger("error");
    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.find(".grain-shimmer").exists()).toBe(false);
    expect(wrapper.get('[data-test="family-placeholder"]').text()).toContain("FLUX");
  });

  it("uses a family-aware placeholder when no image exists", async () => {
    const grid = mount(CatalogCard, {
      props: { entry: entry({ family: "sdxl", thumbnail_url: null }), pulling: false },
    });
    await flushPromises();
    const gridPlaceholder = grid.get('[data-test="family-placeholder"]');
    expect(gridPlaceholder.attributes("data-layout")).toBe("grid");
    expect(gridPlaceholder.text()).toContain("SDXL");
  });

  it("shows a size skeleton while resolving, then the resolved SIZE line", async () => {
    let release: (value: number | null) => void = () => {};
    resolveEntrySizeMock.mockImplementation(
      () =>
        new Promise<number | null>((resolve) => {
          release = resolve;
        }),
    );
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    expect(wrapper.text()).toContain("SIZE …");

    release(6_400_000_000);
    await flushPromises();
    expect(wrapper.text()).toContain("SIZE 6.4 GB");
  });

  it("omits the size line entirely when the size cannot be resolved", async () => {
    resolveEntrySizeMock.mockResolvedValue(null);
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();
    expect(wrapper.text()).not.toContain("SIZE");
  });

  it("opens the model page from the link-out control", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();

    await wrapper.get('[data-test="page-link"]').trigger("click");
    expect(openExternalMock).toHaveBeenCalledWith(
      "https://civitai.com/models/9001?modelVersionId=8001",
    );
    // The external link is a secondary action — it must not open the drawer.
    expect(wrapper.emitted("open")).toBeUndefined();
  });

  it("emits open (the detail drawer) from the card body and title, not the browser", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();

    await wrapper.get('[data-test="catalog-card"]').trigger("click");
    await wrapper.get('[data-test="card-title"]').trigger("click");
    expect(wrapper.emitted("open")).toHaveLength(2);
    expect(wrapper.emitted("open")?.[0]?.[0]).toMatchObject({ id: "cv:8001" });
    expect(openExternalMock).not.toHaveBeenCalled();
  });

  it("exposes the card as a keyboard-focusable button", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();
    const card = wrapper.get('[data-test="catalog-card"]');
    expect(card.attributes("role")).toBe("button");
    expect(card.attributes("tabindex")).toBe("0");
  });

  it("marks the card that backs the open model detail", async () => {
    const wrapper = mount(CatalogCard, {
      props: { entry: entry(), pulling: false, selected: true },
    });
    await flushPromises();
    const card = wrapper.get('[data-test="catalog-card"]');
    expect(card.attributes("data-selected")).toBe("true");
    expect(card.attributes("aria-current")).toBe("true");
    expect(card.classes()).toContain("catalog-card-contained--selected");
  });

  it("opens the drawer from Enter and Space on the focused card", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();
    const card = wrapper.get('[data-test="catalog-card"]');

    await card.trigger("keydown.enter");
    await card.trigger("keydown.space");
    expect(wrapper.emitted("open")).toHaveLength(2);
    expect(wrapper.emitted("open")?.[0]?.[0]).toMatchObject({ id: "cv:8001" });
  });

  it("does not open the drawer from the Pull button", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();

    await wrapper.get("[data-test='pull']").trigger("click");
    expect(wrapper.emitted("pull")).toHaveLength(1);
    expect(wrapper.emitted("open")).toBeUndefined();
  });

  it("does not double-trigger open when a key is pressed on an inner control", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();

    // Enter bubbles from the Pull button to the card; the card must ignore it.
    await wrapper.get("[data-test='pull']").trigger("keydown.enter");
    expect(wrapper.emitted("open")).toBeUndefined();
  });

  it("hides the link-out when no page url can be derived", async () => {
    const wrapper = mount(CatalogCard, {
      props: { entry: entry({ page_url: null, id: "cv:8001" }), pulling: false },
    });
    await flushPromises();
    expect(wrapper.find('[data-test="page-link"]').exists()).toBe(false);
  });

  it("emits pull and shows installed state instead of the button", async () => {
    const wrapper = mount(CatalogCard, { props: { entry: entry(), pulling: false } });
    await flushPromises();
    await wrapper.get("button.pull-button, [data-test='pull']").trigger("click");
    expect(wrapper.emitted("pull")?.[0]?.[0]).toMatchObject({ id: "cv:8001" });

    const installed = mount(CatalogCard, {
      props: { entry: entry({ installed: true }), pulling: false },
    });
    await flushPromises();
    expect(installed.text()).toContain("installed");
    expect(installed.find("[data-test='pull']").exists()).toBe(false);
  });

  it("keeps the Pull button on an installed card another machine is missing", async () => {
    const wrapper = mount(CatalogCard, {
      props: { entry: entry({ installed: true }), pulling: false, installable: true },
    });
    await flushPromises();
    expect(wrapper.text()).toContain("installed");
    await wrapper.get("[data-test='pull']").trigger("click");
    expect(wrapper.emitted("pull")?.[0]?.[0]).toMatchObject({ id: "cv:8001" });
  });
});

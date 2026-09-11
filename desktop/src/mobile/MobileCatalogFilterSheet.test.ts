import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobileCatalogFilterSheet from "./MobileCatalogFilterSheet.vue";
import type { MobileHost } from "./hosts";

function host(id: string, name: string, online = true): MobileHost {
  return {
    id,
    name,
    baseUrl: `http://${id}:7680`,
    apiKey: "",
    online,
    connected: true,
  } as MobileHost;
}

function sheet(props: Record<string, unknown> = {}) {
  return mount(MobileCatalogFilterSheet, {
    props: {
      open: true,
      hosts: [host("studio", "Studio"), host("plato", "plato")],
      selectedHostId: "studio",
      discover: true,
      familyOptions: ["flux", "ltx2"],
      source: "all",
      kind: "",
      family: "",
      sort: "downloads",
      includeNsfw: false,
      ...props,
    },
  });
}

describe("MobileCatalogFilterSheet", () => {
  it("gathers every narrowing control the Styles list used to stack in the scroll", () => {
    const wrapper = sheet();

    // Seven separate surfaces in three visual languages became one sheet.
    expect(wrapper.find(".mobile-catalog-host-picker").exists()).toBe(true);
    expect(wrapper.find(".mobile-catalog-sources").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-catalog-kind-chips']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-catalog-family']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-catalog-sort']").exists()).toBe(true);
    expect(wrapper.find(".mobile-catalog-nsfw input").exists()).toBe(true);
  });

  it("keeps the Discover-only controls out of the Ready-to-use shelf", () => {
    const wrapper = sheet({ discover: false });

    // An installed model has no catalog source, kind, family or sort to pick.
    expect(wrapper.find(".mobile-catalog-sources").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-catalog-kind-chips']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-catalog-family']").exists()).toBe(false);
    // Which machine you are browsing is still a fair question.
    expect(wrapper.find(".mobile-catalog-host-picker").exists()).toBe(true);
  });

  it("hides the machine picker when there is only one machine to browse", () => {
    const wrapper = sheet({ hosts: [host("studio", "Studio")] });
    expect(wrapper.find(".mobile-catalog-host-picker").exists()).toBe(false);
  });

  it("reports each choice up rather than holding filter state of its own", async () => {
    const wrapper = sheet();

    await wrapper.findAll(".mobile-catalog-sources button")[1]!.trigger("click");
    expect(wrapper.emitted("update:source")).toEqual([["hf"]]);

    const lora = wrapper
      .get("[data-test='mobile-catalog-kind-chips']")
      .findAll("button")
      .find((button) => button.text() === "LoRAs")!;
    await lora.trigger("click");
    expect(wrapper.emitted("update:kind")).toEqual([["lora"]]);

    await wrapper.get("[data-test='mobile-catalog-sort']").setValue("rating");
    expect(wrapper.emitted("update:sort")).toEqual([["rating"]]);

    await wrapper.get(".mobile-catalog-nsfw input").setValue(true);
    expect(wrapper.emitted("update:includeNsfw")).toEqual([[true]]);

    await wrapper.get(".mobile-catalog-host-picker select").setValue("plato");
    expect(wrapper.emitted("select-host")).toEqual([["plato"]]);
  });

  it("takes the shared sheet chrome and closes on Done or the scrim", async () => {
    const wrapper = sheet({ open: false });
    expect(wrapper.get("[data-test='mobile-catalog-filters']").classes()).not.toContain("is-open");

    await wrapper.setProps({ open: true });
    expect(wrapper.get("[data-test='mobile-catalog-filters']").classes()).toContain("is-open");
    expect(wrapper.find(".mobile-sheet-grabber").exists()).toBe(true);
    expect(wrapper.get(".mobile-sheet-title").text()).toBe("Filters");

    await wrapper.get(".mobile-sheet-scrim").trigger("click");
    await wrapper.get("[data-test='mobile-catalog-filters-done']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(2);
  });

  it("clears every narrowing at once, because six controls are six things to undo", async () => {
    const wrapper = sheet({ source: "hf", kind: "lora", family: "flux", includeNsfw: true });

    await wrapper.get("[data-test='mobile-catalog-filters-reset']").trigger("click");
    expect(wrapper.emitted("reset")).toHaveLength(1);
  });
});

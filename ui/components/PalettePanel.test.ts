import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import PalettePanel from "./PalettePanel.vue";

const ITEMS = [
  { id: "new-print", section: "Create", label: "New print" },
  { id: "open-library", section: "Go", label: "Open library" },
  { id: "toggle-side", section: "Action", label: "Toggle sidebar" },
] as const;

function make(extra: Record<string, unknown> = {}) {
  return mount(PalettePanel, {
    props: { open: true, query: "", items: ITEMS, ...extra },
  });
}

describe("PalettePanel", () => {
  it("renders nothing when closed", () => {
    const wrapper = make({ open: false });
    expect(wrapper.find(".ms-palette").exists()).toBe(false);
  });

  it("renders a dialog with listbox options when open", () => {
    const wrapper = make();
    expect(wrapper.find("[role=dialog]").exists()).toBe(true);
    expect(wrapper.findAll("[role=option]")).toHaveLength(3);
    expect(wrapper.text()).toContain("Toggle sidebar");
  });

  it("highlights the first item by default", () => {
    const wrapper = make();
    const options = wrapper.findAll("[role=option]");
    expect(options[0]!.attributes("aria-selected")).toBe("true");
    expect(options[1]!.attributes("aria-selected")).toBe("false");
  });

  it("emits close when the scrim is clicked but not the panel", async () => {
    const wrapper = make();
    await wrapper.find(".ms-palette__panel").trigger("click");
    expect(wrapper.emitted("close")).toBeUndefined();
    await wrapper.find(".ms-palette").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits close on Escape", async () => {
    const wrapper = make();
    await wrapper.find("input").trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits update:query when typing", async () => {
    const wrapper = make();
    await wrapper.find("input").setValue("mod");
    expect(wrapper.emitted("update:query")).toEqual([["mod"]]);
  });

  it("keyboard flow: arrows move the highlight, Enter runs it", async () => {
    const wrapper = make();
    const input = wrapper.find("input");
    await input.trigger("keydown", { key: "ArrowDown" });
    await input.trigger("keydown", { key: "ArrowDown" });
    const options = wrapper.findAll("[role=option]");
    expect(options[2]!.attributes("aria-selected")).toBe("true");
    await input.trigger("keydown", { key: "ArrowUp" });
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("run")).toEqual([["open-library"]]);
  });

  it("wraps the highlight past the ends", async () => {
    const wrapper = make();
    await wrapper.find("input").trigger("keydown", { key: "ArrowUp" });
    expect(
      wrapper.findAll("[role=option]")[2]!.attributes("aria-selected"),
    ).toBe("true");
  });

  it("resets the highlight to first when the query changes", async () => {
    const wrapper = make();
    await wrapper.find("input").trigger("keydown", { key: "ArrowDown" });
    await wrapper.setProps({ query: "lib" });
    expect(
      wrapper.findAll("[role=option]")[0]!.attributes("aria-selected"),
    ).toBe("true");
  });

  it("Enter with no items emits nothing", async () => {
    const wrapper = make({ items: [] });
    await wrapper.find("input").trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("run")).toBeUndefined();
  });

  it("emits run with the item id on row click", async () => {
    const wrapper = make();
    await wrapper.findAll("[role=option]")[1]!.trigger("click");
    expect(wrapper.emitted("run")).toEqual([["open-library"]]);
  });

  it("shows the empty state when there are no items", () => {
    const wrapper = make({ items: [], emptyText: "Nothing here" });
    expect(wrapper.find(".ms-palette__empty").text()).toBe("Nothing here");
  });

  it("renders a row's trailing hint and omits it when absent", () => {
    const wrapper = make({
      items: [
        {
          id: "model-sdxl",
          section: "Model",
          label: "Use sdxl",
          hint: "on bender",
        },
        { id: "open-library", section: "Go", label: "Open library" },
      ],
    });
    const rows = wrapper.findAll("[role=option]");
    expect(rows[0]!.find(".ms-palette__hint").text()).toBe("on bender");
    expect(rows[1]!.find(".ms-palette__hint").exists()).toBe(false);
  });

  it("announces a slow source instead of flashing 'no matches'", () => {
    const wrapper = make({ items: [], busy: true, emptyText: "Nothing here" });
    expect(wrapper.find(".ms-palette__empty").exists()).toBe(false);
    const busy = wrapper.find(".ms-palette__busy");
    expect(busy.text()).toBe("Searching models…");
    expect(busy.attributes("role")).toBe("status");
    expect(busy.attributes("aria-live")).toBe("polite");
  });

  it("keeps the busy line below results that already landed", () => {
    const wrapper = make({ busy: true });
    expect(wrapper.findAll("[role=option]")).toHaveLength(3);
    expect(wrapper.find(".ms-palette__busy").exists()).toBe(true);
  });

  it("points aria-activedescendant at the highlighted option", async () => {
    const wrapper = make();
    const input = wrapper.find("input");
    expect(input.attributes("aria-activedescendant")).toBe(
      "ms-palette-opt-new-print",
    );
    await input.trigger("keydown", { key: "ArrowDown" });
    expect(input.attributes("aria-activedescendant")).toBe(
      "ms-palette-opt-open-library",
    );
  });
});

import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import ModelTableRow from "./ModelTableRow.vue";

const openExternal = vi.fn().mockResolvedValue(undefined);
vi.mock("../../lib/openExternal", () => ({
  openExternal: (...a: unknown[]) => openExternal(...a),
}));

function mountRow(props: Record<string, unknown> = {}, slots: Record<string, string> = {}) {
  return mount(ModelTableRow, {
    props: { name: "flux-dev:q8", source: "hf", ...props },
    slots,
  });
}

describe("ModelTableRow", () => {
  it("renders name, source glyph, host chips, quant chip, and family code", () => {
    const wrapper = mountRow({
      hostLabels: ["This Mac", "hal9000"],
      quant: "q8",
      family: "flux",
    });
    expect(wrapper.get("[data-test='row-title']").text()).toBe("flux-dev:q8");
    expect(wrapper.find("svg[data-source='hf']").exists()).toBe(true);
    expect(wrapper.findAll("[data-test='installed-host']").map((c) => c.text())).toEqual([
      "This Mac",
      "hal9000",
    ]);
    expect(wrapper.text()).toContain("q8");
    // The chip shows the family's name, not its wire slug (#806).
    expect(wrapper.get("[data-test='row-family']").text()).toBe("FLUX");
  });

  /**
   * #806 acceptance criterion 1: the family reads as **Wan Video** on every
   * surface. Desktop rendered the raw `wan` slug because the label table lived
   * in `web/`; it is now shared through `@studio/lib/modelFamily`, where a test
   * pins it to `tests/fixtures/wan/surface-parity-v1.json` alongside the CLI
   * and TUI readers.
   */
  it("names wan the way the other surfaces name it", () => {
    const wrapper = mountRow({ name: "wan22-t2v-a14b:q5", family: "wan" });
    expect(wrapper.get("[data-test='row-family']").text()).toBe("Wan Video");
  });

  it("shows a warm residency dot when loaded, a cold placeholder when not, none when omitted", () => {
    expect(mountRow({ loaded: true }).find("[role='img'][aria-label='On GPU']").exists()).toBe(
      true,
    );
    expect(mountRow({ loaded: false }).find("[role='img'][aria-label='Cold']").exists()).toBe(true);
    expect(mountRow().find("[role='img'][aria-label='Cold']").exists()).toBe(false);
    expect(mountRow().find("[role='img'][aria-label='On GPU']").exists()).toBe(false);
  });

  it("renders the two-line size block and the relative usage bar", () => {
    const wrapper = mountRow({
      sizePrimary: "11.8 GB weights",
      sizeSecondary: "23.1 GB with shared runtime",
      barPercent: 40,
    });
    const sizes = wrapper.get("[data-test='row-sizes']");
    expect(sizes.text()).toContain("11.8 GB weights");
    expect(sizes.text()).toContain("23.1 GB with shared runtime");
    const footprint = wrapper.get("[data-test='model-footprint-bar']");
    expect(footprint.html()).toContain("width: 40%");
    expect(footprint.attributes("aria-label")).toContain("23.1 GB with shared runtime");
    expect(footprint.attributes("aria-label")).toContain("largest model in this list");
    expect(footprint.attributes("aria-label")).toContain("not download progress");
    expect(footprint.attributes("role")).toBe("meter");
    expect(footprint.attributes("aria-valuenow")).toBe("40");
    const description = wrapper.get("[data-test='model-footprint-description']");
    expect(wrapper.get("[data-test='model-table-row']").attributes("aria-describedby")).toBe(
      description.attributes("id"),
    );
    expect(description.text()).toBe(footprint.attributes("aria-label"));
  });

  it("explains the footprint bar with the styled tooltip, not a native title", async () => {
    vi.useFakeTimers();
    try {
      const wrapper = mountRow({ sizeSecondary: "23.1 GB with shared runtime", barPercent: 40 });
      const footprint = wrapper.get("[data-test='model-footprint-bar']");
      expect(footprint.attributes("title")).toBeUndefined();

      await footprint.element.parentElement!.dispatchEvent(new Event("mouseenter"));
      vi.advanceTimersByTime(400);
      await wrapper.vm.$nextTick();
      const tip = document.body.querySelector('[role="tooltip"]');
      expect(tip?.textContent).toContain("not download progress");
      wrapper.unmount();
      document.body.innerHTML = "";
    } finally {
      vi.useRealTimers();
    }
  });

  it("opens the external model page without triggering the row's open action", async () => {
    const wrapper = mountRow({ pageUrl: "https://huggingface.co/x/y", clickable: true });
    await wrapper.get("[data-test='model-page-link']").trigger("click");
    expect(openExternal).toHaveBeenCalledWith("https://huggingface.co/x/y");
    expect(wrapper.emitted("open")).toBeUndefined();
  });

  it("emits open from row click and keyboard when clickable, never when static", async () => {
    const clickableRow = mountRow({ clickable: true });
    await clickableRow.get("[data-test='model-table-row']").trigger("click");
    await clickableRow.get("[data-test='model-table-row']").trigger("keydown.enter");
    expect(clickableRow.emitted("open")).toHaveLength(2);

    const staticRow = mountRow();
    await staticRow.get("[data-test='model-table-row']").trigger("click");
    expect(staticRow.emitted("open")).toBeUndefined();
    expect(staticRow.get("[data-test='model-table-row']").attributes("role")).toBeUndefined();
  });

  it("renders parent actions in the actions slot", () => {
    const wrapper = mountRow({}, { actions: "<button data-test='pull'>Pull</button>" });
    expect(wrapper.get("[data-test='pull']").text()).toBe("Pull");
  });

  it("marks the row that backs the open model detail", () => {
    const wrapper = mountRow({ selected: true });
    const row = wrapper.get("[data-test='model-table-row']");
    expect(row.attributes("data-selected")).toBe("true");
    expect(row.attributes("aria-current")).toBe("true");
    expect(row.classes()).toContain("model-table-row--selected");
  });
});

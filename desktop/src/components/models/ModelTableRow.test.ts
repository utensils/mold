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
  it("renders name, source glyph, host chips, and family code", () => {
    const wrapper = mountRow({
      hostLabels: ["This Mac", "hal9000"],
      family: "flux",
    });
    expect(wrapper.get("[data-test='row-title']").text()).toBe("flux-dev:q8");
    expect(wrapper.find("svg[data-source='hf']").exists()).toBe(true);
    expect(wrapper.findAll("[data-test='installed-host']").map((c) => c.text())).toEqual([
      "This Mac",
      "hal9000",
    ]);
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
  it("stacks the request id under a friendlier name, and hides it when they agree", () => {
    const stacked = mountRow({ name: "Photoreal — best quality", id: "flux-dev:q4" });
    expect(stacked.get("[data-test='row-title']").text()).toBe("Photoreal — best quality");
    expect(stacked.get("[data-test='row-id']").text()).toBe("flux-dev:q4");
    // A manifest model's display name IS its id — one line, not two copies.
    expect(
      mountRow({ name: "flux-dev:q4", id: "flux-dev:q4" }).find("[data-test='row-id']").exists(),
    ).toBe(false);
    // A Civitai install id is a bare number: the title alone.
    expect(
      mountRow({ name: "Juggernaut XL", id: "cv:1759168" }).find("[data-test='row-id']").exists(),
    ).toBe(false);
  });

  it("carries a one-line note in plain words", () => {
    const wrapper = mountRow({ note: "Full quality, 20+ passes" });
    expect(wrapper.get("[data-test='row-note']").text()).toBe("Full quality, 20+ passes");
    expect(mountRow().find("[data-test='row-note']").exists()).toBe(false);
  });

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

  it("gives the machine chips a column of their own only when asked", () => {
    const inline = mountRow({ hostLabels: ["This Mac"] });
    expect(inline.find(".model-table-row__machines").exists()).toBe(false);
    expect(inline.get("[data-test='installed-host']").text()).toBe("This Mac");

    const columned = mountRow({ hostLabels: ["This Mac"], machinesColumn: true });
    const cell = columned.get(".model-table-row__machines");
    expect(cell.findAll("[data-test='installed-host']").map((c) => c.text())).toEqual(["This Mac"]);
    expect(columned.get("[data-test='model-table-row']").classes()).toContain(
      "model-table-row--has-machines",
    );
  });

  /**
   * The mock's row is 52px, which is ONE line per cell. The machines cell used
   * to stack a line per host with no cap, so a style on three machines drew a
   * three-line row and dragged the whole table off the mock's rhythm. (Replaces
   * the old assertion that the column listed every label inline — the rest are
   * still reachable, in the overflow chip's tooltip.)
   */
  it("keeps the machines column to one line, with the rest behind a +N", () => {
    const many = mountRow({
      hostLabels: ["This Mac", "hal9000", "plato"],
      machinesColumn: true,
    });
    const cell = many.get(".model-table-row__machines");
    expect(cell.findAll("[data-test='installed-host']").map((c) => c.text())).toEqual(["This Mac"]);
    const more = cell.get("[data-test='installed-host-more']");
    expect(more.text()).toBe("+2");
    expect(more.attributes("title")).toBe("hal9000, plato");

    // One machine needs no overflow chip at all.
    expect(
      mountRow({ hostLabels: ["This Mac"], machinesColumn: true })
        .find("[data-test='installed-host-more']")
        .exists(),
    ).toBe(false);
  });

  /**
   * A pinned table (`--model-row-columns`) fixes the track count, so a row that
   * skips a cell shifts every column after it one track left. `noteColumn` is
   * the parent saying "this table has a Good for column" — the cell is then
   * always emitted, empty or not.
   */
  it("emits the note cell on every row when the table pins a note column", () => {
    const withoutNote = mountRow({ noteColumn: true });
    expect(withoutNote.get("[data-test='row-note']").text()).toBe("");
    expect(withoutNote.get("[data-test='model-table-row']").classes()).toContain(
      "model-table-row--has-note",
    );

    // Unpinned rows keep sizing their own tracks: no note, no cell, no track.
    const unpinned = mountRow();
    expect(unpinned.find("[data-test='row-note']").exists()).toBe(false);
    expect(unpinned.get("[data-test='model-table-row']").classes()).not.toContain(
      "model-table-row--has-note",
    );
  });

  it("counts one cell per pinned track so the axis never shifts", () => {
    const cells = (wrapper: ReturnType<typeof mountRow>) =>
      Array.from(wrapper.get("[data-test='model-table-row']").element.children).length;
    const withNote = mountRow(
      { noteColumn: true, note: "Full quality", machinesColumn: true, hostLabels: ["This Mac"] },
      { actions: "<button>Load</button>" },
    );
    const withoutNote = mountRow(
      { noteColumn: true, machinesColumn: true, hostLabels: ["This Mac"] },
      { actions: "<button>Load</button>" },
    );
    expect(cells(withoutNote)).toBe(cells(withNote));
    expect(cells(withNote)).toBe(5);
  });

  /**
   * The mock's identity cell is star + name. The source mark stays — the brand
   * marks are wanted — but small and on the mono line, never a 44px colour
   * glyph column ahead of the first character.
   */
  it("puts the source mark on the mono line, not ahead of the name", () => {
    const wrapper = mountRow({ name: "Photoreal", id: "flux-dev:q4" });
    const identity = wrapper.get(".model-table-row__identity");
    const glyph = identity.get("svg[data-source='hf']");
    expect(glyph.attributes("width")).toBe("10");
    // It rides the mono id line, so it can never widen the identity's chrome.
    expect(glyph.element.closest("[data-test='row-mono-line']")).not.toBeNull();
    // Never a leading track of its own in the identity flex.
    expect(Array.from(identity.element.children).some((el) => el.tagName === "svg")).toBe(false);
  });

  it("keeps the source mark on rows that have no mono line of their own", () => {
    // A manifest style's display name IS its id, so there is no id line and no
    // family chip — the mark rides the title instead of inventing a line.
    const wrapper = mountRow({ name: "flux-dev:q8", id: "flux-dev:q8" });
    expect(wrapper.find("[data-test='row-mono-line']").exists()).toBe(false);
    expect(wrapper.find("svg[data-source='hf']").exists()).toBe(true);
  });

  it("marks the row that backs the open model detail", () => {
    const wrapper = mountRow({ selected: true });
    const row = wrapper.get("[data-test='model-table-row']");
    expect(row.attributes("data-selected")).toBe("true");
    expect(row.attributes("aria-current")).toBe("true");
    expect(row.classes()).toContain("model-table-row--selected");
  });
});

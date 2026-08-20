import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import TrashTileActions from "./TrashTileActions.vue";

const NOW = 1_700_000_000_000;
const DAY = 86_400;

describe("TrashTileActions", () => {
  it("wears a Purges in N d chip with the warning tone on the number only", () => {
    const wrapper = mount(TrashTileActions, {
      props: { purgeAt: NOW / 1000 + 3 * DAY - 60, nowMs: NOW },
    });
    const chip = wrapper.get("[data-test='purge-chip']");
    expect(chip.text().replace(/\s+/g, " ")).toBe("Purges in 3 d");
    expect(chip.attributes("data-kind")).toBe("purges");
    expect(chip.get("b").text()).toBe("3 d");
    expect(chip.get("b").classes()).toContain("ms-purge__n");
  });

  it("reads Kept when retention is forever and Purges today at the edge", () => {
    expect(
      mount(TrashTileActions, { props: { purgeAt: null, nowMs: NOW } })
        .get("[data-test='purge-chip']")
        .text(),
    ).toBe("Kept");
    const today = mount(TrashTileActions, { props: { purgeAt: NOW / 1000 - 10, nowMs: NOW } });
    expect(today.get("[data-test='purge-chip']").attributes("data-kind")).toBe("today");
  });

  it("emits restore (default) and deleteForever without bubbling to the tile", async () => {
    let tileClicks = 0;
    const wrapper = mount(
      {
        components: { TrashTileActions },
        template:
          "<button class='group' @click='tileClicks++'><TrashTileActions :purge-at='null' @restore='$emit(\"restore\")' @delete-forever='$emit(\"deleteForever\")' /></button>",
        data: () => ({ tileClicks: 0 }),
        watch: { tileClicks: (n: number) => (tileClicks = n) },
      },
      {},
    );
    await wrapper.get("[data-test='trash-restore']").trigger("click");
    expect(wrapper.emitted("restore")).toHaveLength(1);
    await wrapper.get("[data-test='trash-delete-forever']").trigger("click");
    expect(wrapper.emitted("deleteForever")).toHaveLength(1);
    expect(tileClicks).toBe(0);
  });

  it("hides the hover actions in select mode", () => {
    const wrapper = mount(TrashTileActions, { props: { purgeAt: null, showActions: false } });
    expect(wrapper.find("[data-test='trash-actions']").exists()).toBe(false);
    expect(wrapper.find("[data-test='purge-chip']").exists()).toBe(true);
  });
});

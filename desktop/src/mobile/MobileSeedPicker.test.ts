import { mount } from "@vue/test-utils";
import { defineComponent, nextTick, reactive } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";
import MobileSeedPicker from "./MobileSeedPicker.vue";

function mountPicker(seed = "", lastSeed: number | null = null) {
  const state = reactive({ seed, lastSeed, valid: true });
  const Harness = defineComponent({
    components: { MobileSeedPicker },
    setup: () => ({ state }),
    template: `
      <MobileSeedPicker
        v-model="state.seed"
        :last-seed="state.lastSeed"
        @validity-change="state.valid = $event"
      />
    `,
  });
  return { wrapper: mount(Harness), state };
}

afterEach(() => vi.restoreAllMocks());

describe("MobileSeedPicker", () => {
  it("makes random generation explicit without showing a misleading empty input", () => {
    const { wrapper } = mountPicker();

    expect(wrapper.get("[data-test='mobile-seed-mode-random']").attributes("aria-pressed")).toBe(
      "true",
    );
    expect(wrapper.get("[data-test='mobile-seed-mode-fixed']").attributes("aria-pressed")).toBe(
      "false",
    );
    expect(wrapper.find("[data-test='mobile-seed-input']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-seed-note']").text()).toBe("New seed for every print.");
  });

  it("turns Fixed into a populated, numeric mobile input", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const { wrapper, state } = mountPicker();

    await wrapper.get("[data-test='mobile-seed-mode-fixed']").trigger("click");

    expect(state.seed).toBe("2147483647");
    const input = wrapper.get("[data-test='mobile-seed-input']");
    expect(input.attributes()).toMatchObject({
      inputmode: "numeric",
      enterkeyhint: "done",
      pattern: "[0-9]*",
      "aria-label": "Fixed seed value",
    });
    expect(wrapper.get("[data-test='mobile-seed-mode-fixed']").attributes("aria-pressed")).toBe(
      "true",
    );
    expect(wrapper.get("[data-test='mobile-seed-note']").text()).toBe(
      "Repeat this seed for matching results.",
    );
  });

  it("can lock the most recent generated seed or roll a new fixed value", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0.25);
    const { wrapper, state } = mountPicker("", 91);

    await wrapper.get("[data-test='mobile-use-last-seed']").trigger("click");
    expect(state.seed).toBe("91");

    await wrapper.get("[data-test='mobile-new-seed']").trigger("click");
    expect(state.seed).toBe("1073741823");
  });

  it("returns to Random by clearing the fixed value", async () => {
    const { wrapper, state } = mountPicker("1234");

    expect(wrapper.get("[data-test='mobile-seed-mode-fixed']").attributes("aria-pressed")).toBe(
      "true",
    );
    await wrapper.get("[data-test='mobile-seed-mode-random']").trigger("click");

    expect(state.seed).toBe("");
    expect(wrapper.find("[data-test='mobile-seed-input']").exists()).toBe(false);
  });

  it("keeps fixed mode stable while editing and explains invalid values", async () => {
    const { wrapper, state } = mountPicker("7");
    const input = wrapper.get("[data-test='mobile-seed-input']");

    await input.setValue("");
    expect(state.seed).toBe("");
    expect(wrapper.get("[data-test='mobile-seed-mode-fixed']").attributes("aria-pressed")).toBe(
      "true",
    );
    expect(wrapper.get("[data-test='mobile-seed-note']").text()).toBe(
      "Enter a whole number, or choose Random.",
    );

    state.seed = "not-a-number";
    await nextTick();
    expect(wrapper.get("[data-test='mobile-seed-note']").text()).toBe(
      "Use a whole number, 0 or higher.",
    );
  });

  it.each(["-1", "1.5", "not-a-number", "9007199254740992"])(
    "marks %s invalid instead of silently treating it as random",
    async (value) => {
      const { wrapper, state } = mountPicker("7");
      const input = wrapper.get("[data-test='mobile-seed-input']");

      await input.setValue(value);

      expect(state.valid).toBe(false);
      expect(input.attributes("aria-invalid")).toBe("true");
      expect(input.attributes("aria-describedby")).toBe(
        wrapper.get("[data-test='mobile-seed-note']").attributes("id"),
      );
      expect(wrapper.get("[data-test='mobile-seed-note']").attributes("role")).toBe("alert");
    },
  );

  it("accepts zero as a real fixed seed", async () => {
    const { wrapper, state } = mountPicker("7");

    await wrapper.get("[data-test='mobile-seed-input']").setValue("0");

    expect(state.seed).toBe("0");
    expect(state.valid).toBe(true);
    expect(wrapper.get("[data-test='mobile-seed-input']").attributes("aria-invalid")).toBe(
      undefined,
    );
  });
});

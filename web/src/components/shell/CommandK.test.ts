import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import CommandK from "./CommandK.vue";
import PalettePanel from "@ui/components/PalettePanel.vue";
import { theme, themeFamily } from "../../lib/theme";
import type { PaletteItem } from "@ui/components/PalettePanel.vue";

const pushMock = vi.hoisted(() => vi.fn());
const formState = vi.hoisted(() => ({ value: { model: "" } }));
const fetchModelsMock = vi.hoisted(() =>
  vi.fn(async () => [
    { name: "flux-dev:q4", downloaded: true },
    { name: "sdxl:fp16", downloaded: false },
  ]),
);

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: pushMock }),
}));

vi.mock("../../api", () => ({
  fetchModels: fetchModelsMock,
}));

vi.mock("../../composables/useGenerateForm", () => ({
  useGenerateForm: () => ({ state: formState }),
}));

function items(wrapper: ReturnType<typeof mount>): PaletteItem[] {
  return wrapper.findComponent(PalettePanel).props("items") as PaletteItem[];
}

async function openPalette() {
  const wrapper = mount(CommandK, { props: { open: false } });
  await wrapper.setProps({ open: true });
  await flushPromises();
  return wrapper;
}

describe("CommandK", () => {
  beforeEach(() => {
    pushMock.mockClear();
    formState.value.model = "";
  });
  afterEach(() => {
    theme.value = "system";
    themeFamily.value = "mold";
  });

  it("offers navigation, action, and theme commands", async () => {
    const wrapper = await openPalette();
    const ids = items(wrapper).map((i) => i.id);
    expect(ids).toContain("go-gallery");
    expect(ids).toContain("go-machines");
    expect(ids).toContain("go-settings");
    expect(ids).toContain("action-new-print");
    expect(ids).toContain("theme-family-safelight");
  });

  it("lists only installed models as run commands", async () => {
    const wrapper = await openPalette();
    const ids = items(wrapper).map((i) => i.id);
    expect(ids).toContain("model-flux-dev:q4");
    expect(ids).not.toContain("model-sdxl:fp16");
  });

  it("filters the items by a case-insensitive query", async () => {
    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("update:query", "safelight");
    await nextTick();

    const matched = items(wrapper);
    expect(matched.map((i) => i.id)).toEqual(["theme-family-safelight"]);
  });

  it("runs a navigation command and closes", async () => {
    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "go-models");
    await nextTick();

    expect(pushMock).toHaveBeenCalledWith("/models");
    expect(wrapper.emitted("close")).toBeTruthy();
  });

  it("mutates the shared theme refs from theme commands", async () => {
    const wrapper = await openPalette();
    const palette = wrapper.findComponent(PalettePanel);

    palette.vm.$emit("run", "theme-appearance-dark");
    await nextTick();
    expect(theme.value).toBe("dark");

    palette.vm.$emit("run", "theme-family-safelight");
    await nextTick();
    expect(themeFamily.value).toBe("safelight");
  });

  it("selects a model and opens create when a model command runs", async () => {
    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "model-flux-dev:q4");
    await nextTick();

    expect(formState.value.model).toBe("flux-dev:q4");
    expect(pushMock).toHaveBeenCalledWith("/create");
  });
});

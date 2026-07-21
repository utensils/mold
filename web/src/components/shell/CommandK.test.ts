import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import CommandK from "./CommandK.vue";
import PalettePanel from "@ui/components/PalettePanel.vue";
import { theme, themeFamily } from "../../lib/theme";
import {
  useGenerateForm,
  __testing__ as generateFormTesting,
} from "../../composables/useGenerateForm";
import type { PaletteItem } from "@ui/components/types";

const pushMock = vi.hoisted(() => vi.fn());
// Full-enough model rows so applyModelDefaults can resolve family + defaults.
const fetchModelsMock = vi.hoisted(() =>
  vi.fn(async () => [
    {
      name: "flux-dev:q4",
      downloaded: true,
      family: "flux",
      default_width: 1024,
      default_height: 1024,
      default_steps: 20,
      default_guidance: 3.5,
    },
    {
      name: "sdxl:fp16",
      downloaded: false,
      family: "sdxl",
      default_width: 1024,
      default_height: 1024,
      default_steps: 30,
      default_guidance: 7,
    },
  ]),
);

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: pushMock }),
}));

vi.mock("../../api", () => ({
  fetchModels: fetchModelsMock,
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
    localStorage.clear();
    generateFormTesting.resetForTest();
  });
  afterEach(() => {
    theme.value = "system";
    themeFamily.value = "mold";
  });

  it("offers navigation, action, and theme commands", async () => {
    const wrapper = await openPalette();
    const ids = items(wrapper).map((i) => i.id);
    expect(ids).toContain("go-library");
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
    const form = useGenerateForm();
    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "model-flux-dev:q4");
    await nextTick();

    expect(form.state.value.model).toBe("flux-dev:q4");
    expect(pushMock).toHaveBeenCalledWith("/create");
  });

  it("applies the model's defaults and clears stale family state on a model command", async () => {
    const form = useGenerateForm();
    // Start in a Qwen-edit configuration carrying two edit images — the shape
    // that must not survive a switch to a FLUX model.
    form.state.value.model = "qwen-image-edit:q4";
    form.state.value.modelFamily = "qwen-image-edit";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "a.png", base64: "A" },
      { kind: "upload", filename: "b.png", base64: "B" },
    ];
    await nextTick();

    const wrapper = await openPalette();
    wrapper.findComponent(PalettePanel).vm.$emit("run", "model-flux-dev:q4");
    await nextTick();

    expect(form.state.value.model).toBe("flux-dev:q4");
    expect(form.state.value.modelFamily).toBe("flux");
    // FLUX takes a single source image and never emits edit_images.
    expect(form.state.value.imageAttachments.length).toBeLessThanOrEqual(1);
    expect(form.toRequest().edit_images).toBeUndefined();
  });
});

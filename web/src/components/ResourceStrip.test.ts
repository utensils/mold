import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { ref } from "vue";
import ResourceStrip from "./ResourceStrip.vue";
import type { ResourceSnapshot } from "../types";
import { RESOURCES_INJECTION_KEY } from "../composables/useResources";

function mountWith(
  snap: ResourceSnapshot | null,
  variant: "full" | "chip" = "full",
) {
  const snapshot = ref(snap);
  const gpuList = ref(snap?.gpus ?? []);
  return mount(ResourceStrip, {
    props: { variant },
    global: {
      provide: {
        [RESOURCES_INJECTION_KEY]: {
          snapshot,
          gpuList,
          error: ref(null),
          stop: () => {},
        },
      },
    },
  });
}

const cuda: ResourceSnapshot = {
  hostname: "hal",
  timestamp: 1,
  gpus: [
    {
      ordinal: 0,
      name: "RTX 3090",
      backend: "cuda",
      vram_total: 24_000_000_000,
      vram_used: 14_200_000_000,
      vram_used_by_mold: 10_100_000_000,
      vram_used_by_other: 4_100_000_000,
    },
  ],
  system_ram: {
    total: 64_000_000_000,
    used: 38_400_000_000,
    used_by_mold: 22_100_000_000,
    used_by_other: 16_300_000_000,
  },
};

const metal: ResourceSnapshot = {
  hostname: "mbp",
  timestamp: 1,
  gpus: [
    {
      ordinal: 0,
      name: "Apple Metal GPU",
      backend: "metal",
      vram_total: 64_000_000_000,
      vram_used: 38_000_000_000,
      vram_used_by_mold: null,
      vram_used_by_other: null,
    },
  ],
  system_ram: {
    total: 64_000_000_000,
    used: 38_000_000_000,
    used_by_mold: 12_000_000_000,
    used_by_other: 26_000_000_000,
  },
};

describe("ResourceStrip", () => {
  it("renders one GPU row and one RAM row on CUDA", () => {
    const wrapper = mountWith(cuda);
    const rows = wrapper.findAll('[data-test="resource-row"]');
    expect(rows.length).toBe(2);
    // GPU row includes per-process breakdown.
    expect(rows[0].text()).toContain("RTX 3090");
    expect(rows[0].text()).toContain("mold");
    expect(rows[0].text()).toContain("other");
  });

  it("hides per-process breakdown on Metal (null attribution)", () => {
    const wrapper = mountWith(metal);
    const rows = wrapper.findAll('[data-test="resource-row"]');
    expect(rows.length).toBe(2);
    // The GPU row must NOT mention mold/other — they're null.
    const gpuRow = rows[0].text();
    expect(gpuRow).not.toMatch(/mold/i);
  });

  it("hides GPU rows on CPU-only host", () => {
    const cpuOnly: ResourceSnapshot = {
      ...cuda,
      gpus: [],
    };
    const wrapper = mountWith(cpuOnly);
    const rows = wrapper.findAll('[data-test="resource-row"]');
    // Only the RAM row should render.
    expect(rows.length).toBe(1);
  });

  it("renders placeholder when snapshot is null", () => {
    const wrapper = mountWith(null);
    expect(wrapper.text()).toContain("…");
  });

  it("chip variant renders a compact single-line summary", () => {
    const wrapper = mountWith(cuda, "chip");
    expect(wrapper.find('[data-test="resource-chip"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="resource-row"]').exists()).toBe(false);
  });
});

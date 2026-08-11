import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import type { MiniMaxH3Capability } from "../lib/minimaxH3Inventory";
import MinimaxH3InventoryPanel from "./MinimaxH3InventoryPanel.vue";

function capability(): MiniMaxH3Capability {
  const shared = [
    ["qwen", "Qwen encoder", "text-encoder", "qwen", 10],
    ["processor", "Qwen processor", "tokenizer", "processor", 2],
    ["video-vae", "Visual VAE", "vae", "video-vae", 5],
    ["audio-vae", "Audio VAE", "vae", "audio-vae", 3],
  ] as const;
  return {
    runtime_available: true,
    qualification: {
      backend: "cuda",
      metal_supported: false,
      minimum_host_ram_bytes: 128_000_000_000,
      minimum_vram_bytes: 24_000_000_000,
      attention_profile: "SDPA",
      quantization_profile: "BF16 + INT8",
    },
    components: [
      ...shared.map(([id, display_name, kind, role, gb]) => ({
        id,
        display_name,
        kind,
        role,
        scope: "shared" as const,
        size_bytes: gb * 1_000_000_000,
        state:
          id === "audio-vae"
            ? ("downloading" as const)
            : ("installed" as const),
        ...(id === "audio-vae"
          ? { download: { job_id: "job-audio", bytes_done: 1, bytes_total: 3 } }
          : {}),
      })),
      {
        id: "fl-transformer",
        display_name: "FL2VA transformer",
        kind: "checkpoint",
        role: "transformer",
        scope: "fl2va",
        size_bytes: 20_000_000_000,
        state: "installed",
      },
      {
        id: "ref-transformer",
        display_name: "Ref2VA transformer",
        kind: "checkpoint",
        role: "transformer",
        scope: "ref2va",
        size_bytes: 30_000_000_000,
        state: "missing",
      },
    ],
    partitions: [
      {
        task: "fl2va",
        model: "minimax-h3-fl2va",
        display_name: "MiniMax H3 FL2VA",
        runtime_available: true,
        tier: "Standard",
        component_ids: [
          "fl-transformer",
          "qwen",
          "processor",
          "video-vae",
          "audio-vae",
        ],
      },
      {
        task: "ref2va",
        model: "minimax-h3-ref2va",
        display_name: "MiniMax H3 Ref2VA",
        runtime_available: true,
        tier: "Large",
        component_ids: [
          "ref-transformer",
          "qwen",
          "processor",
          "video-vae",
          "audio-vae",
        ],
      },
    ],
  };
}

describe("MinimaxH3InventoryPanel", () => {
  it("renders capability facts and static exact-host plans without an action", () => {
    const wrapper = mount(MinimaxH3InventoryPanel, {
      props: {
        hosts: [
          {
            id: "render-a",
            label: "Render A",
            capabilities: { minimax_h3: capability() },
          },
        ],
      },
    });

    expect(wrapper.text()).toContain("MiniMax H3 FL2VA");
    expect(wrapper.text()).toContain("MiniMax H3 Ref2VA");
    expect(wrapper.text()).toContain("Advertised tasks · 70.0 GB on disk");
    expect(wrapper.text()).toContain("128.0 GB minimum");
    expect(wrapper.text()).toContain("CUDA only");
    expect(wrapper.text()).toContain("Metal");
    expect(wrapper.text()).toContain("Unsupported");
    expect(wrapper.text()).toContain("Downloading on Render A");
    expect(wrapper.findAll("button")).toHaveLength(0);
    expect(
      wrapper.findAll("[data-test^='h3-component-render-a-']"),
    ).toHaveLength(6);
  });

  it("labels a host that advertises only qualified FL2VA", () => {
    const available = capability();
    available.partitions = available.partitions.filter(
      (partition) => partition.task === "fl2va",
    );
    available.components = available.components.filter(
      (component) => component.scope !== "ref2va",
    );
    const wrapper = mount(MinimaxH3InventoryPanel, {
      props: {
        hosts: [
          {
            id: "render-a",
            label: "Render A",
            capabilities: { minimax_h3: available },
          },
        ],
      },
    });

    expect(wrapper.text()).toContain("FL2VA · 40.0 GB on disk");
    expect(wrapper.text()).not.toContain("Ref2VA");
  });

  it("renders nothing when model access denies H3 or runtime availability is false", async () => {
    const available = capability();
    const wrapper = mount(MinimaxH3InventoryPanel, {
      props: {
        hosts: [
          {
            id: "render-a",
            label: "Render A",
            capabilities: {
              minimax_h3: available,
              model_access: {
                restrictions: [
                  {
                    code: "license-not-accepted",
                    family: "minimax-h3",
                    message: "Unavailable",
                    license_url: "https://example.invalid/license",
                    authorization_url: "https://example.invalid/auth",
                  },
                ],
              },
            },
          },
        ],
      },
    });
    expect(wrapper.find("[data-test='h3-inventory']").exists()).toBe(false);

    available.runtime_available = false;
    await wrapper.setProps({
      hosts: [
        {
          id: "render-a",
          label: "Render A",
          capabilities: { minimax_h3: available },
        },
      ],
    });
    expect(wrapper.find("[data-test='h3-inventory']").exists()).toBe(false);
  });
});

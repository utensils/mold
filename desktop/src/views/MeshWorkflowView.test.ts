import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@studio/components/MeshWorkflowStudio.vue", () => ({
  default: {
    props: ["target", "resolveTarget", "availableModels", "desktop", "hostLabel"],
    template: `
      <div data-test="mesh-studio">
        <span data-test="target-url">{{ target.baseUrl }}</span>
        <span data-test="target-key">{{ target.apiKey }}</span>
        <slot name="machine" />
      </div>
    `,
  },
}));

vi.mock("../lib/ipc", () => ({
  ipc: {},
}));

import MeshWorkflowStudio from "@studio/components/MeshWorkflowStudio.vue";
import HostChip from "../components/create/HostChip.vue";
import { useHostModelsStore } from "../stores/hostModels";
import MeshWorkflowView from "./MeshWorkflowView.vue";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";

function mountView() {
  const pinia = createPinia();
  setActivePinia(pinia);
  const connection = useConnectionStore();
  connection.info = {
    mode: "local",
    baseUrl: "http://127.0.0.1:49152",
    apiKey: null,
  };
  connection.status = "ready";
  const hosts = useHostsStore();
  hosts.extras.push({
    id: "renderbox-7680",
    label: "Render box",
    url: "http://renderbox:7680",
    apiKey: "remote-secret",
    status: "ready",
    error: null,
    instanceId: "renderbox-instance",
  });
  vi.spyOn(useHostModelsStore(), "refresh").mockResolvedValue(undefined);
  return mount(MeshWorkflowView, { global: { plugins: [pinia] } });
}

describe("MeshWorkflowView host routing", () => {
  beforeEach(() => vi.clearAllMocks());

  it("routes the complete studio to the selected authenticated machine", async () => {
    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.get("[data-test='target-url']").text()).toBe("http://127.0.0.1:49152");
    wrapper.findComponent(HostChip).vm.$emit("update:modelValue", "renderbox-7680");
    await flushPromises();

    expect(wrapper.get("[data-test='target-url']").text()).toBe("http://renderbox:7680");
    expect(wrapper.get("[data-test='target-key']").text()).toBe("remote-secret");
  });

  it("keeps an unavailable pinned machine visible instead of silently rerouting", async () => {
    const wrapper = mountView();
    const hosts = useHostsStore();
    wrapper.findComponent(HostChip).vm.$emit("update:modelValue", "renderbox-7680");
    await flushPromises();
    hosts.extras[0]!.status = "connecting";
    await flushPromises();

    expect(wrapper.get("[data-test='target-url']").text()).toBe("http://renderbox:7680");
    expect(wrapper.find("[data-test='mesh-studio']").exists()).toBe(true);
    wrapper.unmount();
  });
  it("filters complete workflows before Auto and Most capable rank machines", async () => {
    const wrapper = mountView();
    const hosts = useHostsStore();
    const inventory = useHostModelsStore();
    const mesh = {
      name: "hunyuan3d",
      family: "hunyuan3d",
      downloaded: true,
      generation_profile: {
        default_recipe_id: "shape",
        recipes: [
          {
            id: "shape",
            capabilities: {
              mesh: { workflow_modes: ["text_to_mesh"] },
            },
          },
        ],
      },
    };
    const image = { name: "z-image", family: "z-image", modality: "image", downloaded: true };
    for (const host of hosts.all)
      inventory.byHost[host.id] = {
        entries: [mesh, image] as never,
        fetchedAt: Date.now(),
        error: null,
      };
    hosts.telemetry.local = {
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      gpuInfo: { name: "Apple", backend: "metal", vram_total_mb: 64000, vram_used_mb: 0 },
    };
    hosts.telemetry["renderbox-7680"] = {
      queueDepth: 5,
      queueCapacity: 8,
      version: null,
      gpuInfo: { name: "CUDA", backend: "cuda", vram_total_mb: 96000, vram_used_mb: 0 },
    };
    await flushPromises();
    const resolve = wrapper.findComponent(MeshWorkflowStudio).props("resolveTarget")!;
    const request = {
      mode: "text_to_mesh" as const,
      meshModel: mesh.name,
      imageModel: image.name,
      texture: false,
      delight: false,
    };
    expect((await resolve(request)).target.baseUrl).toBe("http://127.0.0.1:49152");
    wrapper.findComponent(HostChip).vm.$emit("update:modelValue", "capable");
    await flushPromises();
    expect((await resolve(request)).target.baseUrl).toBe("http://renderbox:7680");
    inventory.byHost["renderbox-7680"]!.entries = [mesh] as never;
    expect((await resolve(request)).target.baseUrl).toBe("http://127.0.0.1:49152");
    wrapper.findComponent(HostChip).vm.$emit("update:modelValue", "renderbox-7680");
    await flushPromises();
    await expect(resolve(request)).rejects.toThrow("Render box cannot run all");
    wrapper.unmount();
  });
});

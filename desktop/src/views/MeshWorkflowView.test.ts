import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

// The view reads `?workflow=` — the queue row's route back here — so these
// cases mount it against a stub route they can rewrite.
const routeQuery = vi.hoisted(() => ({ value: {} as Record<string, string> }));
vi.mock("vue-router", () => ({ useRoute: () => ({ query: routeQuery.value }) }));

const generate = vi.hoisted(() => vi.fn());
vi.mock("@studio/components/MeshWorkflowStudio.vue", () => ({
  default: {
    props: ["target", "resolveTarget", "availableModels", "desktop", "hostLabel", "openWorkflow"],
    // Returning an object (not a render function) keeps the template below.
    setup: (_props: unknown, { expose }: { expose: (value: unknown) => void }) => {
      expose({ generate });
      return {};
    },
    template: `
      <div data-test="mesh-studio">
        <span data-test="target-url">{{ target.baseUrl }}</span>
        <span data-test="target-key">{{ target.apiKey }}</span>
        <span data-test="open-workflow">{{ openWorkflow }}</span>
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
import { useUiStore } from "../stores/ui";

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
  beforeEach(() => {
    vi.clearAllMocks();
    routeQuery.value = {};
  });

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
  it("refreshes newly ready hosts without refetching on telemetry changes", async () => {
    const wrapper = mountView();
    await flushPromises();
    const hosts = useHostsStore();
    const inventory = useHostModelsStore();
    vi.mocked(inventory.refresh).mockClear();
    hosts.extras[0]!.status = "connecting";
    await flushPromises();
    vi.mocked(inventory.refresh).mockClear();
    hosts.extras[0]!.status = "ready";
    await flushPromises();
    expect(inventory.refresh).toHaveBeenCalledTimes(1);
    hosts.telemetry["renderbox-7680"] = {
      queueDepth: 2,
      queueCapacity: 8,
      version: null,
      gpuInfo: { name: "CUDA", backend: "cuda", vram_total_mb: 96000, vram_used_mb: 1000 },
    };
    await flushPromises();
    expect(inventory.refresh).toHaveBeenCalledTimes(1);
    wrapper.unmount();
  });

  it("retains cached styles while their host reconnects", async () => {
    const wrapper = mountView();
    const inventory = useHostModelsStore();
    const row = { name: "mesh", family: "hunyuan3d", downloaded: true };
    inventory.byHost["renderbox-7680"] = {
      entries: [row] as never,
      fetchedAt: Date.now(),
      error: null,
    };
    wrapper.findComponent(HostChip).vm.$emit("update:modelValue", "renderbox-7680");
    await flushPromises();
    useHostsStore().extras[0]!.status = "connecting";
    await flushPromises();
    expect(wrapper.findComponent(MeshWorkflowStudio).props("availableModels")).toEqual([row]);
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

describe("MeshWorkflowView shell integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    routeQuery.value = {};
  });

  /*
   * The queue row's route back here. A 3-D Studio stage is admitted as an
   * ordinary generation, so clicking it used to land on New image — which
   * cannot resume a durable workflow at all.
   */
  it("opens on the workflow a deep link names", async () => {
    routeQuery.value = { workflow: "workflow-7" };
    const wrapper = mountView();
    await flushPromises();
    expect(wrapper.get("[data-test='open-workflow']").text()).toBe("workflow-7");
    wrapper.unmount();
  });

  /*
   * Before this, ⌘↩ raised the Generate intent AND pushed `/create`: pressing
   * it here left the view and rendered a picture, while the status bar
   * advertised the hint as though it worked.
   */
  it("generates here when the shell raises ⌘↩, and consumes the intent once", async () => {
    const wrapper = mountView();
    await flushPromises();
    const ui = useUiStore();

    ui.generate();
    await flushPromises();
    expect(generate).toHaveBeenCalledTimes(1);

    ui.generate();
    await flushPromises();
    expect(generate).toHaveBeenCalledTimes(2);
    wrapper.unmount();
  });

  it("does not generate for an intent another view already consumed", async () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const ui = useUiStore();
    ui.generate();
    expect(ui.consumeIntent("generate")).toBe(true);

    const wrapper = mountView();
    await flushPromises();
    expect(generate).not.toHaveBeenCalled();
    wrapper.unmount();
  });
});

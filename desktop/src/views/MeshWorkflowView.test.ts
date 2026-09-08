import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@studio/components/MeshWorkflowStudio.vue", () => ({
  default: {
    props: ["target"],
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
  return mount(MeshWorkflowView, { global: { plugins: [pinia] } });
}

describe("MeshWorkflowView host routing", () => {
  beforeEach(() => vi.clearAllMocks());

  it("routes the complete studio to the selected authenticated machine", async () => {
    const wrapper = mountView();
    await flushPromises();

    expect(wrapper.get("[data-test='target-url']").text()).toBe("http://127.0.0.1:49152");
    const picker = wrapper.get("[data-test='mesh-workflow-host']");
    expect(picker.findAll("option").map((option) => option.text())).toEqual([
      "This device · ready",
      "Render box · ready",
    ]);

    await picker.setValue("renderbox-7680");

    expect(wrapper.get("[data-test='target-url']").text()).toBe("http://renderbox:7680");
    expect(wrapper.get("[data-test='target-key']").text()).toBe("remote-secret");
  });

  it("keeps an unavailable pinned machine visible instead of silently rerouting", async () => {
    const wrapper = mountView();
    const hosts = useHostsStore();
    await wrapper.get("[data-test='mesh-workflow-host']").setValue("renderbox-7680");
    hosts.extras[0]!.status = "connecting";
    await flushPromises();

    expect(wrapper.get("[data-test='mesh-workflow-host']").element).toHaveProperty(
      "value",
      "renderbox-7680",
    );
    expect(wrapper.text()).toContain("Render box is reconnecting");
    expect(wrapper.find("[data-test='mesh-studio']").exists()).toBe(false);
  });
});

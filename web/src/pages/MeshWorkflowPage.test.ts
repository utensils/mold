import { flushPromises, mount } from "@vue/test-utils";
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

vi.mock("../composables/useHostRouting", async () => {
  const { ref } = await import("vue");
  const hosts = ref<Array<Record<string, unknown>>>([]);
  const targetId = ref("auto");
  return {
    useHostRouting: () => ({
      hosts,
      targetId,
      setTarget: (id: string) => {
        targetId.value = id;
      },
    }),
  };
});

import MeshWorkflowPage from "./MeshWorkflowPage.vue";
import { useHostRouting } from "../composables/useHostRouting";

const origin = {
  id: "origin",
  label: "This server",
  url: "http://studio:7680",
  status: "ready" as const,
  queueDepth: 0,
  gpu: null,
};
const remote = {
  id: "renderbox-7680",
  label: "Render box",
  url: "https://renderbox.example",
  apiKey: "remote-secret",
  status: "ready" as const,
  queueDepth: 0,
  gpu: null,
};

describe("MeshWorkflowPage host routing", () => {
  beforeEach(() => {
    const routing = useHostRouting();
    routing.hosts.value = [origin, remote];
    (routing.targetId as unknown as { value: string }).value = "auto";
  });

  it("routes models, workflows, and results to the selected authenticated host", async () => {
    const wrapper = mount(MeshWorkflowPage);
    await flushPromises();

    expect(wrapper.get("[data-test='target-url']").text()).toBe(
      "http://studio:7680",
    );
    await wrapper
      .get("[data-test='mesh-workflow-host']")
      .setValue("renderbox-7680");

    expect(wrapper.get("[data-test='target-url']").text()).toBe(
      "https://renderbox.example",
    );
    expect(wrapper.get("[data-test='target-key']").text()).toBe(
      "remote-secret",
    );
    expect(useHostRouting().targetId.value).toBe("renderbox-7680");
  });

  it("honours an explicitly pinned machine on entry", async () => {
    (useHostRouting().targetId as unknown as { value: string }).value =
      "renderbox-7680";
    const wrapper = mount(MeshWorkflowPage);
    await flushPromises();

    expect(wrapper.get("[data-test='target-url']").text()).toBe(
      "https://renderbox.example",
    );
    expect(
      (
        wrapper.get("[data-test='mesh-workflow-host']")
          .element as HTMLSelectElement
      ).value,
    ).toBe("renderbox-7680");
  });
});

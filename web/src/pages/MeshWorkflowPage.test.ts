import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

// The page reads `?workflow=` and `?host=` so a queue row can route back to a
// workflow on the machine that ran it; these cases mount it outside a router.
const routeQuery = vi.hoisted(() => ({ value: {} as Record<string, string> }));
vi.mock("vue-router", () => ({
  useRoute: () => ({ query: routeQuery.value }),
}));

vi.mock("@studio/components/MeshWorkflowStudio.vue", () => ({
  default: {
    props: ["target", "openWorkflow"],
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

vi.mock("../composables/useHostRouting", async () => {
  const { ref } = await import("vue");
  const hosts = ref<Array<Record<string, unknown>>>([]);
  const targetId = ref("auto");
  return {
    useHostRouting: () => ({
      hosts,
      targetId,
      targetModels: ref([]),
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
    routeQuery.value = {};
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

describe("MeshWorkflowPage — a link names the machine that ran the workflow", () => {
  beforeEach(() => {
    const routing = useHostRouting();
    routing.hosts.value = [origin, remote];
    (routing.targetId as unknown as { value: string }).value = "auto";
    routeQuery.value = {};
  });

  /*
   * A durable workflow lives on ONE machine. A link carrying only the id sent
   * a queue row from another machine to whichever one this page happened to
   * be browsing, which reported the workflow gone.
   *
   * This path had no coverage at all, and the binding was written as an
   * `immediate` watcher ABOVE the ref it writes — so it threw "Cannot access
   * 'selectedHostId' before initialization" out of setup in dev, and was
   * swallowed in production, silently restoring the very bug it fixed.
   */
  it("binds to the machine the link names before loading the workflow", async () => {
    routeQuery.value = { workflow: "workflow-7", host: "renderbox-7680" };
    const wrapper = mount(MeshWorkflowPage);
    await flushPromises();
    expect(wrapper.get("[data-test='target-url']").text()).toBe(
      "https://renderbox.example",
    );
    expect(wrapper.get("[data-test='target-key']").text()).toBe(
      "remote-secret",
    );
    expect(wrapper.get("[data-test='open-workflow']").text()).toBe(
      "workflow-7",
    );
    wrapper.unmount();
  });

  /* A machine this browser does not know is not a reason to move the pin. */
  it("leaves the machine alone when the link names one it does not have", async () => {
    routeQuery.value = { workflow: "workflow-7", host: "a-machine-we-forgot" };
    const wrapper = mount(MeshWorkflowPage);
    await flushPromises();
    expect(wrapper.get("[data-test='target-url']").text()).toBe(
      "http://studio:7680",
    );
    wrapper.unmount();
  });

  /*
   * Opening a 3-D queue row must not silently re-point New image's generate
   * target — that is an app-wide setting, and only the machine picker should
   * move it.
   */
  it("does not repoint the app's generate target", async () => {
    routeQuery.value = { workflow: "workflow-7", host: "renderbox-7680" };
    const wrapper = mount(MeshWorkflowPage);
    await flushPromises();
    expect(useHostRouting().targetId.value).toBe("auto");
    wrapper.unmount();
  });
});

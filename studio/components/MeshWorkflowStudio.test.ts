import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const createMeshWorkflow = vi.hoisted(() =>
  vi.fn(async () => ({ job_id: "workflow-1" })),
);
const listMeshWorkflows = vi.hoisted(() => vi.fn(async () => ({ jobs: [] })));

vi.mock("../api/meshWorkflows", () => ({
  createMeshWorkflow,
  listMeshWorkflows,
  getMeshWorkflow: vi.fn(async () => null),
  cancelMeshWorkflow: vi.fn(),
  resumeMeshWorkflow: vi.fn(),
  deleteMeshWorkflow: vi.fn(),
}));

vi.mock("../api/client", () => ({
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(async (_target, path: string) => {
    if (path === "/api/status") return { instance_id: "local-instance" };
    if (path === "/api/capabilities") return {};
    if (path === "/api/models") {
      return [
        {
          name: "hunyuan3d-mini-turbo:fp16",
          family: "hunyuan3d",
          downloaded: true,
          runtime_available: true,
          default_steps: 5,
          default_guidance: 5,
          default_width: 0,
          default_height: 0,
          generation_profile: {
            default_recipe_id: "shape",
            recipes: [
              {
                id: "shape",
                capabilities: {
                  canvasless: true,
                  mesh: { workflow_modes: ["text_to_mesh"] },
                },
              },
            ],
          },
        },
        {
          name: "z-image-turbo:q8",
          family: "z-image",
          modality: "image",
          downloaded: true,
          runtime_available: true,
          default_steps: 9,
          default_guidance: 0,
          default_width: 1024,
          default_height: 1024,
          generation_profile: {
            default_recipe_id: "image",
            recipes: [
              {
                id: "image",
                capabilities: { prompt: { mode: "required" } },
              },
            ],
          },
        },
      ];
    }
    throw new Error(`unexpected ${path}`);
  }),
}));

vi.mock("../api/referenceUploads", () => ({
  prepareReferenceUploads: vi.fn(),
  requestShouldUseReferenceUploads: vi.fn(() => false),
}));

vi.mock("./MeshViewer.vue", () => ({ default: { template: "<div />" } }));

import MeshWorkflowStudio from "./MeshWorkflowStudio.vue";

describe("MeshWorkflowStudio feature-aware PBR authoring", () => {
  beforeEach(() => vi.clearAllMocks());

  it("keeps text-to-mesh usable without sending texture controls to a host that lacks them", async () => {
    const wrapper = mount(MeshWorkflowStudio, {
      props: {
        target: { baseUrl: "http://metal-host:7680", apiKey: null },
      },
    });
    await flushPromises();

    expect(wrapper.find("[data-test='mesh-workflow-texture']").exists()).toBe(
      false,
    );
    expect(
      wrapper.get("[data-test='mesh-workflow-texture-unavailable']").text(),
    ).toContain("PBR painting is unavailable on this machine");

    await wrapper.get("textarea").setValue("A brass telescope");
    await wrapper.get("form").trigger("submit");
    await flushPromises();

    expect(createMeshWorkflow).toHaveBeenCalledTimes(1);
    const request = (
      createMeshWorkflow.mock.calls as unknown as Array<
        [unknown, { mode: string; mesh_request: { mesh?: unknown } }]
      >
    )[0]![1];
    expect(request.mode).toBe("text_to_mesh");
    expect(request.mesh_request.mesh).toBeUndefined();
  });
});

it("preserves the draft when telemetry recreates an equivalent target", async () => {
  const target = { baseUrl: "http://metal-host:7680", apiKey: null };
  const wrapper = mount(MeshWorkflowStudio, { props: { target } });
  await flushPromises();
  await wrapper.get("textarea").setValue("A brass telescope");
  const calls = listMeshWorkflows.mock.calls.length;
  for (let tick = 0; tick < 3; tick++) {
    await wrapper.setProps({ target: { ...target } });
    await flushPromises();
  }
  expect(listMeshWorkflows.mock.calls.length).toBe(calls);
  expect(wrapper.get("textarea").element.value).toBe("A brass telescope");
  expect(wrapper.text()).not.toContain("Loading workflow capabilities");
  wrapper.unmount();
});

it("submits and reads workflow history on the resolved owner", async () => {
  const target = { baseUrl: "http://remote:7680", apiKey: "remote-key" };
  const resolveTarget = vi.fn(async () => ({ target, label: "Render box" }));
  const wrapper = mount(MeshWorkflowStudio, {
    props: {
      target: { baseUrl: "http://local:7680", apiKey: null },
      resolveTarget,
    },
  });
  await flushPromises();
  await wrapper.get("textarea").setValue("A brass telescope");
  await wrapper.get("form").trigger("submit");
  await flushPromises();
  expect(resolveTarget).toHaveBeenCalledWith(
    expect.objectContaining({
      mode: "text_to_mesh",
      imageModel: "z-image-turbo:q8",
    }),
  );
  expect(createMeshWorkflow).toHaveBeenCalledWith(target, expect.any(Object));
  expect(listMeshWorkflows).toHaveBeenLastCalledWith(target);
  wrapper.unmount();
});

it("retains unchanged result media across progress polls and stops polling on unmount", async () => {
  const { getMeshWorkflow } = await import("../api/meshWorkflows");
  const { apiFetchTo } = await import("../api/client");
  vi.clearAllMocks();
  vi.useFakeTimers();
  vi.mocked(getMeshWorkflow).mockResolvedValue({
    id: "workflow-1",
    state: "running",
    mode: "text_to_mesh",
    stages: [],
    output_filename: "result.glb",
  } as never);
  vi.mocked(apiFetchTo).mockImplementation(
    async () => new Response(new Uint8Array([1])),
  );
  const createUrl = vi
    .spyOn(URL, "createObjectURL")
    .mockReturnValue("blob:result");
  const revokeUrl = vi
    .spyOn(URL, "revokeObjectURL")
    .mockImplementation(() => {});
  const wrapper = mount(MeshWorkflowStudio, {
    props: { target: { baseUrl: "http://local:7680", apiKey: null } },
  });
  try {
    await flushPromises();
    await wrapper.get("textarea").setValue("A brass telescope");
    await wrapper.get("form").trigger("submit");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(750);
    await flushPromises();
    expect(getMeshWorkflow).toHaveBeenCalledTimes(2);
    expect(apiFetchTo).toHaveBeenCalledTimes(2);
    wrapper.unmount();
    await vi.advanceTimersByTimeAsync(1500);
    expect(getMeshWorkflow).toHaveBeenCalledTimes(2);
  } finally {
    wrapper.unmount();
    createUrl.mockRestore();
    revokeUrl.mockRestore();
    vi.useRealTimers();
  }
});

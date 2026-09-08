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

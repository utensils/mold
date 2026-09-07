import { afterEach, describe, expect, it, vi } from "vitest";

import {
  cancelMeshWorkflow,
  createMeshWorkflow,
  deleteMeshWorkflow,
  getMeshWorkflow,
  listMeshWorkflows,
  meshWorkflowEventsUrl,
  resumeMeshWorkflow,
} from "./meshWorkflows";

const target = { baseUrl: "http://hal:7680", apiKey: "secret" };

afterEach(() => vi.unstubAllGlobals());

describe("mesh workflow API", () => {
  it("uses the durable workflow routes and preserves authentication", async () => {
    const fetchMock = vi.fn(
      async (_url: string, init?: RequestInit) =>
        new Response(
          init?.method === "POST" && String(_url).endsWith("/mesh-workflows")
            ? JSON.stringify({ job_id: "mesh-1" })
            : JSON.stringify({ jobs: [] }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const request = {
      mode: "mesh_texture" as const,
      texture_request: { model: "hunyuan3d-2.1:fp16" },
    };
    await expect(createMeshWorkflow(target, request)).resolves.toEqual({
      job_id: "mesh-1",
    });
    await listMeshWorkflows(target);
    await getMeshWorkflow(target, "mesh/1");
    await resumeMeshWorkflow(target, "mesh/1");
    await cancelMeshWorkflow(target, "mesh/1");
    await deleteMeshWorkflow(target, "mesh/1");

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://hal:7680/api/mesh-workflows",
      "http://hal:7680/api/mesh-workflows",
      "http://hal:7680/api/mesh-workflows/mesh%2F1",
      "http://hal:7680/api/mesh-workflows/mesh%2F1/resume",
      "http://hal:7680/api/mesh-workflows/mesh%2F1/cancel",
      "http://hal:7680/api/mesh-workflows/mesh%2F1",
    ]);
    for (const [, init] of fetchMock.mock.calls) {
      expect(new Headers(init?.headers).get("X-Api-Key")).toBe("secret");
    }
    expect(meshWorkflowEventsUrl(target, "mesh/1")).toBe(
      "http://hal:7680/api/mesh-workflows/mesh%2F1/events",
    );
  });
});

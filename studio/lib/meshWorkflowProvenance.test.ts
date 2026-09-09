import { describe, expect, it } from "vitest";

import {
  isMeshWorkflowLead,
  MESH_WORKFLOW_LEAD_ROLE,
  meshWorkflowHostFromQuery,
  meshWorkflowIdFromQuery,
  meshWorkflowProvenanceOf,
  meshWorkflowRouteFor,
} from "./meshWorkflowProvenance";

const run = {
  mesh_workflow: {
    job_id: "workflow-1",
    mode: "text_to_mesh",
    role: MESH_WORKFLOW_LEAD_ROLE,
    stage_index: 4,
  },
};

describe("3-D workflow provenance", () => {
  /*
   * The reported bug: clicking a 3-D Studio job in the queue landed on
   * Generate under its 3-D section. That view cannot resume a durable
   * workflow at all — the stages, Cancel, Resume and history live only under
   * /api/mesh-workflows.
   */
  it("routes work a workflow made to the 3-D Studio, on that workflow", () => {
    expect(meshWorkflowRouteFor(run)).toEqual({
      path: "/create/3d",
      query: { workflow: "workflow-1" },
    });
  });

  /*
   * A durable workflow lives on ONE machine: every read, poll, resume, cancel
   * and result fetch is bound to that host's authenticated target. A link
   * carrying only the id sent a queue row from another machine to whichever
   * one the studio happened to be browsing, which answered that the workflow
   * does not exist.
   */
  it("names the machine that ran it, so the link cannot ask the wrong server", () => {
    expect(meshWorkflowRouteFor(run, "hal9000-7680")?.query).toEqual({
      workflow: "workflow-1",
      host: "hal9000-7680",
    });
  });

  /* A caller that cannot know the machine leaves the studio's pin alone. */
  it("omits the machine rather than guessing one", () => {
    for (const hostId of [undefined, null, "", "   "])
      expect(meshWorkflowRouteFor(run, hostId)?.query.host).toBeUndefined();
  });

  it("reads the machine a deep link names, and nothing else", () => {
    expect(meshWorkflowHostFromQuery({ host: "hal9000-7680" })).toBe(
      "hal9000-7680",
    );
    for (const query of [null, undefined, {}, { host: 7 }])
      expect(meshWorkflowHostFromQuery(query as never)).toBe("");
  });

  /*
   * Absence is an ordinary print or an older host, NEVER a refusal — the
   * `supports_strength` lesson. The caller keeps the routing it already had.
   */
  it("answers nothing for a print no workflow made", () => {
    for (const carrier of [null, undefined, {}, { mesh_workflow: null }])
      expect(meshWorkflowRouteFor(carrier)).toBeNull();
  });

  /* The id is what every door needs; without it the block cannot be acted on. */
  it("refuses a block with no usable job id", () => {
    for (const job_id of ["", "   ", undefined as unknown as string])
      expect(
        meshWorkflowProvenanceOf({
          mesh_workflow: { ...run.mesh_workflow, job_id },
        }),
      ).toBeNull();
  });

  /*
   * A newer host may name a mode or role this build has never heard of. That
   * is still a workflow worth opening — only the id is load-bearing.
   */
  it("keeps routing a workflow whose mode and role this build does not know", () => {
    const future = {
      mesh_workflow: {
        job_id: "workflow-9",
        mode: "multiview_to_mesh",
        role: "retopologised_glb",
        stage_index: 7,
      },
    };
    expect(meshWorkflowRouteFor(future)?.query.workflow).toBe("workflow-9");
    expect(isMeshWorkflowLead(future)).toBe(false);
  });

  it("names only the mesh as its run's lead", () => {
    expect(isMeshWorkflowLead(run)).toBe(true);
    for (const role of ["generated_image", "matted_image", "delighted_image"])
      expect(
        isMeshWorkflowLead({ mesh_workflow: { ...run.mesh_workflow, role } }),
        role,
      ).toBe(false);
  });

  it("reads the workflow a deep link names, and nothing else", () => {
    expect(meshWorkflowIdFromQuery({ workflow: "workflow-1" })).toBe(
      "workflow-1",
    );
    expect(meshWorkflowIdFromQuery({ workflow: ["workflow-2", "x"] })).toBe(
      "workflow-2",
    );
    for (const query of [null, undefined, {}, { workflow: 7 }])
      expect(meshWorkflowIdFromQuery(query as never)).toBe("");
  });
});

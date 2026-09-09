/**
 * Which durable 3-D workflow made a piece of work — the one client-side
 * reading of the server's `mesh_workflow` provenance.
 *
 * Every stage of a workflow is admitted as an ordinary generation, so without
 * this a queue row and a finished print look exactly like a hand-authored
 * render: the row routed to New image, and a run's source picture, its matted
 * and delighted copies and its mesh sat in My images as unrelated tiles.
 *
 * ABSENCE IS NOT A REFUSAL. A print with no block was made outside the 3-D
 * Studio — or on a host that predates the field — and must keep behaving
 * exactly as it does today: one tile, routed to New image, no reopen door.
 */

/** The role whose print represents the whole run. */
export const MESH_WORKFLOW_LEAD_ROLE = "final_glb";

/** The 3-D Studio's route, and the query it reads a workflow id from. */
export const MESH_WORKFLOW_ROUTE = "/create/3d";
export const MESH_WORKFLOW_QUERY = "workflow";

export interface MeshWorkflowProvenance {
  job_id: string;
  /** `text_to_mesh` | `mesh_roundtrip` | `mesh_texture`, or a newer host's. */
  mode: string;
  /** `generated_image` | `matted_image` | `delighted_image` | `final_glb`. */
  role: string;
  stage_index: number;
}

/** The least a caller needs to carry for this module to answer. */
export interface MeshWorkflowProvenanceCarrier {
  mesh_workflow?: MeshWorkflowProvenance | null;
}

/**
 * The block, when the value is one this build can act on.
 *
 * A newer host may name a `mode` or `role` this build has never heard of, and
 * that is still a workflow worth routing to — only a missing or malformed
 * `job_id` makes the block unusable, because the id is what every door needs.
 */
export function meshWorkflowProvenanceOf(
  carrier: MeshWorkflowProvenanceCarrier | null | undefined,
): MeshWorkflowProvenance | null {
  const block = carrier?.mesh_workflow;
  if (!block || typeof block !== "object") return null;
  const jobId = typeof block.job_id === "string" ? block.job_id.trim() : "";
  if (!jobId) return null;
  return {
    job_id: jobId,
    mode: typeof block.mode === "string" ? block.mode : "",
    role: typeof block.role === "string" ? block.role : "",
    stage_index:
      typeof block.stage_index === "number" &&
      Number.isFinite(block.stage_index)
        ? block.stage_index
        : 0,
  };
}

/** Whether this output is the one a collapsed view shows for its run. */
export function isMeshWorkflowLead(
  carrier: MeshWorkflowProvenanceCarrier | null | undefined,
): boolean {
  return meshWorkflowProvenanceOf(carrier)?.role === MESH_WORKFLOW_LEAD_ROLE;
}

/**
 * Where work made by a workflow belongs — the 3-D Studio, opened on that
 * workflow. `null` means "not from a workflow", and the caller keeps whatever
 * routing it already had.
 *
 * A 3-D Studio job routed to New image is not merely the wrong view: New image
 * cannot resume a durable workflow at all. Its stages, its Cancel and Resume,
 * and its history live only under `/api/mesh-workflows`.
 */
export function meshWorkflowRouteFor(
  carrier: MeshWorkflowProvenanceCarrier | null | undefined,
): { path: string; query: Record<string, string> } | null {
  const provenance = meshWorkflowProvenanceOf(carrier);
  if (!provenance) return null;
  return {
    path: MESH_WORKFLOW_ROUTE,
    query: { [MESH_WORKFLOW_QUERY]: provenance.job_id },
  };
}

/** The workflow a `?workflow=` query names, if it names one. */
export function meshWorkflowIdFromQuery(
  query: Record<string, unknown> | null | undefined,
): string {
  const raw = query?.[MESH_WORKFLOW_QUERY];
  const value = Array.isArray(raw) ? raw[0] : raw;
  return typeof value === "string" ? value.trim() : "";
}

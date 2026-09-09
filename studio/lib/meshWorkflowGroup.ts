import {
  MESH_WORKFLOW_LEAD_ROLE,
  meshWorkflowProvenanceOf,
  type MeshWorkflowProvenanceCarrier,
} from "./meshWorkflowProvenance";

/**
 * Collapsing a 3-D workflow's several prints into one gallery item.
 *
 * Every stage of a durable workflow publishes an ordinary print, so one
 * text-to-3-D run with matting and lighting removal leaves FOUR tiles in My
 * images — the source picture, its matted and delighted copies, and the mesh —
 * sorted by time and indistinguishable from four things a person made. The
 * mesh is the result; the rest are how it was reached.
 *
 * This is a SECOND, orthogonal grouping to the cross-host one. That merge asks
 * "are these the same bytes on two machines" (`galleryPrintIdentity`: seed,
 * size and model); a run's picture and its mesh share none of those and can
 * never collapse through it. This asks "were these made by one run".
 *
 * ABSENCE IS AN ORDINARY PRINT. A row with no `mesh_workflow` was not made by
 * a workflow — or came from a host that predates the field — and must keep
 * behaving exactly as it does today: its own tile, no stack, no reopen door.
 */

/** The least a row needs to be sorted into a run. */
export interface GroupableRow {
  /** Stable per-row key the caller already has (usually the filename). */
  key: string;
  metadata: MeshWorkflowProvenanceCarrier | null | undefined;
}

/** What a single row needs to know about the run it belongs to. */
export interface MeshWorkflowGroupMembership {
  jobId: string;
  role: string;
  stageIndex: number;
  /** Whether this row is the one the grid shows. */
  lead: boolean;
  /** How many prints the run produced, this one included. */
  memberCount: number;
}

/**
 * Index every row by its run, in ONE pass.
 *
 * The Library's guards count operations, not milliseconds: a per-tile scan
 * over the gallery is what `organizationIndex` exists to prevent, so this is
 * built once per data change and every tile does a single `Map.get`.
 *
 * A run whose mesh has not landed yet — still rendering, or trashed on its own
 * — has no `final_glb` member. It still groups, and its LEAD falls to the
 * latest stage present, so the run keeps exactly one tile in the grid rather
 * than silently scattering back into several or vanishing entirely.
 */
export function indexMeshWorkflowGroups(rows: readonly GroupableRow[]): {
  membership: Map<string, MeshWorkflowGroupMembership>;
} {
  const staged = new Map<
    string,
    { members: { key: string; role: string; stage: number }[] }
  >();

  for (const row of rows) {
    const provenance = meshWorkflowProvenanceOf(row.metadata);
    if (!provenance) continue;
    const run = staged.get(provenance.job_id) ?? { members: [] };
    run.members.push({
      key: row.key,
      role: provenance.role,
      stage: provenance.stage_index,
    });
    staged.set(provenance.job_id, run);
  }

  const membership = new Map<string, MeshWorkflowGroupMembership>();

  for (const [jobId, run] of staged) {
    // The mesh leads. Without one, the latest stage does — a run still
    // rendering is one tile, not none and not four.
    const lead =
      run.members.find((member) => member.role === MESH_WORKFLOW_LEAD_ROLE) ??
      run.members.reduce((latest, member) =>
        member.stage >= latest.stage ? member : latest,
      );
    for (const member of run.members) {
      membership.set(member.key, {
        jobId,
        role: member.role,
        stageIndex: member.stage,
        lead: member.key === lead.key,
        memberCount: run.members.length,
      });
    }
  }

  return { membership };
}

/** The plain word for a member's part in its run. */
const ROLE_LABEL: Readonly<Record<string, string>> = {
  generated_image: "Source picture",
  matted_image: "Background removed",
  delighted_image: "Lighting removed",
  final_glb: "The 3-D object",
};

/** What a member is, in plain words; an unknown role still reads as itself. */
export function meshWorkflowRoleLabel(role: string): string {
  if (ROLE_LABEL[role]) return ROLE_LABEL[role];
  const spelled = role.replace(/_/g, " ").trim();
  return spelled ? spelled.charAt(0).toUpperCase() + spelled.slice(1) : "Step";
}

/**
 * Whether a row should be drawn in the grid at all.
 *
 * Only the run's lead is; the rest are reachable by opening it. They remain
 * real prints — reusable, exportable, deletable — and a row that belongs to no
 * run is always shown.
 */
export function showsInGrid(
  key: string,
  membership: ReadonlyMap<string, MeshWorkflowGroupMembership>,
): boolean {
  return membership.get(key)?.lead !== false;
}

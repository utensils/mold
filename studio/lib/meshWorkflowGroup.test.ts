import { describe, expect, it } from "vitest";

import {
  indexMeshWorkflowGroups,
  meshWorkflowRoleLabel,
  showsInGrid,
  type GroupableRow,
} from "./meshWorkflowGroup";

function member(key: string, role: string, stage: number, job = "run-1") {
  return {
    key,
    metadata: {
      mesh_workflow: {
        job_id: job,
        mode: "text_to_mesh",
        role,
        stage_index: stage,
      },
    },
  } satisfies GroupableRow;
}

/** What one text-to-3-D run with matting and delight actually publishes. */
const run = [
  member("source.png", "generated_image", 0),
  member("matted.png", "matted_image", 1),
  member("delighted.png", "delighted_image", 2),
  member("object.glb", "final_glb", 3),
];

describe("collapsing a 3-D run into one gallery item", () => {
  /*
   * The reported problem: a run leaves four tiles in My images, sorted by
   * time and indistinguishable from four things a person made. The mesh is
   * the result; the rest are how it was reached.
   */
  it("shows the mesh and hides the steps that made it", () => {
    const { groups, membership } = indexMeshWorkflowGroups(run);
    expect(groups.get("run-1")?.leadKey).toBe("object.glb");
    expect(groups.get("run-1")?.memberKeys).toEqual([
      "object.glb",
      "source.png",
      "matted.png",
      "delighted.png",
    ]);
    expect(membership.get("object.glb")?.memberCount).toBe(4);
    expect(showsInGrid("object.glb", membership)).toBe(true);
    for (const hidden of ["source.png", "matted.png", "delighted.png"])
      expect(showsInGrid(hidden, membership), hidden).toBe(false);
  });

  /* Absence is an ordinary print or an older host — never a refusal. */
  it("leaves every print no workflow made exactly where it is", () => {
    const rows: GroupableRow[] = [
      { key: "a.png", metadata: null },
      { key: "b.png", metadata: {} },
      { key: "c.png", metadata: { mesh_workflow: null } },
      ...run,
    ];
    const { membership } = indexMeshWorkflowGroups(rows);
    for (const key of ["a.png", "b.png", "c.png"])
      expect(showsInGrid(key, membership), key).toBe(true);
    expect(membership.has("a.png")).toBe(false);
  });

  /*
   * A run whose mesh has not landed — still rendering, or the mesh trashed on
   * its own — must stay ONE tile. Falling back to no lead would make the run
   * vanish; falling back to every member would scatter it again.
   */
  it("keeps a run with no mesh yet as a single tile led by its latest step", () => {
    const partial = run.slice(0, 3);
    const { groups, membership } = indexMeshWorkflowGroups(partial);
    expect(groups.get("run-1")?.leadKey).toBe("delighted.png");
    expect(
      partial.filter((row) => showsInGrid(row.key, membership)),
    ).toHaveLength(1);
  });

  it("keeps separate runs separate", () => {
    const { groups, membership } = indexMeshWorkflowGroups([
      ...run,
      member("other.glb", "final_glb", 3, "run-2"),
      member("other.png", "generated_image", 0, "run-2"),
    ]);
    expect(groups.size).toBe(2);
    expect(groups.get("run-2")?.leadKey).toBe("other.glb");
    expect(membership.get("other.png")?.jobId).toBe("run-2");
    expect(membership.get("object.glb")?.memberCount).toBe(4);
    expect(membership.get("other.glb")?.memberCount).toBe(2);
  });

  /* One pass over the rows, whatever the gallery's size — the Library's
   * guards count operations, and a per-tile scan is what they refuse. */
  it("indexes in one pass", () => {
    const rows: GroupableRow[] = [];
    for (let i = 0; i < 2000; i += 1)
      rows.push(member(`p${i}.png`, "generated_image", 0, `run-${i % 500}`));
    let reads = 0;
    const counted = rows.map((row) => ({
      key: row.key,
      get metadata() {
        reads += 1;
        return row.metadata;
      },
    }));
    indexMeshWorkflowGroups(counted);
    expect(reads).toBe(rows.length);
  });

  it("names a member's part in plain words, and an unknown one as itself", () => {
    expect(meshWorkflowRoleLabel("final_glb")).toBe("The 3-D object");
    expect(meshWorkflowRoleLabel("generated_image")).toBe("Source picture");
    expect(meshWorkflowRoleLabel("matted_image")).toBe("Background removed");
    expect(meshWorkflowRoleLabel("delighted_image")).toBe("Lighting removed");
    expect(meshWorkflowRoleLabel("retopologised_glb")).toBe(
      "Retopologised glb",
    );
    expect(meshWorkflowRoleLabel("")).toBe("Step");
  });
});

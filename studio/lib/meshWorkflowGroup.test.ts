import { describe, expect, it } from "vitest";

import {
  indexMeshWorkflowGroups,
  meshWorkflowRoleLabel,
  collapseToLeads,
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

/** The keys the grid would draw, given everything it was about to draw. */
function keysDrawn(
  rows: readonly GroupableRow[],
  membership: ReturnType<typeof indexMeshWorkflowGroups>["membership"],
): string[] {
  return collapseToLeads(rows, (row) => row.key, membership).map(
    (row) => row.key,
  );
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
    const { membership } = indexMeshWorkflowGroups(run);
    expect(membership.get("object.glb")).toMatchObject({
      jobId: "run-1",
      role: "final_glb",
      lead: true,
      memberCount: 4,
    });
    expect(keysDrawn(run, membership)).toEqual(["object.glb"]);
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
    expect(keysDrawn(rows, membership)).toEqual([
      "a.png",
      "b.png",
      "c.png",
      "object.glb",
    ]);
    expect(membership.has("a.png")).toBe(false);
  });

  /*
   * A run whose mesh has not landed — still rendering, or the mesh trashed on
   * its own — must stay ONE tile. Falling back to no lead would make the run
   * vanish; falling back to every member would scatter it again.
   */
  it("keeps a run with no mesh yet as a single tile led by its latest step", () => {
    const partial = run.slice(0, 3);
    const { membership } = indexMeshWorkflowGroups(partial);
    expect(membership.get("delighted.png")?.lead).toBe(true);
    expect(keysDrawn(partial, membership)).toEqual(["delighted.png"]);
  });

  it("keeps separate runs separate", () => {
    const { membership } = indexMeshWorkflowGroups([
      ...run,
      member("other.glb", "final_glb", 3, "run-2"),
      member("other.png", "generated_image", 0, "run-2"),
    ]);
    expect(membership.get("other.glb")?.lead).toBe(true);
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

/*
 * The rule is about REACHABILITY, and it is what every Library filter relies
 * on: a step is hidden because the lead is right there to open. Hand it a list
 * the lead did not survive and it must hand the step back, or the step is gone
 * from the screen with nothing left to open it.
 */
describe("collapseToLeads", () => {
  const { membership } = indexMeshWorkflowGroups(run);

  it("hides the steps when the lead is in the list", () => {
    expect(keysDrawn(run, membership)).toEqual(["object.glb"]);
  });

  it("hands a step back when the lead is not in the list", () => {
    const withoutMesh = run.filter((row) => row.key !== "object.glb");
    expect(keysDrawn(withoutMesh, membership)).toEqual([
      "source.png",
      "matted.png",
      "delighted.png",
    ]);
  });

  it("hides only the runs whose own lead is present", () => {
    const other = [
      member("other-source.png", "generated_image", 0, "run-2"),
      member("other.glb", "final_glb", 1, "run-2"),
    ];
    const index = indexMeshWorkflowGroups([...run, ...other]).membership;
    // run-1 keeps its mesh, run-2's is filtered out by whatever ran before.
    const drawn = keysDrawn([...run, other[0]!], index);
    expect(drawn).toEqual(["object.glb", "other-source.png"]);
  });

  it("never touches a print no run made", () => {
    const plain: GroupableRow[] = [{ key: "plain.png", metadata: null }];
    expect(keysDrawn([...plain, ...run], membership)).toEqual([
      "plain.png",
      "object.glb",
    ]);
  });
});

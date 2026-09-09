import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { meshWorkflowRouteFor } from "@studio/lib/meshWorkflowProvenance";
import viewSource from "./LibraryView.vue?raw";

/**
 * The Library's two doors onto a 3-D run. `reuseSettings` degrades a workflow
 * print to a one-shot on the mesh style, which is not what the person
 * authored — the run itself is the truer door.
 */
describe("the Library's 3-D run doors", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
  });

  /*
   * A durable workflow lives on ONE machine. A link that names only the id
   * asks whichever machine the studio was last browsing and is answered "no
   * longer on this machine" — the defect this rule exists to prevent, which
   * is easy to reintroduce at every new call site.
   */
  it("carries the owning machine from every reopen door", () => {
    const good = viewSource.match(
      /meshWorkflowRouteFor\(\s*entry\.item\.metadata,\s*workflowHostOf\(entry\)\s*\)/g,
    );
    expect(good?.length ?? 0).toBeGreaterThanOrEqual(2);
    // Every call site, no exceptions: a helper test would pass while a caller
    // forgot, which is exactly how this shipped wrong twice.
    const all = viewSource.split("meshWorkflowRouteFor(").length - 1;
    expect(all).toBe(good?.length ?? 0);
    // Never the representative copy's bucket — an auto-saved remote output
    // lands in this Mac's gallery, so `sourceKey` can say "local" for a run
    // this Mac never performed.
    expect(viewSource).not.toContain("metadata, entry.sourceKey)");
  });

  /* The route the doors build, given a print a run made. */
  it("routes to the run on its own machine", () => {
    const route = meshWorkflowRouteFor(
      {
        mesh_workflow: {
          job_id: "run-1",
          mode: "text_to_mesh",
          role: "final_glb",
          stage_index: 3,
        },
      },
      "hal9000-7680",
    );
    expect(route).toEqual({
      path: "/create/3d",
      query: { workflow: "run-1", host: "hal9000-7680" },
    });
  });

  /* Absence is an ordinary print or an older host — no door at all. */
  it("offers no door for a print no workflow made", () => {
    expect(meshWorkflowRouteFor({}, "local")).toBeNull();
  });
});

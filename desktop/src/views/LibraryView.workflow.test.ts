import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { meshWorkflowRouteFor } from "@studio/lib/meshWorkflowProvenance";
import viewSource from "./LibraryView.vue?raw";
import lightboxSource from "../components/gallery/Lightbox.vue?raw";
import releaseNote from "../../../changelog.d/library-3d-stacks.md?raw";
import desktopRules from "../../../.claude/rules/desktop.md?raw";

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
  /*
   * One set, one word. The drill-in chip said "3 prints", the context menu
   * said "3 assets" and the tile's assistive label said "3 pictures" — three
   * names for the same three things. "pictures" is the one that wins, and not
   * by taste: `docs/design/README.md` §2 lists "Prints" in the column of words
   * to REPLACE, and the Library's own chrome already says it — decisively
   * `CollectionCard.vue`, the album card counting the same kind of set with
   * the identical construct, plus TrashBanner and LibraryHeader. Unifying on
   * "prints" instead put a second name beside the one the grid already used.
   */
  it("calls a run's members by one name, on every surface that names them", () => {
    expect(viewSource).toContain('`3-D object · ${count} ${count === 1 ? "picture" : "pictures"}`');
    expect(viewSource).toContain("`Show the ${membership.memberCount} pictures`");
    expect(viewSource).toContain("`One 3-D run, ${tile.model.workflowCount} pictures`");
    // The Lightbox aside and the release note name the same set. A test that
    // read only this view stayed green while those two said "Show all N of
    // these" — three surfaces, two names, one of them checked.
    expect(lightboxSource).toContain("Show the {{ workflowAssetCount }} pictures");
    expect(releaseNote).toContain("**Show the N pictures**");
    // The agent rules are the FOURTH surface naming this set. A test that read
    // only the three shipped ones would let the rule file drift into the word
    // the design lexicon bans, which is where the next reader starts.
    expect(desktopRules).toContain("the number of pictures the run made");
    expect(desktopRules).not.toContain("the number of prints the run made");
    // Only the RUN's own wording — "N pictures" is right elsewhere (the Trash
    // confirm deletes pictures), so a blanket ban would be a false alarm.
    for (const source of [viewSource, lightboxSource, releaseNote])
      for (const wrong of [
        "Show all",
        "memberCount} assets",
        "memberCount} prints",
        // `docs/design/README.md` §2 lists "Prints" as a word to replace, and
        // the album card counting the same kind of set says "pictures".
        "workflowCount} prints",
        "workflowAssetCount }} of these",
      ])
        expect(source).not.toContain(wrong);
  });

  it("offers no door for a print no workflow made", () => {
    expect(meshWorkflowRouteFor({}, "local")).toBeNull();
  });
});

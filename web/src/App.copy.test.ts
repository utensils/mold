import { describe, expect, it } from "vitest";
import source from "./App.vue?raw";

/*
 * The shell's own words. Mounting App.vue means a router, a dozen stores and
 * a live poll for one sentence, so this reads the template instead — the same
 * idiom the desktop lexicon test uses.
 */
describe("App shell copy", () => {
  it("invites the reader into the queue in the app's own words", () => {
    expect(source).toContain("Open the queue →");
    // "View Queue" is title-cased UI-speak for a destination the lexicon
    // already names.
    expect(source).not.toContain("View Queue");
  });
});

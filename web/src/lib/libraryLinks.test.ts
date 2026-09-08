import { describe, expect, it } from "vitest";
import { libraryLink } from "./libraryLinks";
describe("library links", () => {
  it("keeps supported filters and encodes filenames while excluding secrets and unknown query state", () => {
    const link = new URL(
      libraryLink("https://mold.example", {
        scope: "trash",
        q: "a & b",
        print: "a #?.png",
        printHost: "studio",
        apiKey: "secret",
        token: "secret",
        mediaUrl: "https://elsewhere/private",
        type: ["video", "audio"],
      }),
    );
    expect(link.origin + link.pathname).toBe("https://mold.example/library");
    expect([...link.searchParams]).toEqual([
      ["scope", "trash"],
      ["q", "a & b"],
      ["print", "a #?.png"],
      ["printHost", "studio"],
    ]);
  });
});

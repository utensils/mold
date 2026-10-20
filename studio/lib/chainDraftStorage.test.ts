import { beforeEach, expect, it } from "vitest";
import { loadVideoClipsDraft, saveVideoClipsDraft } from "./chainDraftStorage";
beforeEach(() => {
  localStorage.clear();
});
it("keeps reviewed clip text and order scoped to the exact model and machine installation", () => {
  const clips = [
    { id: "one", prompt: "First", frames: 97 },
    { id: "two", prompt: "Second", frames: 49 },
  ];
  saveVideoClipsDraft("plato:original:model", clips);
  expect(loadVideoClipsDraft("plato:original:model")).toEqual(clips);
  expect(loadVideoClipsDraft("plato:replacement:model")).toBeNull();
});

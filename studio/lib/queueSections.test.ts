import { expect, it } from "vitest";
import { queueSection } from "./queueSections";
it("keeps failed and interrupted recoverable video parents out of Being made", () => {
  expect(queueSection("failed")).toBe("attention");
  expect(queueSection("interrupted")).toBe("attention");
  expect(queueSection("running")).toBe("making");
});
it("never counts a stopped parent as Being made", () => {
  // `/api/activity` keeps a stopped video parent visible as recovery work
  // (Start again, Delete); a header that called it "Making 1" lied.
  expect(queueSection("cancelled")).toBe("attention");
  // A parent whose clip still waits for a GPU is reported `queued`.
  expect(queueSection("queued")).toBe("waiting");
  expect(queueSection("cancelling")).toBe("making");
});

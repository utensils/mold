import { expect, it } from "vitest";
import { queueSection } from "./queueSections";
it("keeps failed and interrupted recoverable video parents out of Being made", () => {
  expect(queueSection("failed")).toBe("attention");
  expect(queueSection("interrupted")).toBe("attention");
  expect(queueSection("running")).toBe("making");
});

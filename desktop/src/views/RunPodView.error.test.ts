import { describe, expect, it } from "vitest";
import source from "./RunPodView.vue?raw";

describe("RunPodView operation errors", () => {
  it("renders persistent, selectable error details with an explicit dismiss action", () => {
    expect(source).toContain('v-if="runpod.operationError"');
    expect(source).toContain('role="alert"');
    expect(source).toContain("data-selectable");
    expect(source).toContain("{{ runpod.operationError }}");
    expect(source).toContain('@click="runpod.clearOperationError()"');
  });
});

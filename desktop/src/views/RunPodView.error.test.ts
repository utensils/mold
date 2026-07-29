import { describe, expect, it } from "vitest";
import source from "./RunPodView.vue?raw";

describe("RunPodView operation errors", () => {
  it("uses the shared copyable notice with an explicit dismiss action", () => {
    expect(source).toContain('v-if="runpod.operationError"');
    expect(source).toContain('import ErrorNotice from "@ui/components/ErrorNotice.vue"');
    expect(source).toContain(':message="runpod.operationError"');
    expect(source).toContain('@click="runpod.clearOperationError()"');
  });
});

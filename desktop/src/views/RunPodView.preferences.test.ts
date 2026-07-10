import { describe, expect, it } from "vitest";
import source from "./RunPodView.vue?raw";

describe("RunPodView preferences", () => {
  it("restores and persists the Hugging Face token option", () => {
    expect(source).toContain("const prefs = useAppPrefsStore()");
    expect(source).toContain("prefs.settings?.runpodIncludeHfToken");
    expect(source).toContain("prefs.update({ runpodIncludeHfToken: include })");
  });
});

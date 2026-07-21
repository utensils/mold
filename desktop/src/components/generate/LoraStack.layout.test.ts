import { describe, expect, it } from "vitest";
import source from "./LoraStack.vue?raw";

describe("LoraStack layout", () => {
  it("keeps the picker in document flow so the accordion grows instead of clipping it", () => {
    const picker = source.match(/<div\s+v-if="pickerOpen"\s+[^>]*>/s)?.[0] ?? "";
    expect(picker).toContain('data-test="lora-picker"');
    expect(picker).not.toContain(" absolute ");
  });
});

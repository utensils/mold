import { expect, it } from "vitest";
import { protectLegacyGenerateWire } from "./legacyGenerateWire";
it("protects literal prompt escapes on a wire copy without copying media or altering actual line endings", () => {
  const text = String.raw`C:\new\render` + "\r\nActual line";
  const request = {
    prompt: text,
    negative_prompt: text,
    original_prompt: text,
    prompt_transform: {
      source_prompt: text,
      root_prompt: text,
      unrelated: "keep",
    },
    references: [{ media: { authority: "inline", data: "pixels" } }],
    seed: 123,
  };
  const wire = protectLegacyGenerateWire(request);
  const protectedText = text.replaceAll("\\", "\\\\");
  expect(wire.prompt).toBe(protectedText);
  expect(wire.negative_prompt).toBe(protectedText);
  expect(wire.original_prompt).toBe(protectedText);
  expect(wire.prompt_transform.source_prompt).toBe(protectedText);
  expect(wire.prompt_transform.root_prompt).toBe(protectedText);
  expect(wire.references).toBe(request.references);
  expect(wire.seed).toBe(123);
  expect(request.prompt).toBe(text);
  expect(request.prompt_transform.source_prompt).toBe(text);
});

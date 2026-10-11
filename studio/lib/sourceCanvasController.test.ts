import { describe, expect, it } from "vitest";
import {
  SourceCanvasController,
  type SourceCanvasInput,
} from "./sourceCanvasController";

function input(overrides: Partial<SourceCanvasInput> = {}): SourceCanvasInput {
  return {
    source: { base64: "opening", width: 1024, height: 768 },
    closingSource: "closing",
    recipeKey: "flux:",
    model: "flux",
    pipeline: null,
    intent: "source",
    referenceOnly: false,
    primaryIsTarget: false,
    canvasless: false,
    ...overrides,
  };
}

describe("SourceCanvasController", () => {
  it("preserves a manual canvas on remount, recipe change and opening-frame removal", () => {
    const controller = new SourceCanvasController();
    expect(controller.update(input({ intent: "manual" })).canvas).toBeNull();
    expect(
      controller.update(input({ intent: "manual", recipeKey: "new:" })).canvas,
    ).toBeNull();
    const closing = controller.update(
      input({
        intent: "manual",
        recipeKey: "new:",
        source: { base64: "closing", width: 768, height: 1024 },
      }),
    );
    expect(closing).toMatchObject({
      canvas: null,
      intent: "manual",
      replaced: true,
    });
  });

  it("rearms an automatic canvas on deliberate source replacement", () => {
    const controller = new SourceCanvasController();
    controller.update(input({ intent: "manual" }));
    expect(
      controller.update(
        input({
          intent: "manual",
          source: { base64: "new", width: 1024, height: 1024 },
        }),
      ),
    ).toMatchObject({
      canvas: { width: 1024, height: 1024 },
      intent: "source",
    });
  });

  it("preserves a reused print's explicitly restored dimensions", () => {
    const controller = new SourceCanvasController();
    controller.update(input());
    controller.preserveRestored("reuse");
    expect(
      controller.update(
        input({
          source: { base64: "reuse", width: 1024, height: 1024 },
          intent: "manual",
        }),
      ),
    ).toMatchObject({ canvas: null, intent: "manual" });
  });

  it.each(["manual", "source", "source-exact", "model-default"] as const)(
    "removing every endpoint retains the canvas for %s",
    (intent) => {
      const controller = new SourceCanvasController();
      controller.update(input({ intent }));
      expect(controller.update(input({ source: null, intent }))).toMatchObject({
        canvas: null,
        dimensions: null,
        intent: intent === "manual" ? "manual" : "model-default",
      });
    },
  );

  it("does not let a reference strip or canvasless recipe change dimensions", () => {
    expect(
      new SourceCanvasController().update(input({ referenceOnly: true }))
        .canvas,
    ).toBeNull();
    expect(
      new SourceCanvasController().update(input({ canvasless: true })).canvas,
    ).toBeNull();
    expect(
      new SourceCanvasController().update(
        input({ referenceOnly: true, primaryIsTarget: true }),
      ).canvas,
    ).not.toBeNull();
  });
});

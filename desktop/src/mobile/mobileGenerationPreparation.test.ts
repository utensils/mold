import { beforeEach, describe, expect, it, vi } from "vitest";
import { SourceFitPreprocessCache } from "@ui/lib/sourceFitPreprocessCache";
import { hunyuan3dRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { ModelEntry } from "../lib/api/types";
import { applyModelDefaults, newGenerateForm } from "../lib/generateForm";

const { applyH3BoundaryFit, applySourceFitPreprocess } = vi.hoisted(() => ({
  applyH3BoundaryFit: vi.fn(),
  applySourceFitPreprocess: vi.fn(),
}));

vi.mock("../lib/sourceFitPreprocess", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/sourceFitPreprocess")>()),
  applyH3BoundaryFit,
  applySourceFitPreprocess,
}));

import { prepareMobileGenerationRequest } from "./mobileGenerationPreparation";

function model(overrides: Partial<ModelEntry> = {}): ModelEntry {
  return {
    name: "sdxl:test",
    family: "sdxl",
    size_gb: 1,
    is_loaded: false,
    hf_repo: "test/model",
    default_steps: 4,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "Test model",
    downloaded: true,
    ...overrides,
  };
}

function services() {
  return {
    cache: new SourceFitPreprocessCache(),
    ops: {
      imageSize: vi.fn(),
      fitImage: vi.fn(),
      buildMask: vi.fn(),
    },
    upscale: vi.fn(),
    onStatus: vi.fn(),
  };
}

describe("mobile generation request preparation", () => {
  beforeEach(() => {
    applyH3BoundaryFit.mockReset();
    applySourceFitPreprocess.mockReset();
  });

  it("fits an ordinary source and mask on the frozen draft", async () => {
    const selected = model();
    const draft = newGenerateForm();
    Object.assign(draft, {
      model: selected.name,
      family: selected.family,
      sourceImage: "original-source",
      sourceImageName: "source.png",
      maskImage: "original-mask",
      width: 768,
      height: 512,
    });
    applySourceFitPreprocess.mockResolvedValue({
      source: "fitted-source",
      mask: "fitted-mask",
      changed: true,
    });

    const request = await prepareMobileGenerationRequest(
      {
        target: { baseUrl: "http://studio.test:7680", apiKey: "secret" },
        draft,
        selectedModel: selected,
      },
      services(),
    );

    expect(applySourceFitPreprocess).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "original-source",
        mask: "original-mask",
        target: { width: 768, height: 512 },
      }),
      expect.anything(),
    );
    expect(request.source_image).toBe("fitted-source");
    expect(request.mask_image).toBe("fitted-mask");
  });

  it("fits Qwen edit input as maskless attachment media", async () => {
    const selected = model({ name: "qwen-image-edit:test", family: "qwen-image-edit" });
    const draft = newGenerateForm();
    Object.assign(draft, {
      model: selected.name,
      family: selected.family,
      imageAttachments: ["original-edit"],
      width: 1024,
      height: 768,
    });
    applySourceFitPreprocess.mockResolvedValue({
      source: "fitted-edit",
      mask: null,
      changed: true,
    });

    const request = await prepareMobileGenerationRequest(
      {
        target: { baseUrl: "http://studio.test:7680", apiKey: "secret" },
        draft,
        selectedModel: selected,
      },
      services(),
    );

    expect(applySourceFitPreprocess).toHaveBeenCalledWith(
      expect.objectContaining({ source: "original-edit", mask: null }),
      expect.anything(),
    );
    expect(request.edit_images).toEqual(["fitted-edit"]);
  });

  it("routes H3 frame boundaries through the dedicated fitter", async () => {
    const selected = model({
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
    });
    const draft = newGenerateForm();
    Object.assign(draft, {
      model: selected.name,
      family: selected.family,
      width: 832,
      height: 480,
    });
    draft.h3Authoring!.firstFrame = {
      data: "first-frame",
      filename: "first.png",
      mimeType: "image/png",
      width: 640,
      height: 360,
    };
    applyH3BoundaryFit.mockResolvedValue(draft.h3Authoring);

    await prepareMobileGenerationRequest(
      {
        target: { baseUrl: "http://studio.test:7680", apiKey: "secret" },
        draft,
        selectedModel: selected,
      },
      services(),
    );

    expect(applyH3BoundaryFit).toHaveBeenCalledWith(
      expect.objectContaining({ firstFrame: expect.objectContaining({ data: "first-frame" }) }),
      draft.sourceFit,
      { width: 832, height: 480 },
      expect.anything(),
    );
    expect(applySourceFitPreprocess).not.toHaveBeenCalled();
  });

  it("applies pending Ref2VA reference crops through the canvas ops before building the request", async () => {
    const selected = model({
      name: "minimax-h3-ref2va:comfy-pruned-int8",
      family: "minimax-h3",
    });
    const draft = newGenerateForm();
    Object.assign(draft, {
      model: selected.name,
      family: selected.family,
      prompt: "a subject",
    });
    draft.h3Authoring!.references = [
      {
        reference: {
          kind: "image",
          media: { authority: "inline", data: "SU1BR0U=" },
          provenance: { name: "subject.png", sha256: "a".repeat(64) },
          mime_type: "image/png",
          width: 1024,
          height: 768,
        },
        crop: { x: 256, y: 0, width: 512, height: 768 },
      },
    ];
    const deps = services();
    deps.ops.fitImage.mockResolvedValue("Q1JPUFBFRA==");

    const request = await prepareMobileGenerationRequest(
      {
        target: { baseUrl: "http://studio.test:7680", apiKey: "secret" },
        draft,
        selectedModel: selected,
      },
      deps,
    );

    expect(deps.ops.fitImage).toHaveBeenCalledWith("SU1BR0U=", {
      outputWidth: 512,
      outputHeight: 768,
      drawWidth: 1024,
      drawHeight: 768,
      offsetX: -256,
      offsetY: 0,
      maskPaddedPixels: false,
    });
    expect(request.references?.[0]).toMatchObject({
      kind: "image",
      media: { authority: "inline", data: "Q1JPUFBFRA==" },
      width: 512,
      height: 768,
      provenance: {
        name: "subject.png",
        crop: {
          x: 256,
          y: 0,
          width: 512,
          height: 768,
          source_width: 1024,
          source_height: 768,
          source_sha256: "a".repeat(64),
        },
      },
    });
    expect(request.references?.[0]?.provenance?.sha256).not.toBe("a".repeat(64));
    expect(applyH3BoundaryFit).not.toHaveBeenCalled();
    expect(applySourceFitPreprocess).not.toHaveBeenCalled();
  });

  /**
   * A canvasless recipe renders a mesh from the source photo itself: its
   * advertised canvas is 0×0, so running the ordinary fit would resize the
   * conditioning image to nothing before it ever reached the engine.
   */
  it("never fits the source toward a canvasless recipe's zero canvas", async () => {
    const selected = model({
      name: "hunyuan3d-mini-turbo:fp16",
      family: "hunyuan3d",
      source_image: "required",
      generation_profile: {
        schema_version: 1,
        profile_id: "hunyuan3d.mini",
        profile_hash: "hunyuan3d-mini-hash",
        default_recipe_id: "default",
        recipes: [hunyuan3dRecipe()],
      },
    });
    const draft = newGenerateForm();
    applyModelDefaults(draft, selected);
    Object.assign(draft, {
      sourceImage: "original-source",
      sourceImageName: "armchair.png",
    });
    expect(draft.width).toBe(0);
    expect(draft.height).toBe(0);

    const request = await prepareMobileGenerationRequest(
      {
        target: { baseUrl: "http://studio.test:7680", apiKey: "secret" },
        draft,
        selectedModel: selected,
      },
      services(),
    );

    expect(applySourceFitPreprocess).not.toHaveBeenCalled();
    expect(request.source_image).toBe("original-source");
    expect(request.width).toBe(0);
    expect(request.height).toBe(0);
    expect(request.source_fit).toBeUndefined();
    expect(request.output_format).toBe("glb");
  });

  it("suppresses late preprocessing status after the caller becomes stale", async () => {
    const selected = model();
    const draft = newGenerateForm();
    Object.assign(draft, {
      model: selected.name,
      family: selected.family,
      sourceImage: "source",
    });
    applySourceFitPreprocess.mockImplementation(async (_input, dependencies) => {
      dependencies.onStatus?.("late status");
      return { source: "source", mask: null, changed: false };
    });
    const dependencies = services();

    await prepareMobileGenerationRequest(
      {
        target: { baseUrl: "http://studio.test:7680", apiKey: "secret" },
        draft,
        selectedModel: selected,
        isCurrent: () => false,
      },
      dependencies,
    );

    expect(dependencies.onStatus).not.toHaveBeenCalled();
  });
});

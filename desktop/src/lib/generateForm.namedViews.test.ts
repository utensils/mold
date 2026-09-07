import { describe, expect, it } from "vitest";
import { applyRequestToForm, buildRequest, newGenerateForm } from "./generateForm";

describe("desktop named mesh views", () => {
  it("builds canonical named-image references from the advertised profile", () => {
    const form = newGenerateForm();
    form.family = "hunyuan3d";
    form.model = "hunyuan3d-2mv-turbo:fp16";
    form.recipeCapabilities = {
      outputFormats: ["glb"],
      defaultOutputFormat: "glb",
      canvasless: true,
      supportsStrength: false,
      promptMode: "ignored",
      referenceImages: null,
      mesh: {
        octree_resolutions: [256],
        octree_default: 256,
        threshold: { default: 0.6, min: 0, max: 1, step: 0.01, mode: "adjustable" },
        target_faces_min: 1_000,
        target_faces_max: 1_000_000,
        texture: { mode: "adjustable", required: false },
        named_views: {
          mode: "adjustable",
          roles: ["front", "left", "back", "right"],
          min_count: 1,
          max_count: 4,
        },
      },
    };
    form.namedViews = {
      right: { base64: "RIGHT", filename: "right.jpg", mimeType: "image/jpeg", width: 8, height: 9 },
      front: { base64: "FRONT", filename: "front.png", mimeType: "image/png", width: 10, height: 11 },
    };

    expect(buildRequest(form).references).toEqual([
      expect.objectContaining({ kind: "named_image", role: "front" }),
      expect.objectContaining({ kind: "named_image", role: "right" }),
    ]);
  });

  it("keeps parked named views off recipes that do not advertise them", () => {
    const form = newGenerateForm();
    form.namedViews = {
      front: { base64: "FRONT", filename: "front.png", mimeType: "image/png", width: 10, height: 11 },
    };
    expect(buildRequest(form).references).toBeUndefined();
  });

  it("round-trips inline named views through exact request restoration", () => {
    const form = newGenerateForm();
    form.family = "hunyuan3d";
    form.model = "hunyuan3d-2mv-turbo:fp16";
    form.recipeCapabilities = {
      outputFormats: ["glb"],
      defaultOutputFormat: "glb",
      canvasless: true,
      supportsStrength: false,
      promptMode: "ignored",
      referenceImages: null,
      mesh: {
        octree_resolutions: [256],
        octree_default: 256,
        threshold: { default: 0.5, min: 0, max: 1, step: 0.01, mode: "adjustable" },
        target_faces_min: 1_000,
        target_faces_max: 1_000_000,
        texture: { mode: "adjustable", required: false },
        named_views: {
          mode: "adjustable",
          roles: ["front", "left", "back", "right"],
          min_count: 1,
          max_count: 4,
        },
      },
    };
    form.namedViews = {
      front: { base64: "FRONT", filename: "front.png", mimeType: "image/png", width: 10, height: 11 },
      back: { base64: "BACK", filename: "back.jpg", mimeType: "image/jpeg", width: 12, height: 13 },
    };
    const request = buildRequest(form);

    const restored = newGenerateForm();
    applyRequestToForm(restored, request, []);
    restored.family = "hunyuan3d";
    restored.recipeCapabilities = form.recipeCapabilities;

    expect(buildRequest(restored).references).toEqual(request.references);
  });
});

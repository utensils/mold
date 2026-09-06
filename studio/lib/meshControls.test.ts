import { describe, expect, it } from "vitest";
import { hunyuan3dRecipe } from "./generationProfile.testFixtures";
import {
  emptyMeshForm,
  meshFormFromMetadata,
  meshRequestFromForm,
  meshStatsLabel,
  type MeshFormState,
} from "./meshControls";

const caps = () => hunyuan3dRecipe().capabilities.mesh!;

describe("emptyMeshForm", () => {
  it("holds no explicit values so the profile defaults apply", () => {
    expect(emptyMeshForm()).toEqual({
      octreeResolution: null,
      threshold: null,
      targetFaces: null,
    });
  });
});

describe("meshRequestFromForm", () => {
  it("omits the block entirely when every value is untouched", () => {
    expect(meshRequestFromForm(emptyMeshForm(), caps())).toBeUndefined();
  });

  it("omits values equal to the advertised defaults", () => {
    const form: MeshFormState = {
      octreeResolution: 256,
      threshold: 0.6,
      targetFaces: null,
    };
    expect(meshRequestFromForm(form, caps())).toBeUndefined();
  });

  it("carries only the values that differ from the defaults", () => {
    const form: MeshFormState = {
      octreeResolution: 192,
      threshold: 0.6,
      targetFaces: 50_000,
    };
    expect(meshRequestFromForm(form, caps())).toEqual({
      octree_resolution: 192,
      target_faces: 50_000,
    });
  });

  it("carries every explicit value when no profile advertises defaults", () => {
    const form: MeshFormState = {
      octreeResolution: 256,
      threshold: 0.6,
      targetFaces: null,
    };
    expect(meshRequestFromForm(form, undefined)).toEqual({
      octree_resolution: 256,
      threshold: 0.6,
    });
  });

  it("drops non-finite and non-positive values", () => {
    const form: MeshFormState = {
      octreeResolution: Number.NaN,
      threshold: 0.4,
      targetFaces: 0,
    };
    expect(meshRequestFromForm(form, caps())).toEqual({ threshold: 0.4 });
  });

  it("carries the selected PBR texture stage and advertised atlas size", () => {
    const textureCaps = {
      ...caps(),
      texture: { mode: "adjustable" as const, required: false },
      texture_resolutions: [1024, 2048, 4096],
      texture_default_resolution: 2048,
    };
    expect(
      meshRequestFromForm(
        { ...emptyMeshForm(), texture: true, textureResolution: null },
        textureCaps,
      ),
    ).toEqual({ texture: true, texture_resolution: 2048 });
    expect(
      meshRequestFromForm(
        { ...emptyMeshForm(), texture: true, textureResolution: 4096 },
        textureCaps,
      ),
    ).toEqual({ texture: true, texture_resolution: 4096 });
  });
});

describe("meshFormFromMetadata", () => {
  it("restores the recorded controls", () => {
    expect(
      meshFormFromMetadata({
        octree_resolution: 320,
        threshold: 0.55,
        target_faces: 40_000,
        texture: true,
        texture_resolution: 4096,
      }),
    ).toEqual({
      octreeResolution: 320,
      threshold: 0.55,
      targetFaces: 40_000,
      texture: true,
      textureResolution: 4096,
    });
  });

  it("reads absent, null, and non-mesh metadata as an empty form", () => {
    expect(meshFormFromMetadata(undefined)).toEqual(emptyMeshForm());
    expect(meshFormFromMetadata(null)).toEqual(emptyMeshForm());
    expect(
      meshFormFromMetadata({ octree_resolution: null, threshold: null }),
    ).toEqual(emptyMeshForm());
  });
});

describe("meshStatsLabel", () => {
  it("formats triangles, vertices, and the bounding size", () => {
    expect(
      meshStatsLabel(24_576, 49_152, {
        min: [-0.5, -0.4, -0.3],
        max: [0.5, 0.4, 0.3],
      }),
    ).toBe("49,152 tris · 24,576 verts · 1.00×0.80×0.60");
  });

  it("accepts the wire's separate min/max arrays", () => {
    expect(meshStatsLabel(10, 20, [-1, -1, -1], [1, 1, 1])).toBe(
      "20 tris · 10 verts · 2.00×2.00×2.00",
    );
  });

  it("omits the size when bounds are unknown", () => {
    expect(meshStatsLabel(10, 20, null)).toBe("20 tris · 10 verts");
    expect(meshStatsLabel(null, null, null)).toBe("");
  });
});

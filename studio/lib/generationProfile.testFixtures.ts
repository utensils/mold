/**
 * Recipe fixtures shaped exactly like `docs/generated/generation-profiles-v1.json`
 * (the `hunyuan3d-mini-turbo` and `cyberrealistic-pony` entries), so the
 * studio tests exercise the wire the server actually emits rather than a
 * hand-trimmed approximation of it.
 */
import type { GenerationRecipeProfile } from "./generationProfile";

/** The canvasless GLB mesh recipe: prompt ignored, no strength, mesh block. */
export function hunyuan3dRecipe(): GenerationRecipeProfile {
  return {
    id: "default",
    label: "Default",
    request_selector: {},
    defaults: { width: 0, height: 0, steps: 5, guidance: 5.0 },
    resolution: {
      domain: "none",
      alignment: 1,
      min_width: 0,
      min_height: 0,
      max_pixels: 0,
      aspect_groups: [],
    },
    steps: {
      default: 5,
      min: 1,
      max: 100,
      step: 1,
      recommended: [5],
      mode: "adjustable",
    },
    guidance: {
      default: 5.0,
      min: 0.0,
      max: 100.0,
      step: 0.1,
      mode: "adjustable",
    },
    capabilities: {
      guidance: { adjustable: true, supports_negative_prompt: false },
      negative_prompt: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not encode a negative prompt.",
      },
      source_image: "required",
      supports_lora: false,
      supports_controlnet: false,
      supports_identity: false,
      supports_sequence: false,
      supports_extend: false,
      supports_audio: false,
      source_video: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept a source video.",
      },
      mask: {
        mode: "hidden",
        required: false,
        reason: "This model does not accept an inpainting mask.",
      },
      keyframes: {
        mode: "hidden",
        required: false,
        reason: "This model does not accept keyframes.",
      },
      audio: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept source audio.",
      },
      lora: {
        mode: "hidden",
        max_count: 0,
        reason: "This model does not accept LoRA adapters.",
      },
      controlnet: {
        mode: "hidden",
        max_count: 0,
        reason: "ControlNet generation is available for SD1.5 models.",
      },
      output: {
        default_format: "glb",
        formats: ["glb"],
        audio_requires_mp4: false,
        delivery_reason:
          "3-D delivery uses binary glTF; OBJ, STL and PLY are available as gallery exports.",
      },
      wan_recipe: {
        mode: "hidden",
        supports_distill_strength: false,
        supports_first_last_frame: false,
        reason: "Wan sampler controls apply only to Wan models.",
      },
      prompt: {
        mode: "ignored",
        reason:
          "This model has no text encoder; the prompt is saved as a note.",
      },
      supports_strength: false,
      mesh: {
        octree_resolutions: [128, 192, 256, 320, 384],
        octree_default: 256,
        threshold: {
          default: 0.6,
          min: 0.0,
          max: 1.0,
          step: 0.01,
          mode: "adjustable",
        },
        target_faces_min: 100,
        target_faces_max: 2_000_000,
        texture: {
          mode: "hidden",
          required: false,
          reason:
            "PBR texture generation is not available in this build; omit mesh.texture to render geometry only",
        },
      },
    },
    provenance: [
      {
        kind: "mold-policy",
        source: "mold-qualified compatibility profile",
        qualified: true,
        evidence: "mold.generation-profile.v1",
      },
    ],
  };
}

/** A plain raster recipe: prompt required, strength read, no mesh block. */
export function sdxlRecipe(): GenerationRecipeProfile {
  return {
    id: "default",
    label: "Default",
    request_selector: {},
    defaults: { width: 1024, height: 1024, steps: 25, guidance: 7.0 },
    resolution: {
      domain: "dynamic",
      alignment: 16,
      min_width: 64,
      min_height: 64,
      max_pixels: 1_800_000,
      aspect_groups: [
        {
          id: "1:1",
          label: "1:1",
          presets: [
            { id: "1024x1024", width: 1024, height: 1024, tier: "recommended" },
          ],
        },
      ],
    },
    steps: {
      default: 25,
      min: 1,
      max: 100,
      step: 1,
      recommended: [25],
      mode: "adjustable",
    },
    guidance: {
      default: 7.0,
      min: 0.0,
      max: 100.0,
      step: 0.1,
      mode: "adjustable",
    },
    capabilities: {
      guidance: { adjustable: true, supports_negative_prompt: true },
      negative_prompt: { mode: "adjustable", required: false },
      supports_lora: true,
      supports_controlnet: false,
      supports_identity: false,
      supports_sequence: false,
      supports_extend: false,
      supports_audio: false,
      source_video: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept a source video.",
      },
      mask: { mode: "adjustable", required: false },
      keyframes: {
        mode: "hidden",
        required: false,
        reason: "This model does not accept keyframes.",
      },
      audio: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept source audio.",
      },
      lora: { mode: "adjustable", max_count: 4 },
      controlnet: {
        mode: "hidden",
        max_count: 0,
        reason: "ControlNet generation is available for SD1.5 models.",
      },
      output: {
        default_format: "png",
        formats: ["png", "jpeg", "webp"],
        audio_requires_mp4: false,
      },
      wan_recipe: {
        mode: "hidden",
        supports_distill_strength: false,
        supports_first_last_frame: false,
        reason: "Wan sampler controls apply only to Wan models.",
      },
      schedulers: ["ddim", "euler-ancestral", "uni-pc"],
      prompt: { mode: "required" },
      supports_strength: true,
    },
    provenance: [
      {
        kind: "mold-policy",
        source: "mold-qualified compatibility profile",
        qualified: true,
        evidence: "mold.generation-profile.v1",
      },
    ],
  };
}

/**
 * FLUX.2 [dev]: references REPLACE the source image, the mask and LoRA.
 *
 * Copied verbatim from `docs/generated/generation-profiles-v1.json`
 * (`flux2-dev:bf16`), so these tests exercise the wire the server actually emits.
 */
export function flux2DevRecipe(): GenerationRecipeProfile {
  return {
    id: "default",
    label: "Default",
    request_selector: {},
    defaults: {
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 4.0,
    },
    resolution: {
      domain: "dynamic",
      alignment: 16,
      min_width: 64,
      min_height: 64,
      max_pixels: 1800000,
      aspect_groups: [
        {
          id: "1:1",
          label: "1:1",
          presets: [
            {
              id: "768x768",
              width: 768,
              height: 768,
              tier: "recommended",
            },
            {
              id: "1024x1024",
              width: 1024,
              height: 1024,
              tier: "recommended",
            },
          ],
        },
        {
          id: "4:3",
          label: "4:3",
          presets: [
            {
              id: "1024x768",
              width: 1024,
              height: 768,
              tier: "recommended",
            },
          ],
        },
        {
          id: "3:4",
          label: "3:4",
          presets: [
            {
              id: "768x1024",
              width: 768,
              height: 1024,
              tier: "recommended",
            },
          ],
        },
        {
          id: "16:9",
          label: "16:9",
          presets: [
            {
              id: "1024x576",
              width: 1024,
              height: 576,
              tier: "recommended",
            },
          ],
        },
        {
          id: "9:16",
          label: "9:16",
          presets: [
            {
              id: "576x1024",
              width: 576,
              height: 1024,
              tier: "recommended",
            },
          ],
        },
      ],
    },
    steps: {
      default: 50,
      min: 1,
      max: 100,
      step: 1,
      recommended: [50],
      mode: "adjustable",
    },
    guidance: {
      default: 4.0,
      min: 0.0,
      max: 100.0,
      step: 0.1,
      mode: "adjustable",
    },
    capabilities: {
      guidance: {
        adjustable: true,
        supports_negative_prompt: false,
      },
      negative_prompt: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not encode a negative prompt.",
      },
      supports_lora: false,
      supports_controlnet: false,
      supports_identity: false,
      supports_sequence: false,
      supports_extend: false,
      supports_audio: false,
      source_video: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept a source video.",
      },
      mask: {
        mode: "hidden",
        required: false,
        reason: "This model does not accept an inpainting mask.",
      },
      keyframes: {
        mode: "hidden",
        required: false,
        reason: "This model does not accept keyframes.",
      },
      audio: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept source audio.",
      },
      lora: {
        mode: "hidden",
        max_count: 0,
        reason: "This model does not accept LoRA adapters.",
      },
      controlnet: {
        mode: "hidden",
        max_count: 0,
        reason: "ControlNet generation is available for SD1.5 models.",
      },
      output: {
        default_format: "png",
        formats: ["png", "jpeg", "webp"],
        audio_requires_mp4: false,
      },
      wan_recipe: {
        mode: "hidden",
        supports_distill_strength: false,
        supports_first_last_frame: false,
        reason: "Wan sampler controls apply only to Wan models.",
      },
      prompt: {
        mode: "required",
      },
      supports_strength: false,
      reference_images: {
        mode: "adjustable",
        required: false,
        max_count: 4,
        primary_is_target: false,
        source_relation: "replaces",
        max_pixels_single: 4096576,
        max_pixels_multi: 1048576,
      },
    },
    provenance: [
      {
        kind: "mold-policy",
        source: "mold-qualified compatibility profile",
        qualified: true,
        evidence: "mold.generation-profile.v1",
      },
    ],
  } as GenerationRecipeProfile;
}

/**
 * FLUX.2 [klein]: a source image OR references, never both in one pass — and it keeps img2img strength, the repaint mask and LoRA.
 *
 * Copied verbatim from `docs/generated/generation-profiles-v1.json`
 * (`flux2-klein:bf16`), so these tests exercise the wire the server actually emits.
 */
export function flux2KleinRecipe(): GenerationRecipeProfile {
  return {
    id: "default",
    label: "Default",
    request_selector: {},
    defaults: {
      width: 1024,
      height: 1024,
      steps: 4,
      guidance: 1.0,
    },
    resolution: {
      domain: "dynamic",
      alignment: 16,
      min_width: 64,
      min_height: 64,
      max_pixels: 1800000,
      aspect_groups: [
        {
          id: "1:1",
          label: "1:1",
          presets: [
            {
              id: "768x768",
              width: 768,
              height: 768,
              tier: "recommended",
            },
            {
              id: "1024x1024",
              width: 1024,
              height: 1024,
              tier: "recommended",
            },
          ],
        },
        {
          id: "4:3",
          label: "4:3",
          presets: [
            {
              id: "1024x768",
              width: 1024,
              height: 768,
              tier: "recommended",
            },
          ],
        },
        {
          id: "3:4",
          label: "3:4",
          presets: [
            {
              id: "768x1024",
              width: 768,
              height: 1024,
              tier: "recommended",
            },
          ],
        },
        {
          id: "16:9",
          label: "16:9",
          presets: [
            {
              id: "1024x576",
              width: 1024,
              height: 576,
              tier: "recommended",
            },
          ],
        },
        {
          id: "9:16",
          label: "9:16",
          presets: [
            {
              id: "576x1024",
              width: 576,
              height: 1024,
              tier: "recommended",
            },
          ],
        },
      ],
    },
    steps: {
      default: 4,
      min: 1,
      max: 100,
      step: 1,
      recommended: [4],
      mode: "adjustable",
    },
    guidance: {
      default: 1.0,
      min: 0.0,
      max: 100.0,
      step: 0.1,
      mode: "adjustable",
    },
    capabilities: {
      guidance: {
        adjustable: true,
        supports_negative_prompt: false,
      },
      negative_prompt: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not encode a negative prompt.",
      },
      supports_lora: true,
      supports_controlnet: false,
      supports_identity: false,
      supports_sequence: false,
      supports_extend: false,
      supports_audio: false,
      source_video: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept a source video.",
      },
      mask: {
        mode: "adjustable",
        required: false,
      },
      keyframes: {
        mode: "hidden",
        required: false,
        reason: "This model does not accept keyframes.",
      },
      audio: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept source audio.",
      },
      lora: {
        mode: "adjustable",
        max_count: 4,
      },
      controlnet: {
        mode: "hidden",
        max_count: 0,
        reason: "ControlNet generation is available for SD1.5 models.",
      },
      output: {
        default_format: "png",
        formats: ["png", "jpeg", "webp"],
        audio_requires_mp4: false,
      },
      wan_recipe: {
        mode: "hidden",
        supports_distill_strength: false,
        supports_first_last_frame: false,
        reason: "Wan sampler controls apply only to Wan models.",
      },
      prompt: {
        mode: "required",
      },
      supports_strength: true,
      reference_images: {
        mode: "adjustable",
        required: false,
        max_count: 4,
        primary_is_target: false,
        source_relation: "exclusive",
        max_pixels_single: 4096576,
        max_pixels_multi: 1048576,
      },
    },
    provenance: [
      {
        kind: "mold-policy",
        source: "mold-qualified compatibility profile",
        qualified: true,
        evidence: "mold.generation-profile.v1",
      },
    ],
  } as GenerationRecipeProfile;
}

/**
 * Qwen-Image-Edit: the first image is the edit TARGET, count unbounded.
 *
 * Copied verbatim from `docs/generated/generation-profiles-v1.json`
 * (`qwen-image-edit-2511:q4`), so these tests exercise the wire the server actually emits.
 */
export function qwenImageEditRecipe(): GenerationRecipeProfile {
  return {
    id: "default",
    label: "Default",
    request_selector: {},
    defaults: {
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 4.0,
    },
    resolution: {
      domain: "source-driven",
      alignment: 16,
      min_width: 64,
      min_height: 64,
      max_pixels: 1800000,
      source_max_pixels: 1048576,
      aspect_groups: [
        {
          id: "1:1",
          label: "1:1",
          presets: [
            {
              id: "1328x1328",
              width: 1328,
              height: 1328,
              tier: "recommended",
            },
          ],
        },
        {
          id: "\u224816:9",
          label: "\u224816:9",
          presets: [
            {
              id: "1664x928",
              width: 1664,
              height: 928,
              tier: "recommended",
            },
          ],
        },
        {
          id: "\u22489:16",
          label: "\u22489:16",
          presets: [
            {
              id: "928x1664",
              width: 928,
              height: 1664,
              tier: "recommended",
            },
          ],
        },
        {
          id: "4:3",
          label: "4:3",
          presets: [
            {
              id: "1472x1104",
              width: 1472,
              height: 1104,
              tier: "recommended",
            },
          ],
        },
        {
          id: "3:4",
          label: "3:4",
          presets: [
            {
              id: "1104x1472",
              width: 1104,
              height: 1472,
              tier: "recommended",
            },
          ],
        },
        {
          id: "3:2",
          label: "3:2",
          presets: [
            {
              id: "1584x1056",
              width: 1584,
              height: 1056,
              tier: "recommended",
            },
          ],
        },
        {
          id: "2:3",
          label: "2:3",
          presets: [
            {
              id: "1056x1584",
              width: 1056,
              height: 1584,
              tier: "recommended",
            },
          ],
        },
      ],
    },
    steps: {
      default: 50,
      min: 1,
      max: 100,
      step: 1,
      recommended: [50],
      mode: "adjustable",
    },
    guidance: {
      default: 4.0,
      min: 0.0,
      max: 100.0,
      step: 0.1,
      mode: "adjustable",
    },
    capabilities: {
      guidance: {
        adjustable: true,
        supports_negative_prompt: true,
      },
      negative_prompt: {
        mode: "adjustable",
        required: false,
      },
      supports_lora: true,
      supports_controlnet: false,
      supports_identity: false,
      supports_sequence: false,
      supports_extend: false,
      supports_audio: false,
      source_video: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept a source video.",
      },
      mask: {
        mode: "hidden",
        required: false,
        reason: "This model does not accept an inpainting mask.",
      },
      keyframes: {
        mode: "hidden",
        required: false,
        reason: "This model does not accept keyframes.",
      },
      audio: {
        mode: "hidden",
        required: false,
        reason: "This recipe does not accept source audio.",
      },
      lora: {
        mode: "adjustable",
        max_count: 4,
      },
      controlnet: {
        mode: "hidden",
        max_count: 0,
        reason: "ControlNet generation is available for SD1.5 models.",
      },
      output: {
        default_format: "png",
        formats: ["png", "jpeg", "webp"],
        audio_requires_mp4: false,
      },
      wan_recipe: {
        mode: "hidden",
        supports_distill_strength: false,
        supports_first_last_frame: false,
        reason: "Wan sampler controls apply only to Wan models.",
      },
      prompt: {
        mode: "required",
      },
      supports_strength: false,
      reference_images: {
        mode: "adjustable",
        required: true,
        primary_is_target: true,
        source_relation: "replaces",
        max_pixels_single: 1048576,
        max_pixels_multi: 1048576,
      },
    },
    provenance: [
      {
        kind: "mold-policy",
        source: "Mold source-driven Qwen Image Edit guidance",
        qualified: true,
        evidence:
          "source fitting preserves the input aspect on the dynamic /16 canvas and caps edit inputs at upstream's 1024x1024 VAE area; optional shape presets reuse Mold's qualified Qwen Image aspect set",
      },
    ],
  } as GenerationRecipeProfile;
}

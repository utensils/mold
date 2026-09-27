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
          "3-D delivery uses binary glTF; OBJ, OBJ+PBR ZIP, STL and PLY are available as gallery exports.",
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
        target_faces_texture_default: 40_000,
        texture: {
          mode: "hidden",
          required: false,
          reason:
            "PBR texture generation is not available in this build; omit mesh.texture to render geometry only",
        },
        matting: {
          mode: "adjustable",
          default: "auto",
          choices: ["auto", "on", "off"],
          reason: "Auto preserves useful alpha and removes opaque backgrounds.",
        },
        delight: {
          mode: "adjustable",
          required: false,
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
      // The three-rung ladder the server publishes for an adjustable control:
      // half the default, the default, half again.
      recommended: [13, 25, 38],
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
 * SDXL with IP-Adapter image prompting: the first — and so far only —
 * `combines` recipe. The reference is an image PROMPT, so it rides WITH the
 * source image, its strength, the mask, ControlNet and a LoRA, and the
 * adapter's own injection strength travels in the block as a `FloatControl`.
 *
 * Copied from `generation_profile::reference_images_for_recipe`'s `sd15|sdxl`
 * arm, so these tests exercise the wire the server actually emits.
 */
export function sdxlIpAdapterRecipe(): GenerationRecipeProfile {
  const recipe = sdxlRecipe();
  return {
    ...recipe,
    capabilities: {
      ...recipe.capabilities,
      reference_images: {
        mode: "adjustable",
        required: false,
        max_count: 1,
        primary_is_target: false,
        source_relation: "combines",
        reason: null,
        weight: {
          default: 1.0,
          min: 0.0,
          max: 2.0,
          step: 0.05,
          mode: "adjustable",
        },
      },
    },
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

/**
 * Qwen Image 2.1: up to ten ordered references that REPLACE the source,
 * sized from the last one (`canvas: last-reference`), PNG/JPEG/WebP, plus
 * the transparent-background block.
 *
 * Copied verbatim from `docs/generated/generation-profiles-v1.json`
 * (`qwen-image-2.1:bf16`), so these tests exercise the wire the server
 * actually emits.
 */
export function qwenImage21Recipe(): GenerationRecipeProfile {
  return {
    id: "default",
    label: "Default",
    request_selector: {},
    defaults: {
      width: 1024,
      height: 1024,
      steps: 40,
      guidance: 1.0,
    },
    resolution: {
      domain: "dynamic",
      alignment: 32,
      min_width: 64,
      min_height: 64,
      max_pixels: 4300800,
      max_axis_pixels: 2752,
      aspect_groups: [
        {
          id: "1:1",
          label: "1:1",
          presets: [
            {
              id: "1024x1024",
              width: 1024,
              height: 1024,
              tier: "recommended",
            },
            {
              id: "2048x2048",
              width: 2048,
              height: 2048,
              tier: "recommended",
            },
          ],
        },
        {
          id: "4:3",
          label: "4:3",
          presets: [
            {
              id: "1184x896",
              width: 1184,
              height: 896,
              tier: "recommended",
            },
            {
              id: "2400x1792",
              width: 2400,
              height: 1792,
              tier: "recommended",
            },
          ],
        },
        {
          id: "3:4",
          label: "3:4",
          presets: [
            {
              id: "896x1184",
              width: 896,
              height: 1184,
              tier: "recommended",
            },
            {
              id: "1792x2400",
              width: 1792,
              height: 2400,
              tier: "recommended",
            },
          ],
        },
        {
          id: "3:2",
          label: "3:2",
          presets: [
            {
              id: "1248x832",
              width: 1248,
              height: 832,
              tier: "recommended",
            },
            {
              id: "2528x1696",
              width: 2528,
              height: 1696,
              tier: "recommended",
            },
          ],
        },
        {
          id: "2:3",
          label: "2:3",
          presets: [
            {
              id: "832x1248",
              width: 832,
              height: 1248,
              tier: "recommended",
            },
            {
              id: "1696x2528",
              width: 1696,
              height: 2528,
              tier: "recommended",
            },
          ],
        },
        {
          id: "16:9",
          label: "16:9",
          presets: [
            {
              id: "1376x768",
              width: 1376,
              height: 768,
              tier: "recommended",
            },
            {
              id: "2752x1536",
              width: 2752,
              height: 1536,
              tier: "recommended",
            },
          ],
        },
        {
          id: "9:16",
          label: "9:16",
          presets: [
            {
              id: "768x1376",
              width: 768,
              height: 1376,
              tier: "recommended",
            },
            {
              id: "1536x2752",
              width: 1536,
              height: 2752,
              tier: "recommended",
            },
          ],
        },
      ],
    },
    steps: {
      default: 40,
      min: 1,
      max: 100,
      step: 1,
      recommended: [20, 40, 60],
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
        required: false,
        max_count: 10,
        primary_is_target: false,
        source_relation: "replaces",
        max_pixels_single: 1048576,
        max_pixels_multi: 1048576,
        canvas: "last-reference",
        formats: ["png", "jpeg", "webp"],
      },
      transparency: {
        mode: "adjustable",
        default: false,
        formats: ["png", "webp"],
        native_alpha: true,
      },
    },
    provenance: [
      {
        kind: "upstream",
        source:
          "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/README.md",
        revision: "b3179ad355be050328e483a9dfdd9e60cd62adfa",
        qualified: true,
        evidence:
          "docs/qualification/qwen-image-2.1-metal-uat.json: SHA-256-verified official checkpoint, full default 1024x1024/40-step Metal render, and decoded RGB PNG delivery; the seven 2K presets are the pinned README's Supported Aspect Ratios table, admitted by the family's 2400x1792 / 2752 px ceilings, with CUDA renders recorded in docs/qualification/qwen-image-2.1-cuda-performance.json",
      },
      {
        kind: "mold-policy",
        source: "Mold ~1 MP Qwen Image 2.1 aspect presets",
        qualified: true,
        evidence:
          "6 Mold-chosen ~1 MP presets (1184x896, 896x1184, 1248x832, 832x1248, 1376x768, 768x1376) keep the 2K table's aspects at the default area on the 32 px grid; not published upstream",
      },
    ],
  } as GenerationRecipeProfile;
}

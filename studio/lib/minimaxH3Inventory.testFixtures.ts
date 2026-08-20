import type { MiniMaxH3CapabilityRecord } from "./minimaxH3Inventory";

export const AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256 = "a".repeat(64);
export const AUTHENTICATED_MINIMAX_H3_TURBO_PROFILE_SHA256 = "c".repeat(64);

export const MINIMAX_H3_TURBO_8STEP_MODEL =
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step";
export const MINIMAX_H3_TURBO_4STEP_768P_MODEL =
  "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p";

/** The installed capability plus additive Turbo variants: the 8-step adapter
 * landed and executable, the 4-step 768p adapter still missing. */
export function authenticatedMiniMaxH3TurboCapabilities(): MiniMaxH3CapabilityRecord {
  const record = authenticatedMiniMaxH3Capabilities();
  record.minimax_h3!.partitions[0]!.turbo = [
    {
      model: MINIMAX_H3_TURBO_8STEP_MODEL,
      display_name: "MiniMax H3 FL2VA Turbo 8-step",
      tier: "Turbo 8-step",
      adapter_size_bytes: 1_956_193_000,
      installed: true,
      request: {
        width: 1344,
        height: 768,
        frames: 124,
        fps: 24,
        steps: 9,
        batch_size: 1,
        output_format: "mp4",
        required_endpoint: "first",
        generation_profile_sha256:
          AUTHENTICATED_MINIMAX_H3_TURBO_PROFILE_SHA256,
      },
    },
    {
      model: MINIMAX_H3_TURBO_4STEP_768P_MODEL,
      display_name: "MiniMax H3 FL2VA Turbo 4-step 768p",
      tier: "Turbo 4-step 768p",
      adapter_size_bytes: 1_956_192_992,
      installed: false,
    },
  ];
  return record;
}

/** Complete installed compact FL2VA capability used by cross-surface tests. */
export function authenticatedMiniMaxH3Capabilities(): MiniMaxH3CapabilityRecord {
  const definitions: Array<
    [
      string,
      "transformer" | "qwen" | "processor" | "video-vae" | "audio-vae",
      "fl2va" | "shared",
    ]
  > = [
    ["transformer", "transformer", "fl2va"],
    ["qwen", "qwen", "shared"],
    ["processor", "processor", "shared"],
    ["video-vae", "video-vae", "shared"],
    ["audio-vae", "audio-vae", "shared"],
  ];
  const components = definitions.map(([id, role, scope]) => ({
    id,
    display_name: id,
    kind: role === "transformer" ? "checkpoint" : "component",
    role,
    scope,
    size_bytes: 1,
    state: "installed" as const,
  }));
  return {
    model_access: {
      restrictions: [
        {
          code: "MINIMAX_H3_AUTHORIZATION_REQUIRED",
          family: "minimax-h3",
          message: "MiniMax H3 is not activated.",
          license_url: "https://example.test/license",
          authorization_url: "https://example.test/authorize",
        },
      ],
    },
    minimax_h3: {
      runtime_available: true,
      qualification: {
        backend: "cuda",
        metal_supported: false,
        minimum_host_ram_bytes: 1,
        minimum_vram_bytes: 1,
        attention_profile: "reviewed attention",
        quantization_profile: "reviewed compact layout",
      },
      partitions: [
        {
          task: "fl2va",
          model: "minimax-h3-fl2va:comfy-pruned-int8",
          display_name: "MiniMax H3 FL2VA",
          runtime_available: true,
          tier: "Compact",
          component_ids: components.map((component) => component.id),
          request: {
            width: 1344,
            height: 768,
            frames: 124,
            fps: 24,
            steps: 21,
            batch_size: 1,
            output_format: "mp4",
            required_endpoint: "first",
            generation_profile_sha256: AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256,
          },
        },
      ],
      components,
    },
  };
}

import type { MiniMaxH3CapabilityRecord } from "./minimaxH3Inventory";

export const AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256 = "a".repeat(64);

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

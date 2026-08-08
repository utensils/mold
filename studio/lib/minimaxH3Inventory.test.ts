import { describe, expect, it } from "vitest";
import {
  formatMiniMaxH3Bytes,
  planMiniMaxH3Install,
  presentMiniMaxH3Fleet,
  presentMiniMaxH3Host,
  type MiniMaxH3Capability,
  type MiniMaxH3ComponentCapability,
  type MiniMaxH3HostInput,
} from "./minimaxH3Inventory";

const GB = 1_000_000_000;

function component(
  overrides: Partial<MiniMaxH3ComponentCapability> &
    Pick<
      MiniMaxH3ComponentCapability,
      "id" | "display_name" | "kind" | "role" | "scope"
    >,
): MiniMaxH3ComponentCapability {
  return {
    size_bytes: GB,
    state: "installed",
    ...overrides,
  };
}

function h3Capability(): MiniMaxH3Capability {
  return {
    runtime_available: true,
    qualification: {
      backend: "cuda",
      metal_supported: false,
      minimum_host_ram_bytes: 128 * GB,
      minimum_vram_bytes: 24 * GB,
      attention_profile: "SDPA · tiled",
      quantization_profile: "BF16 transformer · INT8 text",
    },
    components: [
      component({
        id: "shared-qwen",
        display_name: "Qwen text encoder",
        kind: "text-encoder",
        role: "qwen",
        scope: "shared",
        size_bytes: 10 * GB,
      }),
      component({
        id: "shared-processor",
        display_name: "Qwen processor",
        kind: "tokenizer",
        role: "processor",
        scope: "shared",
        size_bytes: 2 * GB,
      }),
      component({
        id: "shared-video-vae",
        display_name: "Visual VAE",
        kind: "vae",
        role: "video-vae",
        scope: "shared",
        size_bytes: 5 * GB,
        state: "missing",
      }),
      component({
        id: "shared-audio-vae",
        display_name: "Audio VAE",
        kind: "vae",
        role: "audio-vae",
        scope: "shared",
        size_bytes: 3 * GB,
        state: "downloading",
        download: { job_id: "job-audio", bytes_done: GB, bytes_total: 3 * GB },
      }),
      component({
        id: "fl2va-transformer",
        display_name: "FL2VA transformer",
        kind: "checkpoint",
        role: "transformer",
        scope: "fl2va",
        size_bytes: 20 * GB,
      }),
      component({
        id: "ref2va-transformer",
        display_name: "Ref2VA transformer",
        kind: "checkpoint",
        role: "transformer",
        scope: "ref2va",
        size_bytes: 30 * GB,
        state: "failed",
        error: "checksum mismatch",
      }),
    ],
    partitions: [
      {
        task: "fl2va",
        model: "minimax-h3-fl2va",
        display_name: "MiniMax H3 FL2VA",
        runtime_available: true,
        tier: "24 GB VRAM",
        component_ids: [
          "fl2va-transformer",
          "shared-qwen",
          "shared-processor",
          "shared-video-vae",
          "shared-audio-vae",
        ],
      },
      {
        task: "ref2va",
        model: "minimax-h3-ref2va",
        display_name: "MiniMax H3 Ref2VA",
        runtime_available: true,
        tier: "48 GB VRAM",
        component_ids: [
          "ref2va-transformer",
          "shared-qwen",
          "shared-processor",
          "shared-video-vae",
          "shared-audio-vae",
        ],
      },
    ],
  };
}

function host(
  id = "render-a",
  capability: MiniMaxH3Capability | null = h3Capability(),
): MiniMaxH3HostInput {
  return {
    id,
    label: id === "render-a" ? "Render A" : "Render B",
    capabilities: { minimax_h3: capability },
  };
}

describe("MiniMax H3 inventory presentation", () => {
  it("partitions tasks, deduplicates shared components, and accounts exact disk", () => {
    const presented = presentMiniMaxH3Host(host());
    expect(presented).not.toBeNull();
    expect(
      presented?.tasks.map((task) => [
        task.taskLabel,
        task.readiness,
        task.diskBytes,
      ]),
    ).toEqual([
      ["FL2VA", "downloading", 40 * GB],
      ["Ref2VA", "failed", 50 * GB],
    ]);
    expect(presented?.sharedComponents.map((entry) => entry.role)).toEqual([
      "qwen",
      "processor",
      "video-vae",
      "audio-vae",
    ]);
    expect(presented?.bothTasksDiskBytes).toBe(70 * GB);
    expect(presented?.remainingBytes).toBe(38 * GB);
    expect(
      presented?.components.find((entry) => entry.role === "qwen")?.kindLabel,
    ).toBe("Text encoder");
  });

  it("plans one or both tasks on an exact host without duplicate shared work", () => {
    const presented = presentMiniMaxH3Host(host())!;
    const fl2va = planMiniMaxH3Install(presented, ["fl2va"]);
    expect(fl2va).toMatchObject({
      hostId: "render-a",
      action: "repair",
      bytesToDownload: 5 * GB,
    });
    expect(fl2va?.components.map((entry) => entry.id)).toEqual([
      "shared-video-vae",
    ]);
    expect(fl2va?.downloads).toEqual([
      {
        hostId: "render-a",
        hostLabel: "Render A",
        jobId: "job-audio",
        bytesDone: GB,
        bytesTotal: 3 * GB,
      },
    ]);

    const both = planMiniMaxH3Install(presented, ["fl2va", "ref2va", "fl2va"]);
    expect(both?.components.map((entry) => entry.id)).toEqual([
      "shared-video-vae",
      "ref2va-transformer",
    ]);
    expect(both?.bytesToDownload).toBe(35 * GB);
    expect(both?.downloads).toHaveLength(1);
  });

  it("keeps mixed-fleet download ownership isolated through refresh and reconnect", () => {
    const first = presentMiniMaxH3Fleet([host("render-a"), host("render-b")]);
    expect(first).toHaveLength(2);
    expect(
      first.map(
        (entry) => planMiniMaxH3Install(entry, ["fl2va"])?.downloads[0]?.hostId,
      ),
    ).toEqual(["render-a", "render-b"]);

    const refreshedB = h3Capability();
    const audio = refreshedB.components.find(
      (entry) => entry.id === "shared-audio-vae",
    )!;
    audio.state = "installed";
    delete audio.download;
    const refreshed = presentMiniMaxH3Fleet([
      { ...host("render-a"), capabilities: undefined },
      host("render-b", refreshedB),
    ]);
    expect(refreshed.map((entry) => entry.hostId)).toEqual(["render-b"]);
    expect(planMiniMaxH3Install(refreshed[0]!, ["fl2va"])?.downloads).toEqual(
      [],
    );
  });

  it("deduplicates one exact-host download job shared by multiple components", () => {
    const capability = h3Capability();
    const processor = capability.components.find(
      (entry) => entry.id === "shared-processor",
    )!;
    processor.state = "downloading";
    processor.download = {
      job_id: "job-audio",
      bytes_done: GB,
      bytes_total: 3 * GB,
    };
    const presented = presentMiniMaxH3Host(host("render-a", capability))!;
    expect(
      planMiniMaxH3Install(presented, ["fl2va", "ref2va"])?.downloads,
    ).toHaveLength(1);
  });

  it("uses install only when a target has no landed or active components", () => {
    const capability = h3Capability();
    for (const entry of capability.components) {
      entry.state = "missing";
      delete entry.download;
      delete entry.error;
    }
    const presented = presentMiniMaxH3Host(host("render-a", capability))!;
    const plan = planMiniMaxH3Install(presented, ["fl2va", "ref2va"]);
    expect(plan?.action).toBe("install");
    expect(plan?.components).toHaveLength(6);
    expect(plan?.bytesToDownload).toBe(70 * GB);
  });

  it("fails closed for access denial, disabled runtime, Metal, and malformed graphs", () => {
    const denied = host();
    denied.capabilities = {
      ...denied.capabilities,
      model_access: {
        restrictions: [
          {
            code: "license-not-accepted",
            family: "MiniMax-H3",
            message: "Unavailable",
            license_url: "https://example.invalid/license",
            authorization_url: "https://example.invalid/auth",
          },
        ],
      },
    };
    expect(presentMiniMaxH3Host(denied)).toBeNull();

    const disabled = h3Capability();
    disabled.runtime_available = false;
    expect(presentMiniMaxH3Host(host("render-a", disabled))).toBeNull();

    const partitionDisabled = h3Capability();
    partitionDisabled.partitions[0]!.runtime_available = false;
    expect(
      presentMiniMaxH3Host(host("render-a", partitionDisabled)),
    ).toBeNull();

    const metal = h3Capability();
    Object.assign(metal.qualification, {
      backend: "metal",
      metal_supported: true,
    });
    expect(presentMiniMaxH3Host(host("render-a", metal))).toBeNull();

    const malformed = h3Capability();
    malformed.components = malformed.components.filter(
      (entry) => entry.role !== "audio-vae",
    );
    expect(presentMiniMaxH3Host(host("render-a", malformed))).toBeNull();
  });

  it("formats host-supplied byte facts without manufacturing precision", () => {
    expect(formatMiniMaxH3Bytes(24 * GB)).toBe("24.0 GB");
    expect(formatMiniMaxH3Bytes(500_000_000)).toBe("500 MB");
  });
});

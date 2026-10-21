import { expect, it, vi, beforeEach } from "vitest";
const { json, fetch, inventory } = vi.hoisted(() => ({
  json: vi.fn(),
  fetch: vi.fn(),
  inventory: vi.fn(),
}));
vi.mock("./client", () => ({ apiJsonTo: json, apiFetchTo: fetch }));
vi.mock("./gallerySourceMedia", () => ({
  retainedSourceMediaInventory: inventory,
}));
import { inspectLegacyVideoSettings } from "./legacyVideoSettings";
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "installation",
  target: { baseUrl: "http://plato", apiKey: "private" },
};
const metadata = {
  model: "opaque",
  prompt: "summary",
  seed: 18446744073709552000,
  width: 768,
  height: 768,
  steps: 9,
  guidance: 0,
  fps: 24,
  negative_prompt: " first\\n\r\n ",
  chain: {
    stage_count: 2,
    motion_tail_frames: 1,
    stages: [
      {
        prompt: " A\\n\r\n ",
        frames: 107,
        transition: "smooth",
        seed: "18446744073709551615",
        loras: [{ path: "adapter", scale: 0.5 }],
      },
      {
        prompt: " B ",
        frames: 131,
        transition: "fade",
        fade_frames: 4,
        seed: "18446744073709551614",
        loras: [],
      },
    ],
  },
};
beforeEach(() => {
  inventory.mockReset();
  json.mockReset();
  fetch.mockReset();
  inventory.mockResolvedValue({ availability: "available", members: [] });
  json.mockImplementation(async (_t, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : { metadata, archive_identity_sha256: "a".repeat(64), members: [] },
  );
});
it("restores literal ordered stages and effective full-u64 seeds, not rounded metadata.seed", async () => {
  const result = await inspectLegacyVideoSettings(owner, "movie.mp4", metadata);
  expect(result.request.seed).toBe("18446744073709551615");
  expect(result.request.stages).toMatchObject([
    {
      prompt: " A\\n\r\n ",
      frames: 107,
      seed_offset: "0",
      loras: [{ path: "adapter", scale: 0.5 }],
      negative_prompt: " first\\n\r\n ",
    },
    {
      prompt: " B ",
      frames: 131,
      transition: "fade",
      fade_frames: 4,
      seed_offset: "1",
    },
  ]);
  expect(result.request.stages[1]?.negative_prompt).toBeUndefined();
  expect(result.notice).toContain("negative");
  expect(fetch).not.toHaveBeenCalled();
});
it("refuses a linked part and missing effective seeds before media downloads", async () => {
  await expect(
    inspectLegacyVideoSettings(owner, "movie.mp4", {
      ...metadata,
      chain: {
        ...metadata.chain,
        continue_from: { job_id: "old", finalize_take: 1 },
      },
    }),
  ).rejects.toThrow(/linked/i);
  await expect(
    inspectLegacyVideoSettings(owner, "movie.mp4", {
      ...metadata,
      chain: {
        ...metadata.chain,
        stage_count: 1,
        stages: [{ ...metadata.chain.stages[0], seed: null }],
      },
    }),
  ).rejects.toThrow(/seed/);
  expect(fetch).not.toHaveBeenCalled();
});
it("refuses unavailable archives and owner replacement", async () => {
  inventory.mockResolvedValue({
    availability: "unavailable_missing_or_corrupt",
    members: [],
  });
  await expect(
    inspectLegacyVideoSettings(owner, "movie.mp4", metadata),
  ).rejects.toThrow(/archive/i);
  inventory.mockResolvedValue({ availability: "available", members: [] });
  json.mockResolvedValue({ instance_id: "replacement" });
  await expect(
    inspectLegacyVideoSettings(owner, "movie.mp4", metadata),
  ).rejects.toThrow(/installation|changed/i);
});
it("hydrates and verifies ordered typed references and exact keyframe indices", async () => {
  const bytes = new TextEncoder().encode("reference bytes");
  const sha = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
  )
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  const m = {
    ...metadata,
    chain: {
      ...metadata.chain,
      stages: [
        {
          ...metadata.chain.stages[0],
          generation_references: [
            {
              index: 1,
              kind: "image",
              sha256: sha,
              mime_type: "image/png",
              width: 12,
              height: 14,
              crop: {
                x: 1,
                y: 2,
                width: 12,
                height: 14,
                source_width: 30,
                source_height: 40,
                source_sha256: "b".repeat(64),
              },
            },
          ],
          keyframes: [{ frame: 106, sha256: sha, name: "last.png" }],
        },
        metadata.chain.stages[1],
      ],
    },
  };
  const members = [
    {
      member_id: "ref",
      role: "stage_reference:0:0",
      size_bytes: bytes.length,
      display_name: "reference.png",
      sha256: sha,
    },
    {
      member_id: "key",
      role: "stage_keyframe:0:0",
      size_bytes: bytes.length,
      display_name: "last.png",
      sha256: sha,
      frame_index: 106,
    },
  ];
  inventory.mockResolvedValue({ availability: "available", members });
  json.mockImplementation(async (_t, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : { metadata: m, archive_identity_sha256: "a".repeat(64), members },
  );
  fetch.mockImplementation(async () => new Response(bytes));
  const r = await inspectLegacyVideoSettings(owner, "movie.mp4", m);
  expect(r.request.stages[0]?.generation_references?.[0]).toMatchObject({
    kind: "image",
    width: 12,
    height: 14,
    provenance: { sha256: sha, crop: { x: 1, y: 2 } },
    media: { authority: "inline", data: btoa("reference bytes") },
  });
  expect(r.request.stages[0]?.keyframes).toEqual([
    { frame: 106, image: btoa("reference bytes"), name: "last.png" },
  ]);
});
it("checks aggregate budgets before downloading any input and rejects archive changes", async () => {
  inventory.mockResolvedValue({
    availability: "available",
    members: [
      {
        member_id: "big",
        role: "stage_source:0",
        size_bytes: 32 * 1024 * 1024 + 1,
        display_name: "large",
      },
    ],
  });
  await expect(
    inspectLegacyVideoSettings(owner, "movie.mp4", metadata),
  ).rejects.toThrow(/32 MiB/);
  expect(fetch).not.toHaveBeenCalled();
  inventory.mockResolvedValue({ availability: "available", members: [] });
  let offers = 0;
  json.mockImplementation(async (_t, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : {
          metadata,
          archive_identity_sha256: (offers++ ? "b" : "a").repeat(64),
          members: [],
        },
  );
  await expect(
    inspectLegacyVideoSettings(owner, "movie.mp4", metadata),
  ).rejects.toThrow(/changed/);
});
it("keeps a proven automatic legacy video as one prompt with honest saved-length duration", async () => {
  const stages = metadata.chain.stages.map((s) => ({
    ...s,
    prompt: " uniform\\n ",
    frames: 107,
    transition: "smooth",
    fade_frames: null,
    seed: "18446744073709551615",
    loras: [],
  }));
  const m = {
    ...metadata,
    frames: 213,
    negative_prompt: "",
    output_mode: "one-shot",
    chain: { ...metadata.chain, stages },
  };
  json.mockImplementation(async (_t, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : { metadata: m, archive_identity_sha256: "a".repeat(64), members: [] },
  );
  const r = await inspectLegacyVideoSettings(owner, "movie.mp4", m);
  expect(r.automaticDuration).toMatchObject({
    durationMs: "8875",
    scenesEnabled: false,
    scenes: [],
  });
  expect(r.automaticDuration?.reuseNotice).toContain(
    "current automatic joins may differ",
  );
});

it("refuses controls the authored chain cannot represent before any input download", async () => {
  for (const key of [
    "pipeline",
    "scheduler",
    "guidance_overrides",
    "sample_shift",
    "distill_strength_high",
  ])
    await expect(
      inspectLegacyVideoSettings(owner, "movie.mp4", {
        ...metadata,
        [key]: key === "guidance_overrides" ? { dit: 1 } : 1,
      }),
    ).rejects.toThrow(key);
  expect(fetch).not.toHaveBeenCalled();
});
it("requires the archived opening when its source hash was recorded", async () => {
  const m = { ...metadata, source_image_sha256: "b".repeat(64) };
  json.mockImplementation(async (_t, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : { metadata: m, archive_identity_sha256: "a".repeat(64), members: [] },
  );
  await expect(
    inspectLegacyVideoSettings(owner, "movie.mp4", m),
  ).rejects.toThrow(/first-frame/);
  expect(fetch).not.toHaveBeenCalled();
});
it("restores a proven indexed automatic seed pattern but refuses arbitrary overrides", async () => {
  const m = {
    ...metadata,
    frames: 213,
    negative_prompt: "",
    output_mode: "one-shot",
    chain: {
      ...metadata.chain,
      stages: metadata.chain.stages.map((s, i) => ({
        ...s,
        prompt: "same",
        frames: 107,
        transition: "smooth",
        fade_frames: null,
        loras: [],
        seed: String(7n ^ BigInt(i)),
      })),
    },
  };
  json.mockImplementation(async (_t, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : { metadata: m, archive_identity_sha256: "a".repeat(64), members: [] },
  );
  expect(
    (await inspectLegacyVideoSettings(owner, "movie.mp4", m)).automaticDuration
      ?.durationMs,
  ).toBe("8875");
  m.chain.stages[1]!.seed = "9";
  await expect(
    inspectLegacyVideoSettings(owner, "movie.mp4", m),
  ).rejects.toThrow(/single-prompt/);
});

it("restores a complete no-input legacy movie from a fresh authenticated exact Gallery row", async () => {
  inventory.mockResolvedValue({
    availability: "unavailable_legacy",
    members: [],
  });
  json.mockImplementation(async (_t, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : [{ filename: "movie.mp4", metadata }],
  );
  const result = await inspectLegacyVideoSettings(owner, "movie.mp4", metadata);
  expect(result.request.stages[1]?.seed_offset).toBe("1");
  expect(result.request.stages[1]?.prompt).toBe(" B ");
  expect(fetch).not.toHaveBeenCalled();
});

import { reactive } from "vue";
import { beforeEach, expect, it, vi } from "vitest";
const json = vi.hoisted(() => vi.fn());
const ownerCheck = vi.hoisted(() => vi.fn());
const inventoryRead = vi.hoisted(() => vi.fn());
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiJsonTo: json,
}));
vi.mock("./videoJobs", () => ({ assertVideoJobOwner: ownerCheck }));
vi.mock("./gallerySourceMedia", async (original) => ({
  ...(await original<typeof import("./gallerySourceMedia")>()),
  retainedSourceMediaInventory: inventoryRead,
}));
import { inspectVideoLibraryExtension } from "./videoLibraryExtension";
const owner = {
  hostId: "plato",
  instanceId: "installation",
  label: "Plato",
  target: { baseUrl: "http://plato", apiKey: "private-key" },
};
const metadata = {
  model: "video",
  prompt: "literal\\new",
  seed: 42,
  width: 768,
  height: 768,
  frames: 107,
  fps: 24,
  source_image: "old",
  keyframes: [{ frame: 106, image: "old-end" }],
  references: [{ kind: "image", sha256: "a".repeat(64) }],
};
const member = {
  member_id: "member",
  role: "references",
  display_name: "hero.png",
  size_bytes: 50,
  content_sha256: "a".repeat(64),
  media_kind: "image",
};
const offer = () => ({
  archive_identity_sha256: "b".repeat(64),
  output_sha256: "c".repeat(64),
  output_size_bytes: 100,
  metadata,
  members: [
    {
      member_id: "member",
      role: "references",
      position: "item:0",
      sink: "memory",
      size_bytes: 50,
      sha256: "d".repeat(64),
    },
  ],
});
beforeEach(() => {
  vi.clearAllMocks();
  ownerCheck.mockResolvedValue(undefined);
  inventoryRead.mockResolvedValue({
    availability: "available",
    members: [member],
  });
  json.mockResolvedValue(offer());
});
it("binds exact archive metadata and ordered references without downloading pixels", async () => {
  const result = await inspectVideoLibraryExtension(
    owner,
    "original.mp4",
    metadata,
    () => true,
    "Original",
  );
  expect(result.source).toEqual({
    filename: "original.mp4",
    expected_archive_identity_sha256: "b".repeat(64),
  });
  expect(result.metadata.prompt).toBe("literal\\new");
  expect(result.metadata.seed).toBeUndefined();
  expect(result.metadata.source_image).toBeUndefined();
  expect(result.metadata.keyframes).toBeUndefined();
  expect(result.originalReferences.map((r) => r.member?.member_id)).toEqual([
    "member",
  ]);
  expect(metadata.seed).toBe(42);
  expect(metadata.keyframes).toHaveLength(1);
  expect(json.mock.calls.map((c) => c[1])).toEqual([
    "/api/gallery/source-media/original.mp4/transfer",
  ]);
  expect(ownerCheck).toHaveBeenCalledTimes(2);
});
it("accepts only explicit legacy absence, never a generic404 fallback", async () => {
  inventoryRead.mockResolvedValue({
    availability: "unavailable_legacy",
    members: [],
  });
  expect(
    (await inspectVideoLibraryExtension(owner, "old.mp4", metadata)).source,
  ).toEqual({ filename: "old.mp4" });
  expect(json).not.toHaveBeenCalled();
  inventoryRead.mockRejectedValue(new Error("HTTP404"));
  await expect(
    inspectVideoLibraryExtension(owner, "old.mp4", metadata),
  ).rejects.toThrow("404");
});
it.each(["unavailable_auth", "unavailable_missing_or_corrupt", "unknown"])(
  "refuses %s instead of claiming no inputs",
  async (availability) => {
    inventoryRead.mockResolvedValue({ availability, members: [] });
    await expect(
      inspectVideoLibraryExtension(owner, "old.mp4", metadata),
    ).rejects.toThrow();
    expect(json).not.toHaveBeenCalled();
  },
);
it("fences stale owner replies and metadata replacement", async () => {
  let current = true;
  inventoryRead.mockImplementation(async () => {
    current = false;
    return { availability: "available", members: [member] };
  });
  await expect(
    inspectVideoLibraryExtension(owner, "x.mp4", metadata, () => current),
  ).rejects.toThrow(/changed/i);
  expect(json).not.toHaveBeenCalled();
  current = true;
  inventoryRead.mockResolvedValue({
    availability: "available",
    members: [member],
  });
  json.mockResolvedValue({
    ...offer(),
    metadata: { ...metadata, prompt: "changed" },
  });
  await expect(
    inspectVideoLibraryExtension(owner, "x.mp4", metadata),
  ).rejects.toThrow(/changed/i);
});
it("refuses incomplete member inventory and retains only the exact final scene references", async () => {
  json.mockResolvedValue({ ...offer(), members: [] });
  await expect(
    inspectVideoLibraryExtension(owner, "x.mp4", metadata),
  ).rejects.toThrow();
  const last = { kind: "image", sha256: "e".repeat(64) };
  const staged = {
    model: "video",
    prompt: "all scenes joined",
    chain: {
      stages: [
        {
          prompt: "first scene",
          generation_references: [{ kind: "image", sha256: "a".repeat(64) }],
        },
        { generation_references: [last] },
      ],
    },
  };
  const finalMember = {
    ...member,
    role: "stage_reference:1:0",
    content_sha256: last.sha256,
  };
  inventoryRead.mockResolvedValue({
    availability: "available",
    members: [finalMember],
  });
  json.mockResolvedValue({
    ...offer(),
    metadata: staged,
    members: [{ ...offer().members[0], role: finalMember.role }],
  });
  const result = await inspectVideoLibraryExtension(owner, "x.mp4", staged);
  expect(result.metadata.prompt).toBe("first scene");
  expect(result.originalReferences.map((r) => r.descriptor.sha256)).toEqual([
    last.sha256,
  ]);
  expect(result.originalReferences[0]?.member).toEqual(finalMember);
});

it("refuses malformed original reference descriptors rather than hiding them", async () => {
  const broken = { ...metadata, references: [null] };
  json.mockResolvedValue({ ...offer(), metadata: broken });
  await expect(
    inspectVideoLibraryExtension(owner, "x.mp4", broken),
  ).rejects.toThrow();
});

it("restores descriptor visibility only through the exact persisted archive hash", async () => {
  const restored = await inspectVideoLibraryExtension(
    owner,
    "x.mp4",
    {},
    () => true,
    "Original",
    "b".repeat(64),
  );
  expect(restored.originalReferences).toHaveLength(1);
  await expect(
    inspectVideoLibraryExtension(
      owner,
      "x.mp4",
      {},
      () => true,
      "Original",
      "e".repeat(64),
    ),
  ).rejects.toThrow(/changed/i);
  inventoryRead.mockResolvedValue({
    availability: "unavailable_legacy",
    members: [],
  });
  await expect(
    inspectVideoLibraryExtension(
      owner,
      "x.mp4",
      {},
      () => true,
      "Original",
      "b".repeat(64),
    ),
  ).rejects.toThrow(/changed/i);
});

it("keeps original compatible scalar pipeline controls while clearing historical media and seed", async () => {
  const recipe = {
    ...metadata,
    width: 1536,
    height: 1536,
    generation_width: 768,
    generation_height: 768,
    spatial_upscale: "x2",
    true_cfg: 1.7,
    cfg_start_step: 2,
    control_model: "local-control",
    control_scale: 0.5,
    ic_lora_control: "depth",
    temporal_upscale: "x2",
    guidance_overrides: { stg_scale: 0.5 },
    sample_shift: 1.25,
    distill_strength_high: 0.7,
    distill_strength_low: 0.4,
    scheduler: "euler",
    cfg_plus: true,
    video_only: true,
  };
  json.mockResolvedValue({ ...offer(), metadata: recipe });
  const source = await inspectVideoLibraryExtension(
    owner,
    "original.mp4",
    recipe,
  );
  for (const key of [
    "spatial_upscale",
    "true_cfg",
    "cfg_start_step",
    "control_model",
    "control_scale",
    "ic_lora_control",
    "temporal_upscale",
    "guidance_overrides",
    "sample_shift",
    "distill_strength_high",
    "distill_strength_low",
    "scheduler",
    "cfg_plus",
    "video_only",
  ])
    expect(source.metadata[key]).toEqual(recipe[key as keyof typeof recipe]);
  expect(source.metadata.width).toBe(768);
  expect(source.metadata.height).toBe(768);
  expect(source.metadata.seed).toBeUndefined();
  expect(source.metadata.source_image).toBeUndefined();
});

it("accepts a reactive Gallery row without cloning its Vue proxy", async () => {
  const row = reactive({ ...metadata, keyframes: [...metadata.keyframes] });
  const result = await inspectVideoLibraryExtension(owner, "original.mp4", row);
  expect(result.metadata.prompt).toBe(metadata.prompt);
  expect(row.keyframes[0]?.image).toBe("old-end");
});

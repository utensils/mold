import { beforeEach, expect, it, vi } from "vitest";
const json = vi.hoisted(() => vi.fn());
const fetch = vi.hoisted(() => vi.fn());
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiJsonTo: json,
  apiFetchTo: fetch,
}));
import {
  planningVideoRequest,
  validateVideoJob,
  admitVideoJob,
  mutateVideoJob,
  getVideoJob,
} from "./videoJobs";
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "installation",
  target: { baseUrl: "http://plato", apiKey: "private" },
};
const request = {
  duration_ms: "30000",
  generation: {
    model: "opaque",
    prompt: String.raw`literal \n`,
    width: 768,
    height: 768,
    steps: 2,
    fps: 24,
    batch_size: 1,
  },
};
const plan = {
  validation_revision: "proof",
  model: "opaque",
  width: 768,
  height: 768,
  fps: 24,
  single_clip_max_frames: 107,
  overlap_frames: 1,
  join_mode: "continuous" as const,
  clip_count: "7",
  part_count: "1",
  added_frames: "720",
  source_frames: "0",
  total_frames: "720",
  added_duration_ms: "30000",
  total_duration_ms: "30000",
  warnings: [],
};
beforeEach(() => {
  json.mockReset();
  fetch.mockReset();
  fetch.mockImplementation(async () =>
    Response.json({ accepted: true }, { status: 202 }),
  );
  json.mockImplementation(async (_, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : path === "/api/capabilities"
        ? {
            video_jobs: {
              supported: true,
              literal_prompts: true,
              library_extend: true,
            },
          }
        : plan,
  );
});
it("planning sends scalars and the source locator without copying inline media", () => {
  const input = {
    ...request,
    source: {
      filename: "source.mp4",
      expected_archive_identity_sha256: "a".repeat(64),
    },
    generation: {
      ...request.generation,
      source_image: "private image",
      keyframes: [{ frame: 106, image: "private endpoint" }],
      references: [{ data: "private reference" }],
      audio_file: "private audio",
      audio_file_path: "/private/path",
      pipeline: "auto",
      enable_audio: true,
    },
  };
  const projected = planningVideoRequest(input);
  expect(projected).toEqual({
    ...request,
    source: input.source,
    generation: { ...request.generation, pipeline: "auto", enable_audio: true },
  });
  expect(input.generation.source_image).toBe("private image");
});
it("keeps literal prompts and lossless requested duration in full validation", async () => {
  await validateVideoJob(owner, {
    ...request,
    duration_ms: "18446744073709551615",
  });
  const [, path, options] = json.mock.calls.at(-1)!;
  expect(path).toBe("/api/video-jobs/validate");
  expect(options.headers["x-mold-destination-instance"]).toBe("installation");
  expect(JSON.parse(options.body)).toMatchObject({
    duration_ms: "18446744073709551615",
    generation: { prompt: String.raw`literal \n` },
  });
});
it("persists the owner and operation before admitting the validated body and keeps a lost reply explicit", async () => {
  const order: string[] = [];
  json.mockImplementation(async (_, path) => {
    if (path === "/api/status") return { instance_id: "installation" };
    if (path === "/api/capabilities")
      return { video_jobs: { supported: true, literal_prompts: true } };
    order.push("post");
    throw new Error("lost reply");
  });
  let saved: unknown;
  await expect(
    admitVideoJob(owner, request, plan, "known-operation", (record) => {
      order.push("persist");
      saved = record;
    }),
  ).rejects.toThrow("lost reply");
  expect(order).toEqual(["persist", "post"]);
  expect(saved).toMatchObject({
    operationId: "known-operation",
    instanceId: "installation",
    baseUrl: "http://plato",
  });
  expect(JSON.stringify(saved)).not.toContain("private");
  expect(JSON.parse(json.mock.calls.at(-1)![2].body).validation_revision).toBe(
    "proof",
  );
});
it("refuses legacy capabilities, stale owners, display-only plans and unsafe seeds before create", async () => {
  json.mockImplementation(async (_, path) =>
    path === "/api/status" ? { instance_id: "installation" } : {},
  );
  await expect(
    admitVideoJob(owner, request, plan, "known-operation", vi.fn()),
  ).rejects.toThrow("Update");
  expect(json.mock.calls.some(([, path]) => path === "/api/video-jobs")).toBe(
    false,
  );
  json.mockReset();
  await expect(
    admitVideoJob(
      owner,
      request,
      { ...plan, validation_revision: "" },
      "known-operation",
      vi.fn(),
    ),
  ).rejects.toThrow("Validate");
  await expect(
    validateVideoJob(owner, {
      ...request,
      generation: { ...request.generation, seed: 9007199254740992 },
    }),
  ).rejects.toThrow("seed");
  expect(json).not.toHaveBeenCalled();
});
it("cancels an unanswered admission using only its known operation and exact installation", async () => {
  await mutateVideoJob(owner, "known-operation", "cancel");
  expect(fetch).toHaveBeenCalledExactlyOnceWith(
    owner.target,
    "/api/video-jobs/known-operation/cancel",
    {
      method: "POST",
      headers: { "x-mold-destination-instance": "installation" },
    },
  );
  fetch.mockClear();
  await expect(
    mutateVideoJob(owner, "known-operation", "cancel", () => false),
  ).rejects.toThrow("changed");
  expect(fetch).not.toHaveBeenCalled();
});

it("rejects lossy numeric plan counters instead of displaying a rounded clip count", async () => {
  json.mockImplementation(async (_, path) =>
    path === "/api/status"
      ? { instance_id: "installation" }
      : path === "/api/capabilities"
        ? { video_jobs: { supported: true, literal_prompts: true } }
        : { ...plan, clip_count: 9007199254740992 },
  );
  await expect(validateVideoJob(owner, request)).rejects.toThrow("plan");
});

it("passes prepared pixels, fitting, endpoints and typed reference provenance through full validation", async () => {
  const prepared = {
    ...request,
    generation: {
      ...request.generation,
      source_image: "fitted opening bytes",
      source_fit: { mode: "pad-fit" },
      frames: 107,
      keyframes: [{ frame: 106, image: "fitted closing bytes" }],
      references: [
        {
          kind: "image",
          media: { authority: "upload", handle: "bound session" },
          provenance: { crop: { x: 0.2, y: 0, width: 0.5, height: 1 } },
        },
      ],
    },
  };
  await validateVideoJob(owner, prepared);
  expect(JSON.parse(json.mock.calls.at(-1)![2].body)).toEqual(prepared);
});

it("keeps retained-media session authority in memory and reuses it for validation and one create", async () => {
  await validateVideoJob(owner, request, {
    retainedMediaSession: "private-session",
  });
  expect(
    json.mock.calls.at(-1)![2].headers["X-Mold-Retained-Media-Session"],
  ).toBe("private-session");
  let saved: unknown;
  await admitVideoJob(
    owner,
    request,
    plan,
    "known-operation",
    (record) => {
      saved = record;
    },
    () => true,
    { retainedMediaSession: "private-session" },
  );
  expect(
    json.mock.calls.at(-1)![2].headers["X-Mold-Retained-Media-Session"],
  ).toBe("private-session");
  expect(JSON.stringify(saved)).not.toContain("private-session");
});

it.each([{}, { accepted: false }, { accepted: "true" }])(
  "does not acknowledge an unaccepted Stop response %j",
  async (body) => {
    fetch.mockResolvedValueOnce(Response.json(body, { status: 200 }));
    await expect(
      mutateVideoJob(owner, "known-operation", "cancel"),
    ).rejects.toThrow("confirm");
  },
);
it("fences parent reads with the captured installation header", async () => {
  json.mockResolvedValue({ id: "known", state: "queued" });
  await getVideoJob(owner, "known");
  expect(json.mock.calls[0]?.[2]?.headers["x-mold-destination-instance"]).toBe(
    owner.instanceId,
  );
});

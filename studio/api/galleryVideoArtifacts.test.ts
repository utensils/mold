import { beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({ fetch: vi.fn() }));
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiFetchTo: api.fetch,
}));
import {
  transferGalleryVideoArtifacts,
  declaredVideoArtifacts,
  requireVideoArtifactTransfer,
} from "./galleryVideoArtifacts";
const source = {
  id: "source",
  label: "Source",
  instanceId: "s1",
  target: { baseUrl: "https://source", apiKey: "private-source" },
};
const destination = {
  id: "target",
  label: "Target",
  instanceId: "d1",
  target: { baseUrl: "https://target", apiKey: "private-target" },
};
const identity = "a".repeat(64),
  targetIdentity = "b".repeat(64);
const handle = "a".repeat(64);
const artifacts = {
  version: 1,
  bundle_id: "c".repeat(64),
  provenance: { sha256: "d".repeat(64), size_bytes: "3" },
  pcm: {
    file: { sha256: "e".repeat(64), size_bytes: "4194312" },
    sample_rate: 48000,
    channels: 2,
    frames: "524289",
  },
};
const options = () => ({
  source,
  destination,
  sourceFilename: "original.mp4",
  destinationFilename: "copy.mp4",
  sourceArchive: identity,
  destinationArchive: targetIdentity,
  artifacts,
  check: vi.fn(),
});
beforeEach(() => {
  api.fetch.mockReset();
});
it("refuses old source or destination before output import and does not treat a malformed declaration as absent", async () => {
  api.fetch.mockResolvedValue(
    new Response(
      JSON.stringify({ retained_media_transfer: { video_artifacts: false } }),
    ),
  );
  await expect(
    requireVideoArtifactTransfer(source, destination, () => {}),
  ).rejects.toThrow("Update");
  expect(() =>
    declaredVideoArtifacts({ video_job: { artifacts: null } }),
  ).toThrow("saved data");
  expect(declaredVideoArtifacts({ video_job: {} })).toBeUndefined();
});
it("streams PCM independently in bounded chunks, resuming persisted offsets without whole-file buffering", async () => {
  let offsets = { provenance: "3", pcm: "4" };
  const writes: number[] = [];
  api.fetch.mockImplementation(async (target, path, init) => {
    const headers = new Headers(init?.headers);
    if (target === source.target) {
      expect(headers.get("x-mold-source-instance")).toBe("s1");
      expect(headers.get("x-mold-gallery-archive-identity")).toBe(identity);
      const offset = BigInt(
        new URL(`https://x${path}`).searchParams.get("offset_bytes")!,
      );
      expect(path).toContain("/pcm?");
      return new Response(
        new Uint8Array(
          Number(4194312n - offset > 4194304n ? 4194304n : 4194312n - offset),
        ),
      );
    }
    expect(headers.get("x-mold-destination-instance")).toBe("d1");
    expect(headers.get("x-mold-gallery-archive-identity")).toBe(targetIdentity);
    if (init?.method === "PUT") {
      const bytes = new Uint8Array(await new Response(init.body).arrayBuffer());
      writes.push(bytes.length);
      offsets = {
        ...offsets,
        pcm: String(BigInt(offsets.pcm) + BigInt(bytes.length)),
      };
    }
    return new Response(
      JSON.stringify({
        handle,
        archive_identity_sha256: targetIdentity,
        offsets,
        complete: path.endsWith("/complete"),
      }),
    );
  });
  await transferGalleryVideoArtifacts(options());
  expect(writes).toEqual([4194304, 4]);
  expect(api.fetch.mock.calls.some(([, p]) => p.includes("/provenance?"))).toBe(
    false,
  );
});
it("retains incomplete work after lost chunk acknowledgement and resumes from server state on retry", async () => {
  let offset = "0",
    lost = true;
  api.fetch.mockImplementation(async (target, path, init) => {
    if (target === source.target) return new Response(new Uint8Array(3));
    if (init?.method === "PUT") {
      offset = "3";
      if (lost) {
        lost = false;
        throw new Error("connection lost");
      }
    }
    return new Response(
      JSON.stringify({
        handle,
        archive_identity_sha256: targetIdentity,
        offsets: { provenance: offset },
        complete: path.endsWith("/complete"),
      }),
    );
  });
  const opts = {
    ...options(),
    artifacts: {
      version: artifacts.version,
      bundle_id: artifacts.bundle_id,
      provenance: artifacts.provenance,
    },
  };
  await expect(transferGalleryVideoArtifacts(opts)).rejects.toThrow(
    "connection lost",
  );
  await transferGalleryVideoArtifacts(opts);
  expect(
    api.fetch.mock.calls.filter(
      ([, p]) => p.includes("/provenance?") && !p.includes("/transfer/"),
    ),
  ).toHaveLength(1);
});
it("refuses changed destination authority and dishonest completion offsets", async () => {
  api.fetch.mockResolvedValue(
    new Response(
      JSON.stringify({
        handle,
        archive_identity_sha256: "f".repeat(64),
        offsets: { provenance: "3", pcm: "4194312" },
        complete: true,
      }),
    ),
  );
  await expect(transferGalleryVideoArtifacts(options())).rejects.toThrow(
    "changed",
  );
  api.fetch.mockImplementation(
    async () =>
      new Response(
        JSON.stringify({
          handle,
          archive_identity_sha256: targetIdentity,
          offsets: { provenance: "3", pcm: "0" },
          complete: true,
        }),
      ),
  );
  await expect(transferGalleryVideoArtifacts(options())).rejects.toThrow(
    "incomplete",
  );
});
it("checks cancellation and owner changes between source reads and destination writes", async () => {
  const opts = options();
  let reads = 0;
  opts.check.mockImplementation(() => {
    if (reads) throw new Error("owner changed");
  });
  api.fetch.mockImplementation(async (target) =>
    target === source.target
      ? (reads++, new Response(new Uint8Array(3)))
      : new Response(
          JSON.stringify({
            handle,
            archive_identity_sha256: targetIdentity,
            offsets: { provenance: "0" },
            complete: false,
          }),
        ),
  );
  await expect(
    transferGalleryVideoArtifacts({
      ...opts,
      artifacts: {
        version: artifacts.version,
        bundle_id: artifacts.bundle_id,
        provenance: artifacts.provenance,
      },
    }),
  ).rejects.toThrow("owner changed");
  expect(
    api.fetch.mock.calls.some(([, , init]) => init?.method === "PUT"),
  ).toBe(false);
});
it("keeps artifact offsets exact above the JavaScript safe integer range without a total-size cap", async () => {
  const size = "9007199254741000",
    start = "9007199254740992";
  const large = {
    ...artifacts,
    pcm: {
      file: { sha256: "e".repeat(64), size_bytes: size },
      sample_rate: 48000,
      channels: 2,
      frames: "1125899906842625",
    },
  };
  let offset = start;
  api.fetch.mockImplementation(async (target, path, init) => {
    if (target === source.target) {
      expect(path).toContain(`offset_bytes=${start}`);
      return new Response(new Uint8Array(8));
    }
    if (init?.method === "PUT") offset = size;
    return new Response(
      JSON.stringify({
        handle,
        archive_identity_sha256: targetIdentity,
        offsets: { provenance: "3", pcm: offset },
        complete: path.endsWith("/complete"),
      }),
    );
  });
  await transferGalleryVideoArtifacts({ ...options(), artifacts: large });
  expect(declaredVideoArtifacts({ video_job: { artifacts: large } })).toEqual(
    large,
  );
});
it("refuses unexpected roles and changes to unrelated role progress", async () => {
  api.fetch.mockResolvedValue(
    new Response(
      JSON.stringify({
        handle,
        archive_identity_sha256: targetIdentity,
        offsets: { provenance: "3", pcm: "0" },
        complete: false,
      }),
    ),
  );
  await expect(
    transferGalleryVideoArtifacts({
      ...options(),
      artifacts: {
        version: 1,
        bundle_id: artifacts.bundle_id,
        provenance: artifacts.provenance,
      },
    }),
  ).rejects.toThrow("unexpected");
  api.fetch.mockImplementation(async (target, _path, init) =>
    target === source.target
      ? new Response(new Uint8Array(4194304))
      : new Response(
          JSON.stringify({
            handle,
            archive_identity_sha256: targetIdentity,
            offsets: {
              provenance: init?.method === "PUT" ? "0" : "3",
              pcm: init?.method === "PUT" ? "4194304" : "0",
            },
            complete: false,
          }),
        ),
  );
  await expect(transferGalleryVideoArtifacts(options())).rejects.toThrow(
    "unexpectedly",
  );
});

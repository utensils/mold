import { beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({ fetch: vi.fn() }));
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiFetchTo: api.fetch,
}));
vi.mock("./galleryMirrorOrganization", () => ({
  captureMirrorOrganization: vi.fn(async () => ({ item: {}, collections: [] })),
  applyMirrorOrganization: vi.fn(async () => undefined),
}));
import {
  copyGalleryPrint,
  rawJsonProperty,
  copyMachineIdentity,
  type CopyJournal,
  type CopyReceipt,
} from "./galleryCopy";
import { ApiError } from "./client";
import {
  captureMirrorOrganization,
  applyMirrorOrganization,
} from "./galleryMirrorOrganization";
const source = {
  id: "source",
  label: "Source",
  instanceId: "s1",
  target: { baseUrl: "https://source", apiKey: "s" },
};
const destination = {
  id: "destination",
  label: "Destination",
  instanceId: "d1",
  target: { baseUrl: "https://destination", apiKey: "d" },
};
const journal: CopyJournal = {
  read: vi.fn(async () => null),
  write: vi.fn(async () => undefined),
};
beforeEach(() => {
  api.fetch.mockReset();
  vi.mocked(journal.read).mockResolvedValue(null);
  vi.mocked(journal.write).mockClear();
  vi.mocked(captureMirrorOrganization).mockResolvedValue({
    item: {},
    collections: [],
  });
  vi.mocked(applyMirrorOrganization).mockClear();
  vi.mocked(applyMirrorOrganization).mockResolvedValue(undefined);
});
it("preserves exact 64-bit recipe integers in the import descriptor", () => {
  expect(
    rawJsonProperty(
      '{"metadata":{"seed":18446744073709551615,"prompt":"}\\\"{"},"members":[]}',
      "metadata",
    ),
  ).toBe('{"seed":18446744073709551615,"prompt":"}\\\"{"}');
});
it("refuses unsupported retained-media receivers before importing any output", async () => {
  const offer = {
    archive_identity_sha256: "a".repeat(64),
    output_sha256: "b".repeat(64),
    output_size_bytes: 3,
    metadata: { seed: 1 },
    members: [
      {
        member_id: "member",
        role: "source_image",
        position: "scalar",
        sink: "memory",
        size_bytes: 1,
        sha256: "c".repeat(64),
      },
    ],
  };
  api.fetch.mockImplementation(
    async (target, path) =>
      new Response(
        JSON.stringify(
          path === "/api/status"
            ? { instance_id: target === source.target ? "s1" : "d1" }
            : path === "/api/capabilities"
              ? {}
              : offer,
        ),
      ),
  );
  await expect(
    copyGalleryPrint({
      source,
      destination,
      filename: "print.png",
      current: () => true,
      journal,
    }),
  ).rejects.toThrow("retain");
  expect(
    api.fetch.mock.calls.some(([, path]) => path.includes("/import/")),
  ).toBe(false);
});
it("refuses a changed authority before any mutation", async () => {
  await expect(
    copyGalleryPrint({
      source,
      destination,
      filename: "print.png",
      current: () => false,
      journal,
    }),
  ).rejects.toThrow("changed");
  expect(api.fetch).not.toHaveBeenCalled();
});
it("imports exact bytes with an output receipt and confirms both installations before completion", async () => {
  const bytes = new TextEncoder().encode("abc");
  const digest = [
    ...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
  ]
    .map((n) => n.toString(16).padStart(2, "0"))
    .join("");
  const offerText = `{"archive_identity_sha256":"${"a".repeat(64)}","output_sha256":"${digest}","output_size_bytes":3,"metadata":{"seed":18446744073709551615},"members":[]}`;
  let envelope: Blob | undefined;
  let imported = false;
  api.fetch.mockImplementation(async (target, path, init) => {
    if (path === "/api/status")
      return new Response(
        JSON.stringify({ instance_id: target === source.target ? "s1" : "d1" }),
      );
    if (path.includes("/import/")) {
      imported = true;
      envelope = init.body;
      return new Response('{"filename":"print.png"}');
    }
    if (
      target === destination.target &&
      path.endsWith("/transfer") &&
      !imported
    )
      throw new ApiError("not found", 404);
    if (path.includes("/image/")) return new Response(bytes);
    return new Response(offerText);
  });
  const receipt = await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal,
  });
  expect(receipt.stage).toBe("complete");
  expect(journal.write).toHaveBeenCalledWith(
    expect.any(String),
    expect.objectContaining({
      stage: "output",
      destinationFilename: "print.png",
    }),
  );
  const body = new Uint8Array(await envelope!.arrayBuffer());
  const length = new DataView(body.buffer).getUint32(0);
  expect(new TextDecoder().decode(body.slice(12, 12 + length))).toContain(
    "18446744073709551615",
  );
  expect([...body.slice(12 + length)]).toEqual([...bytes]);
});

async function existingFixture(failure?: Error) {
  const digest = [
    ...new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode("abc")),
    ),
  ]
    .map((n) => n.toString(16).padStart(2, "0"))
    .join("");
  const metadata = '{"seed":1}';
  const recipe = [
    ...new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(metadata)),
    ),
  ]
    .map((n) => n.toString(16).padStart(2, "0"))
    .join("");
  const receipt: CopyReceipt = {
    version: 1,
    source: copyMachineIdentity(source),
    destination: copyMachineIdentity(destination),
    sourceFilename: "print.png",
    destinationFilename: "copy.png",
    digest,
    size: 3,
    recipe,
    stage: "complete",
  };
  const offer = {
    archive_identity_sha256: "a".repeat(64),
    output_sha256: digest,
    output_size_bytes: 3,
    metadata: { seed: 1 },
    members: [],
  };
  vi.mocked(journal.read).mockResolvedValue(receipt);
  api.fetch.mockImplementation(async (target, path) => {
    if (path === "/api/status")
      return new Response(
        JSON.stringify({ instance_id: target === source.target ? "s1" : "d1" }),
      );
    if (path === "/api/gallery" || path === "/api/gallery?view=trash")
      return new Response("[]");
    if (target === destination.target && path.includes("copy.png") && failure)
      throw failure;
    return new Response(JSON.stringify(offer));
  });
  return { receipt, offer };
}
it.each([401, 500])(
  "does not turn destination HTTP %s into an intentionally removed copy",
  async (status) => {
    await existingFixture(new ApiError("failure", status));
    await expect(
      copyGalleryPrint({
        source,
        destination,
        filename: "print.png",
        current: () => true,
        journal,
        deliberate: false,
      }),
    ).rejects.toThrow("failure");
  },
);
it("suppresses automatic recreation only after authoritative missing-output inventories", async () => {
  const { receipt } = await existingFixture(new ApiError("gone", 404));
  expect(
    await copyGalleryPrint({
      source,
      destination,
      filename: "print.png",
      current: () => true,
      journal,
      deliberate: false,
    }),
  ).toEqual(receipt);
  expect(
    api.fetch.mock.calls.some(([, path]) => path === "/api/gallery?view=trash"),
  ).toBe(true);
});
it("does not inherit another contributor organization when matching destination bytes", async () => {
  const { receipt } = await existingFixture();
  vi.mocked(journal.read).mockResolvedValue(null);
  const matching: CopyJournal = {
    ...journal,
    matching: async () => ({
      ...receipt,
      source: "independent",
      organization: {
        version: 1,
        observedTags: ["mine"],
        ownedTags: ["mine"],
        observedCollections: [],
        ownedCollections: [],
        suppressedTags: [],
        suppressedCollections: [],
        title: "Their title",
      },
    }),
  };
  const result = await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal: matching,
  });
  expect(result.organization).toBeUndefined();
});
it("retains narrowly verified legacy missing-input outcomes", async () => {
  const { offer } = await existingFixture();
  vi.mocked(journal.read).mockResolvedValue(null);
  api.fetch.mockImplementation(async (target, path) => {
    if (path === "/api/status")
      return new Response(
        JSON.stringify({ instance_id: target === source.target ? "s1" : "d1" }),
      );
    if (path.endsWith("/print.png"))
      return new Response('{"availability":"unavailable_legacy"}');
    return new Response(
      JSON.stringify({
        ...offer,
        metadata: { seed: 1, source_image_sha256: "c".repeat(64) },
      }),
    );
  });
  const result = await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal,
  });
  expect(result.outcome).toBe("legacy_inputs_unavailable");
});
it.each([null, {}, ""])(
  "refuses malformed legacy inventories with members=%j",
  async (members) => {
    const { offer } = await existingFixture();
    api.fetch.mockImplementation(async (target, path) => {
      if (path === "/api/status")
        return new Response(
          JSON.stringify({
            instance_id: target === source.target ? "s1" : "d1",
          }),
        );
      if (path.endsWith("/print.png"))
        return new Response(
          JSON.stringify({ availability: "unavailable_legacy", members }),
        );
      return new Response(
        JSON.stringify({
          ...offer,
          metadata: { seed: 1, source_image_sha256: "c".repeat(64) },
        }),
      );
    });
    await expect(
      copyGalleryPrint({
        source,
        destination,
        filename: "print.png",
        current: () => true,
        journal,
      }),
    ).rejects.toThrow("original source media could not be verified");
  },
);
it.each([null, {}, ""])(
  "refuses malformed older-host inventories with members=%j",
  async (members) => {
    const wire = await import("./galleryCopyWire");
    const nativeLegacyOffer = vi.fn(async () => ({
      archive_identity_sha256: "a".repeat(64),
      output_sha256: "b".repeat(64),
      output_size_bytes: 3,
      metadata: { seed: 1 },
      rawMetadata: '{"seed":1}',
      members: [],
    }));
    api.fetch.mockImplementation(async (_target, path) => {
      if (path.endsWith("/transfer")) throw new ApiError("old endpoint", 405);
      if (path === "/api/gallery")
        return new Response('[{"filename":"print.png","metadata":{"seed":1}}]');
      return new Response(
        JSON.stringify({ availability: "unavailable_legacy", members }),
      );
    });
    await expect(
      wire.offer(
        source,
        "print.png",
        undefined,
        wire.MAX_BYTES,
        nativeLegacyOffer,
      ),
    ).rejects.toThrow("old endpoint");
    expect(nativeLegacyOffer).not.toHaveBeenCalled();
  },
);
it("preserves original organization through A to B to C to A without echoing copied local edits", async () => {
  const { offer } = await existingFixture();
  vi.mocked(journal.read).mockResolvedValue(null);
  const { copyJournal } = await import("../lib/galleryCopyJournal");
  const { reconcileMirrorOrganization } =
    await import("../lib/mirrorOrganizationPolicy");
  const mirror = await import("./galleryMirrorOrganization");
  const third = {
    id: "third",
    label: "Third",
    instanceId: "c1",
    target: { baseUrl: "https://third", apiKey: "c" },
  };
  let document: any = null;
  const storage = {
    read: async () => (document ? structuredClone(document) : null),
    write: async (next: any) => {
      document = structuredClone(next);
    },
  };
  const state = new Map<string, any>([
    [
      source.target.baseUrl,
      {
        tags: ["Original"],
        title: "Original title",
        favorite: true,
        collections: [],
      },
    ],
    [
      destination.target.baseUrl,
      { tags: [], title: null, favorite: false, collections: [] },
    ],
    [
      third.target.baseUrl,
      { tags: [], title: null, favorite: false, collections: [] },
    ],
  ]);
  vi.mocked(mirror.captureMirrorOrganization).mockImplementation(
    async (target) => ({
      item: structuredClone(state.get(target.baseUrl)),
      collections: [],
    }),
  );
  vi.mocked(mirror.applyMirrorOrganization).mockImplementation(
    async (target, _name, incoming, reconciliation) => {
      const result = reconcileMirrorOrganization(
        {
          ...incoming.item,
          tags: incoming.item.tags ?? [],
          title: incoming.item.title ?? null,
          favorite: incoming.item.favorite === true,
          collections: [],
          hidden: [],
        },
        state.get(target.baseUrl),
        reconciliation?.prior,
        reconciliation,
      );
      await reconciliation!.intent({
        state: result.state,
        desired: result.desired,
        baseline: state.get(target.baseUrl),
      });
      state.set(target.baseUrl, result.desired);
      return result.state;
    },
  );
  api.fetch.mockImplementation(async (target, path) => {
    if (path === "/api/status")
      return new Response(
        JSON.stringify({
          instance_id:
            target.baseUrl === source.target.baseUrl
              ? "s1"
              : target.baseUrl === destination.target.baseUrl
                ? "d1"
                : "c1",
        }),
      );
    return new Response(JSON.stringify(offer));
  });
  const graph = copyJournal(storage);
  const copy = (from: typeof source, to: typeof source) =>
    copyGalleryPrint({
      source: from,
      destination: to,
      filename: "print.png",
      current: () => true,
      journal: graph,
      deliberate: false,
    });
  const aToB = await copy(source, destination);
  state.get(destination.target.baseUrl).title = "Local edit on B";
  const bToC = await copy(destination, third);
  expect(bToC.origin).toBe(aToB.origin);
  expect(state.get(third.target.baseUrl).title).toBe("Original title");
  state.get(source.target.baseUrl).tags = [];
  state.get(source.target.baseUrl).title = "Original changed";
  await copy(third, source);
  expect(state.get(source.target.baseUrl).title).toBe("Original changed");
  expect(state.get(source.target.baseUrl).tags).toEqual([]);
  expect(Object.keys(document.receipts)).toHaveLength(3);
  await copy(source, destination);
  expect(state.get(destination.target.baseUrl).title).toBe("Local edit on B");
  await copy(destination, third);
  expect(state.get(third.target.baseUrl).title).toBe("Original changed");
});

it("copies an older source only after verifying its no-retained-media inventory and exact bytes", async () => {
  const bytes = new TextEncoder().encode("abc");
  const digest = [
    ...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
  ]
    .map((n) => n.toString(16).padStart(2, "0"))
    .join("");
  const metadata = '{"seed":18446744073709551615}';
  const destinationOffer = `{"archive_identity_sha256":"${"a".repeat(64)}","output_sha256":"${digest}","output_size_bytes":3,"metadata":${metadata},"members":[]}`;
  let imported = false,
    descriptor = "";
  vi.mocked(captureMirrorOrganization).mockResolvedValue({
    item: { metadata_synthetic: true },
    collections: [],
  });
  api.fetch.mockImplementation(async (target, path, init) => {
    if (path === "/api/status")
      return new Response(
        JSON.stringify({ instance_id: target === source.target ? "s1" : "d1" }),
      );
    if (path.endsWith("/transfer")) {
      if (target === source.target) throw new ApiError("old endpoint", 405);
      if (!imported) throw new ApiError("missing output", 404);
      return new Response(destinationOffer);
    }
    if (path === "/api/gallery/source-media/print.png") {
      if (target === destination.target)
        throw new ApiError("missing output", 404);
      return new Response('{"availability":"unavailable_legacy"}');
    }
    if (path === "/api/gallery")
      return new Response(`[{"filename":"print.png","metadata":${metadata}}]`);
    if (path.includes("/image/")) return new Response(bytes);
    if (path.includes("/import/")) {
      const body = new Uint8Array(await (init.body as Blob).arrayBuffer());
      descriptor = new TextDecoder().decode(
        body.slice(12, 12 + new DataView(body.buffer).getUint32(0)),
      );
      imported = true;
      return new Response('{"filename":"print.png"}');
    }
    throw new Error("unexpected " + path);
  });
  const result = await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal,
  });
  expect(result.stage).toBe("complete");
  expect(descriptor).toContain("18446744073709551615");
  expect(descriptor).toContain('"metadata_synthetic":true');
});
it("does not use old-source fallback for authentication failures", async () => {
  api.fetch.mockImplementation(async (_target, path) => {
    if (path === "/api/status") return new Response('{"instance_id":"s1"}');
    throw new ApiError("unauthorized", 401);
  });
  await expect(
    copyGalleryPrint({
      source,
      destination: { ...destination, instanceId: "s1" },
      filename: "print.png",
      current: () => true,
      journal,
    }),
  ).rejects.toThrow("unauthorized");
  expect(api.fetch.mock.calls.some(([, path]) => path === "/api/gallery")).toBe(
    false,
  );
});
it("streams native output and retained media with a durable receipt between the callbacks", async () => {
  const large = 200 * 1024 * 1024;
  const member = {
    member_id: "video",
    role: "source_video",
    position: "scalar",
    sink: "private_staging",
    size_bytes: large,
    sha256: "c".repeat(64),
  };
  const sourceOffer = {
    archive_identity_sha256: "a".repeat(64),
    output_sha256: "b".repeat(64),
    output_size_bytes: large,
    metadata: { seed: 1 },
    members: [member],
  };
  let imported = false,
    transferred = false;
  api.fetch.mockImplementation(async (target, path) => {
    if (path === "/api/status")
      return new Response(
        JSON.stringify({ instance_id: target === source.target ? "s1" : "d1" }),
      );
    if (path === "/api/capabilities")
      return new Response('{"retained_media_transfer":{"protocol_version":1}}');
    if (target === destination.target && !imported)
      throw new ApiError("missing output", 404);
    return new Response(
      JSON.stringify({
        ...sourceOffer,
        members: target === destination.target && !transferred ? [] : [member],
      }),
    );
  });
  const nativeImportOutput = vi.fn(async () => {
    imported = true;
    return "print.png";
  });
  const nativeTransferSources = vi.fn(async () => {
    expect(journal.write).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        stage: "output",
        destinationFilename: "print.png",
      }),
    );
    transferred = true;
  });
  const nativeMatchesOutput = vi.fn(async () => true);
  const result = await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal,
    nativeImportOutput,
    nativeTransferSources,
    nativeMatchesOutput,
  });
  expect(result.stage).toBe("complete");
  expect(nativeImportOutput).toHaveBeenCalledOnce();
  expect(nativeTransferSources).toHaveBeenCalledOnce();
  expect(
    api.fetch.mock.calls.some(
      ([, path]) => path.includes("/image/") || path.endsWith("/video"),
    ),
  ).toBe(false);
});
it("does not recreate an intentionally removed graph copy for a new source edge", async () => {
  const { receipt } = await existingFixture(new ApiError("gone", 404));
  vi.mocked(journal.read).mockResolvedValue(null);
  const graph: CopyJournal = {
    ...journal,
    matching: async () => ({ ...receipt, source: "another verified origin" }),
  };
  const result = await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal: graph,
    deliberate: false,
  });
  expect(result.destinationFilename).toBe("copy.png");
  expect(result.organization).toBeUndefined();
  expect(result.stage).toBe("complete");
  expect(
    api.fetch.mock.calls.some(([, path]) => path.includes("/import/")),
  ).toBe(false);
});

it("stops organization writes when canceled even if machine authority stays current", async () => {
  await existingFixture();
  const controller = new AbortController();
  vi.mocked(applyMirrorOrganization).mockImplementation(
    async (_target, _filename, _snapshot, reconciliation) => {
      controller.abort();
      expect(reconciliation?.current?.()).toBe(false);
      throw new DOMException("Canceled", "AbortError");
    },
  );
  await expect(
    copyGalleryPrint({
      source,
      destination,
      filename: "print.png",
      current: () => true,
      journal,
      signal: controller.signal,
    }),
  ).rejects.toThrow("Canceled");
});

it("persists the verified destination recipe separately from compatible source archive facts", async () => {
  const { offer, receipt } = await existingFixture();
  api.fetch.mockImplementation(async (target, path) => {
    if (path === "/api/status")
      return new Response(
        JSON.stringify({ instance_id: target === source.target ? "s1" : "d1" }),
      );
    return new Response(
      JSON.stringify({
        ...offer,
        metadata:
          target === source.target
            ? { seed: 1, generation_time_ms: 10 }
            : { seed: 1 },
      }),
    );
  });
  const result = await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal,
  });
  expect(result.destinationRecipe).toBe(receipt.recipe);
  expect(result.destinationRecipe).not.toBe(result.recipe);
  expect(
    vi
      .mocked(journal.write)
      .mock.calls.some(
        ([, row]) =>
          row.stage === "output" && row.destinationRecipe === receipt.recipe,
      ),
  ).toBe(true);
});

async function artifactCopyFixture(
  completeAcknowledgement = true,
  targetSupported = true,
) {
  const { receipt, offer: original } = await existingFixture();
  const artifacts = {
    version: 1,
    bundle_id: "c".repeat(64),
    provenance: { sha256: "d".repeat(64), size_bytes: "3" },
  };
  const portable = {
    ...original,
    metadata: { seed: 1, video_job: { artifacts } },
  };
  let complete = false;
  const offsets = { provenance: "0" };
  api.fetch.mockImplementation(async (target, path, init) => {
    if (path === "/api/status")
      return new Response(
        JSON.stringify({ instance_id: target === source.target ? "s1" : "d1" }),
      );
    if (path === "/api/capabilities")
      return new Response(
        JSON.stringify({
          retained_media_transfer: {
            protocol_version: 1,
            video_artifacts: target === source.target || targetSupported,
          },
        }),
      );
    if (path.includes("/video-artifacts/")) {
      if (target === source.target) return new Response(new Uint8Array(3));
      if (init?.method === "PUT") offsets.provenance = "3";
      if (path.endsWith("/complete")) complete = completeAcknowledgement;
      return new Response(
        JSON.stringify({
          handle: "a".repeat(64),
          archive_identity_sha256: portable.archive_identity_sha256,
          offsets,
          complete,
        }),
      );
    }
    return new Response(
      JSON.stringify({
        ...portable,
        video_artifacts_complete: target === source.target || complete,
      }),
    );
  });
  return { receipt, artifacts };
}
it("repairs missing video artifacts even when an equal output and complete receipt already exist", async () => {
  await artifactCopyFixture();
  const result = await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal,
  });
  expect(
    api.fetch.mock.calls.some(
      ([, path, init]) =>
        path.includes("/video-artifacts/") && init?.method === "PUT",
    ),
  ).toBe(true);
  expect(result.stage).toBe("complete");
  expect(
    api.fetch.mock.calls.some(([, path]) => path.includes("/import/")),
  ).toBe(false);
});
it("does not publish a complete receipt before durable video artifact completion", async () => {
  await artifactCopyFixture(false);
  await expect(
    copyGalleryPrint({
      source,
      destination,
      filename: "print.png",
      current: () => true,
      journal,
    }),
  ).rejects.toThrow("Saving all video data");
  expect(
    vi
      .mocked(journal.write)
      .mock.calls.some(([, receipt]) => receipt.stage === "complete"),
  ).toBe(false);
});
it("refuses old artifact receivers before any output import or organization mutation", async () => {
  await artifactCopyFixture(true, false);
  vi.mocked(journal.read).mockResolvedValue(null);
  await expect(
    copyGalleryPrint({
      source,
      destination,
      filename: "print.png",
      current: () => true,
      journal,
    }),
  ).rejects.toThrow("Update");
  expect(
    api.fetch.mock.calls.some(([, path]) => path.includes("/import/")),
  ).toBe(false);
  expect(applyMirrorOrganization).not.toHaveBeenCalled();
});
it("imports authenticated raw video recipe bytes without rounding u64 or normalizing literal prompts", async () => {
  await artifactCopyFixture();
  vi.mocked(journal.read).mockResolvedValue(null);
  const fallback = api.fetch.getMockImplementation()!;
  let imported = false;
  let body: Blob | undefined;
  const rawRecipe =
    '{"seed":18446744073709551615,"prompt":"literal\\\\n\\r\\nnext","negative_prompt":"literal\\\\t","video_job":{"artifacts":{"version":1,"bundle_id":"' +
    "c".repeat(64) +
    '","provenance":{"sha256":"' +
    "d".repeat(64) +
    '","size_bytes":"3"}}}}';
  api.fetch.mockImplementation(async (target, path, init) => {
    if (path.includes("/import/")) {
      expect(new Headers(init.headers).get("x-mold-destination-instance")).toBe(
        "d1",
      );
      imported = true;
      body = init.body;
      return new Response('{"filename":"print.png"}');
    }
    if (path.includes("/image/"))
      return new Response(new TextEncoder().encode("abc"));
    if (
      target === destination.target &&
      path.includes("/source-media/") &&
      !imported
    )
      throw new ApiError("missing", 404);
    const response = await fallback(target, path, init);
    if (path.includes("/source-media/") && path.endsWith("/transfer")) {
      const value = await response.json();
      const text = JSON.stringify(value);
      return new Response(
        text.replace(
          '"metadata":' + rawJsonProperty(text, "metadata"),
          '"metadata":' + rawRecipe,
        ),
      );
    }
    return response;
  });
  await copyGalleryPrint({
    source,
    destination,
    filename: "print.png",
    current: () => true,
    journal,
  });
  const bytes = new Uint8Array(await body!.arrayBuffer());
  const size = new DataView(bytes.buffer).getUint32(0);
  expect(
    rawJsonProperty(
      new TextDecoder().decode(bytes.slice(12, 12 + size)),
      "metadata",
    ),
  ).toBe(rawRecipe);
});

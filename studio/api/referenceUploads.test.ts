import { afterEach, describe, expect, it, vi } from "vitest";
import type { GenerationReference } from "../lib/generationReferences";
import {
  prepareReferenceUploads,
  requestNeedsReferenceUpload,
  type ReferenceUploadCapabilities,
  type ReferenceUploadRequest,
} from "./referenceUploads";

const TARGET = {
  baseUrl: "https://mold.example",
  apiKey: "durable-api-key",
};
const INSTANCE_ID = "instance-exact";
const SCOPE_DIGEST = "a".repeat(64);
const REBOUND_SCOPE_DIGEST = "b".repeat(64);
const CAPABILITIES: ReferenceUploadCapabilities = {
  available: true,
  protocol_version: 2,
  requires_api_key: true,
  session_path: "/api/generate/reference-upload-sessions",
  upload_path: "/api/generate/reference-upload",
  session_handle_header: "x-mold-reference-upload-session",
  upload_handle_header: "x-mold-reference-upload",
  max_file_bytes: 1024 * 1024,
  max_session_bytes: 4 * 1024 * 1024,
  session_ttl_ms: 30_000,
};

function base64(value: string): string {
  return btoa(value);
}

async function digest(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const result = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(result)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function requestFixture(): Promise<ReferenceUploadRequest> {
  const image = "image-one";
  const audio = "audio-two";
  return {
    model: "minimax-h3-ref2va",
    prompt: "A synchronized new shot",
    frames: 124,
    fps: 24,
    guidance: 0,
    batch_size: 1,
    references: [
      {
        kind: "image",
        media: { authority: "inline", data: base64(image) },
        provenance: { name: "identity.png", sha256: await digest(image) },
        mime_type: "image/png",
        width: 32,
        height: 24,
      },
      {
        kind: "audio",
        media: { authority: "inline", data: base64(audio) },
        provenance: { name: "timing.wav", sha256: await digest(audio) },
        mime_type: "audio/wav",
        duration_ms: 2_000,
        sample_rate: 24_000,
        channels: 1,
        sample_count: 48_000,
      },
    ],
  };
}

function metadataFor(
  reference: GenerationReference,
  index: number,
): Record<string, unknown> {
  const base = {
    kind: reference.kind,
    index,
    name: reference.provenance?.name,
    sha256: reference.provenance?.sha256,
    mime_type: reference.mime_type,
  };
  if (reference.kind === "image") {
    return { ...base, width: reference.width, height: reference.height };
  }
  if (reference.kind === "audio") {
    return {
      ...base,
      duration_ms: reference.duration_ms,
      sample_rate: reference.sample_rate,
      channels: reference.channels,
      sample_count: reference.sample_count,
    };
  }
  return {
    ...base,
    width: reference.width,
    height: reference.height,
    frame_count: reference.frame_count,
    duration_ms: reference.duration_ms,
    fps: reference.fps,
    has_audio: reference.has_audio,
    audio_duration_ms: reference.audio_duration_ms,
    audio_sample_count: reference.audio_sample_count,
    audio_sample_rate: reference.audio_sample_rate,
    audio_channels: reference.audio_channels,
  };
}

function uploadComplete(
  reference: GenerationReference,
  index: number,
  sessionComplete: boolean,
  requestScopeSha256 = REBOUND_SCOPE_DIGEST,
): Record<string, unknown> {
  return {
    instance_id: INSTANCE_ID,
    reference: index,
    metadata: metadataFor(reference, index),
    request_scope_sha256: requestScopeSha256,
    session_complete: sessionComplete,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("MiniMax H3 reference upload leases", () => {
  it("uploads exact bytes through stable URLs and header-only one-use authorities", async () => {
    const request = await requestFixture();
    const calls: Array<{
      url: string;
      method: string;
      headers: Headers;
      body: BodyInit | null | undefined;
    }> = [];
    let descriptors: GenerationReference[] = [];
    const handles = new Map([
      [1, "upload-secret-one"],
      [2, "upload-secret-two"],
    ]);
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async (
          input: RequestInfo | URL,
          init?: RequestInit,
        ): Promise<Response> => {
          const method = init?.method ?? "GET";
          const headers = new Headers(init?.headers);
          calls.push({
            url: String(input),
            method,
            headers,
            body: init?.body,
          });
          if (method === "POST") {
            const payload = JSON.parse(String(init?.body)) as {
              request: ReferenceUploadRequest;
              upload_references: number[];
            };
            descriptors = payload.request.references ?? [];
            expect(payload.upload_references).toEqual([1, 2]);
            expect(
              descriptors.map((reference) => reference.media.authority),
            ).toEqual(["descriptor", "descriptor"]);
            expect(JSON.stringify(payload)).not.toContain(base64("image-one"));
            return Response.json({
              instance_id: INSTANCE_ID,
              expires_at_ms: 20_000,
              request_scope_sha256: SCOPE_DIGEST,
              session_handle: "session-secret",
              // Slot order is not request authority; the one-based index is.
              uploads: [
                { reference: 2, handle: handles.get(2) },
                { reference: 1, handle: handles.get(1) },
              ],
            });
          }
          if (method === "PUT") {
            const handle = headers.get("x-mold-reference-upload");
            const index = handle === handles.get(1) ? 1 : 2;
            const body = init?.body;
            expect(body).toBeInstanceOf(Blob);
            expect(headers.has("content-length")).toBe(false);
            expect((body as Blob).size).toBe(
              index === 1 ? "image-one".length : "audio-two".length,
            );
            return Response.json(
              uploadComplete(descriptors[index - 1]!, index, index === 2),
            );
          }
          expect(method).toBe("DELETE");
          return new Response(null, { status: 204 });
        },
      ),
    );

    const lease = await prepareReferenceUploads({
      target: TARGET,
      expectedInstanceId: INSTANCE_ID,
      capabilities: CAPABILITIES,
      request,
      now: () => 10_000,
    });

    expect(
      request.references?.map((reference) => reference.media.authority),
    ).toEqual(["inline", "inline"]);
    expect(
      lease.request.references?.map((reference) => reference.media),
    ).toEqual([
      { authority: "upload", handle: "upload-secret-one" },
      { authority: "upload", handle: "upload-secret-two" },
    ]);
    expect(lease.expiresAtMs).toBe(20_000);
    expect(lease.requestScopeSha256).toBe(REBOUND_SCOPE_DIGEST);
    expect(JSON.stringify(lease)).not.toContain("secret");
    expect(calls).toHaveLength(3);
    expect(calls.every((call) => call.url.startsWith(TARGET.baseUrl))).toBe(
      true,
    );
    expect(calls.every((call) => !call.url.includes("secret"))).toBe(true);
    expect(
      calls.every((call) => call.headers.get("x-api-key") === TARGET.apiKey),
    ).toBe(true);

    await lease.cancel();
    expect(calls).toHaveLength(4);
    const cancellation = calls[3]!;
    expect(cancellation.url).toBe(
      `${TARGET.baseUrl}${CAPABILITIES.session_path}`,
    );
    expect(cancellation.headers.get(CAPABILITIES.session_handle_header)).toBe(
      "session-secret",
    );
    await lease.cancel();
    expect(calls).toHaveLength(4);
  });

  it("computes a missing inline digest before creating the descriptor scope", async () => {
    const request = await requestFixture();
    delete request.references?.[0]?.provenance?.sha256;
    let descriptorDigest = "";
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async (
          _input: RequestInfo | URL,
          init?: RequestInit,
        ): Promise<Response> => {
          if (init?.method === "POST") {
            const payload = JSON.parse(String(init.body)) as {
              request: ReferenceUploadRequest;
            };
            descriptorDigest =
              payload.request.references?.[0]?.provenance?.sha256 ?? "";
            return Response.json({
              instance_id: INSTANCE_ID,
              expires_at_ms: 20_000,
              request_scope_sha256: SCOPE_DIGEST,
              session_handle: "session-secret",
              uploads: [
                { reference: 1, handle: "upload-one" },
                { reference: 2, handle: "upload-two" },
              ],
            });
          }
          if (init?.method === "DELETE") {
            return new Response(null, { status: 204 });
          }
          const handle = new Headers(init?.headers).get(
            CAPABILITIES.upload_handle_header,
          );
          const index = handle === "upload-one" ? 1 : 2;
          const reference = request.references?.[index - 1];
          expect(reference).toBeDefined();
          const copy = JSON.parse(
            JSON.stringify(reference),
          ) as GenerationReference;
          copy.provenance = {
            ...(copy.provenance ?? {}),
            sha256:
              index === 1
                ? descriptorDigest
                : (request.references?.[1]?.provenance?.sha256 ?? null),
          };
          return Response.json(uploadComplete(copy, index, index === 2));
        },
      ),
    );

    const lease = await prepareReferenceUploads({
      target: TARGET,
      expectedInstanceId: INSTANCE_ID,
      capabilities: CAPABILITIES,
      request,
      now: () => 10_000,
    });
    expect(descriptorDigest).toBe(await digest("image-one"));
    expect(lease.request.references?.[0]?.provenance?.sha256).toBe(
      descriptorDigest,
    );
  });

  it("replaces provisional media facts with canonical upload metadata and its rebound scope", async () => {
    const request = await requestFixture();
    request.references![0]!.provenance!.name = "  identity.png  ";
    let descriptors: GenerationReference[] = [];
    const methods: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async (
          _input: RequestInfo | URL,
          init?: RequestInit,
        ): Promise<Response> => {
          const method = init?.method ?? "GET";
          methods.push(method);
          if (method === "POST") {
            const payload = JSON.parse(String(init?.body)) as {
              request: ReferenceUploadRequest;
            };
            descriptors = payload.request.references ?? [];
            expect(descriptors[0]?.provenance?.name).toBe("identity.png");
            return Response.json({
              instance_id: INSTANCE_ID,
              expires_at_ms: 20_000,
              request_scope_sha256: SCOPE_DIGEST,
              session_handle: "canonical-session",
              uploads: [
                { reference: 1, handle: "canonical-image" },
                { reference: 2, handle: "canonical-audio" },
              ],
            });
          }
          const handle = new Headers(init?.headers).get(
            CAPABILITIES.upload_handle_header,
          );
          const index = handle === "canonical-image" ? 1 : 2;
          const canonical = JSON.parse(
            JSON.stringify(descriptors[index - 1]),
          ) as GenerationReference;
          if (canonical.kind === "image") {
            canonical.width = 40;
            canonical.height = 30;
          } else if (canonical.kind === "audio") {
            // The provisional packet/container timeline is not queue
            // authority; the host's decoded PCM count wins.
            canonical.duration_ms = 2_001;
            canonical.sample_count = 48_001;
          }
          return Response.json(
            uploadComplete(
              canonical,
              index,
              index === 2,
              (index === 1 ? "c" : "d").repeat(64),
            ),
          );
        },
      ),
    );

    const lease = await prepareReferenceUploads({
      target: TARGET,
      expectedInstanceId: INSTANCE_ID,
      capabilities: CAPABILITIES,
      request,
      now: () => 10_000,
    });

    expect(methods).toEqual(["POST", "PUT", "PUT"]);
    expect(lease.requestScopeSha256).toBe("d".repeat(64));
    expect(lease.request.references?.[0]).toMatchObject({
      provenance: { name: "identity.png" },
      width: 40,
      height: 30,
      media: { authority: "upload", handle: "canonical-image" },
    });
    expect(lease.request.references?.[1]).toMatchObject({
      duration_ms: 2_001,
      sample_count: 48_001,
      media: { authority: "upload", handle: "canonical-audio" },
    });
  });

  it("fails closed before network access when capability, auth, or digest authority is invalid", async () => {
    const request = await requestFixture();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      prepareReferenceUploads({
        target: TARGET,
        expectedInstanceId: INSTANCE_ID,
        capabilities: { ...CAPABILITIES, available: false },
        request,
      }),
    ).rejects.toMatchObject({
      code: "REFERENCE_UPLOAD_UNAVAILABLE",
    });
    await expect(
      prepareReferenceUploads({
        target: { ...TARGET, apiKey: null },
        expectedInstanceId: INSTANCE_ID,
        capabilities: CAPABILITIES,
        request,
      }),
    ).rejects.toMatchObject({
      code: "REFERENCE_UPLOAD_AUTH_REQUIRED",
    });
    const mismatched = await requestFixture();
    mismatched.references![0]!.provenance!.sha256 = "f".repeat(64);
    await expect(
      prepareReferenceUploads({
        target: TARGET,
        expectedInstanceId: INSTANCE_ID,
        capabilities: CAPABILITIES,
        request: mismatched,
      }),
    ).rejects.toMatchObject({
      code: "REFERENCE_UPLOAD_DIGEST_MISMATCH",
    });
    await expect(
      prepareReferenceUploads({
        target: TARGET,
        expectedInstanceId: INSTANCE_ID,
        capabilities: {
          ...CAPABILITIES,
          upload_handle_header: "Content-Length",
        },
        request,
      }),
    ).rejects.toMatchObject({
      code: "REFERENCE_UPLOAD_CAPABILITY_INVALID",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("cancels a returned session when instance identity or upload metadata drifts", async () => {
    const request = await requestFixture();
    const methods: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async (
          _input: RequestInfo | URL,
          init?: RequestInit,
        ): Promise<Response> => {
          methods.push(init?.method ?? "GET");
          if (init?.method === "POST") {
            return Response.json({
              instance_id: "replacement-instance",
              expires_at_ms: 20_000,
              request_scope_sha256: SCOPE_DIGEST,
              session_handle: "session-to-cancel",
              uploads: [
                { reference: 1, handle: "upload-one" },
                { reference: 2, handle: "upload-two" },
              ],
            });
          }
          return new Response(null, { status: 204 });
        },
      ),
    );
    await expect(
      prepareReferenceUploads({
        target: TARGET,
        expectedInstanceId: INSTANCE_ID,
        capabilities: CAPABILITIES,
        request,
        now: () => 10_000,
      }),
    ).rejects.toMatchObject({
      code: "REFERENCE_UPLOAD_INSTANCE_MISMATCH",
    });
    expect(methods).toEqual(["POST", "DELETE"]);

    methods.length = 0;
    let descriptor: GenerationReference | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async (
          _input: RequestInfo | URL,
          init?: RequestInit,
        ): Promise<Response> => {
          methods.push(init?.method ?? "GET");
          if (init?.method === "POST") {
            const payload = JSON.parse(String(init.body)) as {
              request: ReferenceUploadRequest;
            };
            descriptor = payload.request.references?.[0] ?? null;
            return Response.json({
              instance_id: INSTANCE_ID,
              expires_at_ms: 20_000,
              request_scope_sha256: SCOPE_DIGEST,
              session_handle: "session-to-cancel",
              uploads: [
                { reference: 1, handle: "upload-one" },
                { reference: 2, handle: "upload-two" },
              ],
            });
          }
          if (init?.method === "PUT") {
            return Response.json({
              ...uploadComplete(descriptor!, 1, false),
              metadata: {
                ...metadataFor(descriptor!, 1),
                mime_type: "image/jpeg",
              },
            });
          }
          return new Response(null, { status: 204 });
        },
      ),
    );
    await expect(
      prepareReferenceUploads({
        target: TARGET,
        expectedInstanceId: INSTANCE_ID,
        capabilities: CAPABILITIES,
        request,
        now: () => 10_000,
      }),
    ).rejects.toMatchObject({
      code: "REFERENCE_UPLOAD_RESPONSE_MISMATCH",
    });
    expect(methods).toEqual(["POST", "PUT", "DELETE"]);
  });

  it("rejects duplicate one-based slots and cleans up the untrusted session", async () => {
    const request = await requestFixture();
    const calls: RequestInit[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async (
          _input: RequestInfo | URL,
          init?: RequestInit,
        ): Promise<Response> => {
          calls.push(init ?? {});
          if (init?.method === "POST") {
            return Response.json({
              instance_id: INSTANCE_ID,
              expires_at_ms: 20_000,
              request_scope_sha256: SCOPE_DIGEST,
              session_handle: "duplicate-session",
              uploads: [
                { reference: 1, handle: "upload-one" },
                { reference: 1, handle: "upload-one-again" },
              ],
            });
          }
          return new Response(null, { status: 204 });
        },
      ),
    );

    await expect(
      prepareReferenceUploads({
        target: TARGET,
        expectedInstanceId: INSTANCE_ID,
        capabilities: CAPABILITIES,
        request,
        now: () => 10_000,
      }),
    ).rejects.toMatchObject({ code: "REFERENCE_UPLOAD_SESSION_INVALID" });
    expect(calls.map((call) => call.method)).toEqual(["POST", "DELETE"]);
    expect(
      new Headers(calls[1]?.headers).get(CAPABILITIES.session_handle_header),
    ).toBe("duplicate-session");
  });

  it("identifies only fresh inline Ref2VA snapshots as upload work", async () => {
    const request = await requestFixture();
    expect(requestNeedsReferenceUpload(request)).toBe(true);
    expect(
      requestNeedsReferenceUpload({
        ...request,
        model: "minimax-h3-fl2va",
      }),
    ).toBe(false);
    expect(
      requestNeedsReferenceUpload({
        ...request,
        references: request.references?.map((reference) => ({
          ...reference,
          media: { authority: "descriptor" as const },
        })) as GenerationReference[],
      }),
    ).toBe(false);
  });
});

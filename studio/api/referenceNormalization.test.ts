import { afterEach, describe, expect, it, vi } from "vitest";
import { sha256 } from "@noble/hashes/sha2.js";
import { prepareNormalizedReferenceUploads } from "./referenceNormalization";
import type { ReferenceUploadCapabilities } from "./referenceUploads";
const hex = (bytes: Uint8Array) =>
  Array.from(bytes, (x) => x.toString(16).padStart(2, "0")).join("");
const bytes = new TextEncoder().encode("original-video");
const original = {
  mime_type: "video/x-m4v",
  name: "clip.m4v",
  byte_length: bytes.length,
  sha256: hex(sha256(bytes)),
};
const cap: ReferenceUploadCapabilities = {
  available: true,
  protocol_version: 2,
  requires_api_key: true,
  session_path: "/api/generate/reference-upload-sessions",
  upload_path: "/api/generate/reference-upload",
  session_handle_header: "x-mold-reference-upload-session",
  upload_handle_header: "x-mold-reference-upload",
  max_file_bytes: 268435456,
  max_session_bytes: 1073741824,
  max_active_sessions: 4,
  session_ttl_ms: 30000,
  video_normalization: {
    available: true,
    protocol_version: 1,
    input_roles: ["reference_video"],
    candidate_session_path: "/api/generate/reference-normalization-sessions",
    finalize_path: "/api/generate/reference-normalization-finalize",
    candidate_digest_header: "x-mold-reference-candidate-sha256",
    max_original_file_bytes: 268435456,
    max_canonical_file_bytes: 268435456,
    max_session_bytes: 1073741824,
    accepted_mime_types: ["video/mp4", "video/x-m4v"],
    accepted_extensions: ["mp4", "m4v"],
    accepted_video_codecs: ["h264"],
    canonical_mime_type: "video/mp4",
  },
};
const metadata = {
  index: 1,
  kind: "video",
  name: "clip.m4v",
  sha256: original.sha256,
  mime_type: "video/mp4",
  width: 64,
  height: 64,
  frame_count: 48,
  duration_ms: 2000,
  fps: 24,
  has_audio: false,
  audio_duration_ms: null,
  audio_sample_count: null,
  audio_sample_rate: null,
  audio_channels: null,
};
function transport(
  tamper:
    | "original"
    | "scope"
    | "instance"
    | "final"
    | "compact"
    | "converted"
    | "changed_unchanged"
    | null = null,
) {
  let fingerprint = "";
  const canonical =
    tamper === "converted" || tamper === "changed_unchanged"
      ? "c".repeat(64)
      : original.sha256;
  const receipt = {
    reference: 1,
    kind: "video",
    original_sha256: original.sha256,
    canonical_sha256: canonical,
  };
  return vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
    const url = String(_url);
    if (init?.method === "DELETE") {
      expect(new URL(url).pathname).toBe(cap.session_path);
      expect(new Headers(init.headers).get(cap.session_handle_header)).toBe(
        "mrs_secret",
      );
      return new Response(null, { status: 204 });
    }
    if (url.endsWith("reference-normalization-sessions")) {
      const body = String(init?.body);
      expect(JSON.parse(body).request).not.toHaveProperty("references");
      fingerprint = hex(
        sha256(
          new TextEncoder().encode(
            `mold-reference-normalization-candidate-v1\0${body}`,
          ),
        ),
      );
      expect(
        new Headers(init?.headers).get("x-mold-reference-candidate-sha256"),
      ).toBe(fingerprint);
      return Response.json({
        protocol_version: 1,
        instance_id: "instance",
        session_id: "session",
        candidate_scope_sha256: fingerprint,
        expires_at_ms: Date.now() + 30000,
        session_handle: "mrs_secret",
        session_complete: false,
        uploads: [{ reference: 1, handle: "mru_secret" }],
      });
    }
    const envelope = {
      protocol_version: 1,
      instance_id: tamper === "instance" ? "replacement" : "instance",
      session_id: "session",
      candidate_scope_sha256: tamper === "scope" ? "0".repeat(64) : fingerprint,
    };
    if (url.endsWith("reference-normalization-finalize")) {
      const response = {
        ...envelope,
        session_complete: true,
        request_scope_sha256: "f".repeat(64),
        receipts: [receipt],
        references: [
          {
            kind: "video",
            media: { authority: "descriptor" },
            provenance: { name: original.name, sha256: canonical },
            mime_type: "video/mp4",
            width: tamper === "final" ? 128 : 64,
            height: 64,
            frame_count: 48,
            duration_ms: 2000,
            fps: 24,
            has_audio: false,
            audio_duration_ms: null,
            audio_sample_count: null,
            audio_sample_rate: null,
            audio_channels: null,
          },
        ],
      };
      if (tamper === "compact") {
        const descriptor = response.references[0]! as Record<string, unknown>;
        for (const key of [
          "has_audio",
          "audio_duration_ms",
          "audio_sample_count",
          "audio_sample_rate",
          "audio_channels",
        ])
          delete descriptor[key];
      }
      return Response.json(response);
    }
    const wireMetadata: Record<string, unknown> = {
      ...metadata,
      sha256: canonical,
    };
    if (tamper === "compact")
      for (const key of [
        "has_audio",
        "audio_duration_ms",
        "audio_sample_count",
        "audio_sample_rate",
        "audio_channels",
      ])
        delete wireMetadata[key];
    return Response.json({
      ...envelope,
      reference: 1,
      session_complete: false,
      metadata: wireMetadata,
      receipt: {
        version: 1,
        kind: "video",
        original_sha256:
          tamper === "original" ? "0".repeat(64) : original.sha256,
        original_byte_length: bytes.length,
        original_mime_type: original.mime_type,
        canonical_sha256: canonical,
        canonical_byte_length: bytes.length,
        canonical_mime_type: "video/mp4",
        converted: tamper === "converted",
      },
    });
  });
}
const options = () => ({
  target: { baseUrl: "https://gpu.example", apiKey: "secret" },
  expectedInstanceId: "instance",
  capabilities: cap,
  request: { model: "minimax-h3-ref2va", prompt: "test", frames: 124, fps: 24 },
  candidates: [
    {
      index: 1,
      type: "fresh_video" as const,
      kind: "video" as const,
      original,
    },
  ],
  uploadBodies: new Map([[1, new Blob([bytes], { type: original.mime_type })]]),
  current: () => true,
});
afterEach(() => vi.unstubAllGlobals());
describe("negotiated reference normalization", () => {
  it("requires finalization and preserves canonical M4V bytes despite a canonical MIME alias", async () => {
    const fetch = transport();
    vi.stubGlobal("fetch", fetch);
    const lease = await prepareNormalizedReferenceUploads(options());
    expect(fetch.mock.calls.map((c) => new URL(String(c[0])).pathname)).toEqual(
      [
        cap.video_normalization!.candidate_session_path,
        cap.upload_path,
        cap.video_normalization!.finalize_path,
      ],
    );
    expect(lease.request.references?.[0]).toMatchObject({
      kind: "video",
      mime_type: "video/mp4",
      provenance: { sha256: original.sha256 },
      media: { authority: "upload", handle: "mru_secret" },
    });
    expect(JSON.stringify(lease)).not.toContain("secret");
    expect(Object.isFrozen(lease.request)).toBe(true);
    expect(Object.isFrozen(lease.request.references?.[0]?.media)).toBe(true);
  });
  it("accepts actual Rust serde omission of false has_audio and absent optional soundtrack facts", async () => {
    vi.stubGlobal("fetch", transport("compact"));
    const lease = await prepareNormalizedReferenceUploads(options());
    expect(lease.request.references?.[0]).toMatchObject({
      kind: "video",
      has_audio: false,
    });
  });
  it("binds a converted video to its original upload digest and canonical generation digest separately", async () => {
    vi.stubGlobal("fetch", transport("converted"));
    const value = options();
    const lease = await prepareNormalizedReferenceUploads(value);
    expect(lease.request.references?.[0]?.provenance?.sha256).toBe(
      "c".repeat(64),
    );
    expect(value.candidates[0]!.original.sha256).toBe(original.sha256);
  });
  it("refuses changed bytes when the receipt claims conversion was unnecessary", async () => {
    const fetch = transport("changed_unchanged");
    vi.stubGlobal("fetch", fetch);
    await expect(prepareNormalizedReferenceUploads(options())).rejects.toThrow(
      /normalization/i,
    );
    expect(fetch.mock.calls.at(-1)?.[1]?.method).toBe("DELETE");
  });
  it.each(["original", "scope", "instance", "final"] as const)(
    "refuses %s receipt tampering and cancels the entire session",
    async (tamper) => {
      const fetch = transport(tamper);
      vi.stubGlobal("fetch", fetch);
      await expect(
        prepareNormalizedReferenceUploads(options()),
      ).rejects.toThrow(/normalization/i);
      expect(fetch.mock.calls.at(-1)?.[1]?.method).toBe("DELETE");
    },
  );
  it("cancels the captured session and sends no bytes when authority changes after session creation", async () => {
    const fetch = transport();
    let current = true;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const response = await fetch(url, init);
        if (init?.method === "POST") current = false;
        return response;
      }),
    );
    await expect(
      prepareNormalizedReferenceUploads({
        ...options(),
        current: () => current,
      }),
    ).rejects.toThrow(/no longer current/i);
    expect(fetch.mock.calls.map((call) => call[1]?.method)).toEqual([
      "POST",
      "DELETE",
    ]);
  });
  it("cancels an upload session when Stop arrives during an upload response", async () => {
    const fetch = transport(),
      controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        const response = await fetch(url, init);
        if (init?.method === "PUT") controller.abort();
        return response;
      }),
    );
    await expect(
      prepareNormalizedReferenceUploads({
        ...options(),
        signal: controller.signal,
      }),
    ).rejects.toThrow();
    expect(fetch.mock.calls.map((call) => call[1]?.method)).toEqual([
      "POST",
      "PUT",
      "DELETE",
    ]);
  });
  it("refuses an older machine without normalization before opening a session", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const value = options();
    const { video_normalization: _, ...older } = value.capabilities;
    await expect(
      prepareNormalizedReferenceUploads({ ...value, capabilities: older }),
    ).rejects.toThrow(/normalization/i);
    expect(fetch).not.toHaveBeenCalled();
  });
  it("captures target and request fields before hashing/upload await", async () => {
    const fetch = transport();
    const value = options();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
        value.target.baseUrl = "https://replacement.example";
        value.target.apiKey = "replacement";
        value.request.prompt = "mutated";
        return fetch(url, init);
      }),
    );
    const lease = await prepareNormalizedReferenceUploads(value);
    expect(lease.request.prompt).toBe("test");
    expect(
      fetch.mock.calls.every((c) =>
        String(c[0]).startsWith("https://gpu.example/"),
      ),
    ).toBe(true);
  });
});
it("fingerprints the exact protected legacy wire body while keeping final prompts literal", async () => {
  const input = options(),
    text = String.raw`C:\new\render` + "\r\nActual line";
  const request = {
    ...input.request,
    prompt: text,
    negative_prompt: text,
    original_prompt: text,
    prompt_transform: { root_prompt: text, source_prompt: text },
  };
  const fetchMock = transport();
  vi.stubGlobal("fetch", fetchMock);
  const lease = await prepareNormalizedReferenceUploads<typeof request>({
    ...input,
    request,
  });
  const body = JSON.parse(String(fetchMock.mock.calls[0]![1]!.body));
  expect(body.request.prompt).toBe(text.replaceAll("\\", "\\\\"));
  expect(body.request.prompt_transform.root_prompt).toBe(
    text.replaceAll("\\", "\\\\"),
  );
  expect(lease.request.prompt).toBe(text);
  expect(lease.request.prompt_transform.source_prompt).toBe(text);
});

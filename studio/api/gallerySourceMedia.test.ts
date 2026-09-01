import { afterEach, describe, expect, it, vi } from "vitest";
import {
  __testing__,
  createRetainedSourceMediaReuseSession,
  retainedSourceMediaDisclosable,
  retainedSourceMediaDisclosure,
  retainedSourceMediaInventory,
  relayRetainedSourceMedia,
} from "./gallerySourceMedia";

afterEach(() => vi.restoreAllMocks());

describe("gallery retained source media API", () => {
  it("encodes the exact gallery identity and opaque member", () => {
    expect(__testing__.inventoryPath("source one.png")).toBe(
      "/api/gallery/source-media/source%20one.png",
    );
    expect(__testing__.memberPath("print.png", "a".repeat(64))).toBe(
      `/api/gallery/source-media/print.png/${"a".repeat(64)}`,
    );
  });

  it("creates a request-bound one-time session without exposing member paths", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          instance_id: "host-a",
          expires_at: 42,
          request_sha256: "b".repeat(64),
          session_handle: "opaque-session",
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    const target = { baseUrl: "http://host-a:7680", apiKey: "secret" };
    const request = { model: "flux", prompt: "reuse", source_image: undefined };

    await expect(
      createRetainedSourceMediaReuseSession(
        "print one.png",
        ["opaque-member"],
        request,
        target,
      ),
    ).resolves.toMatchObject({
      session_handle: "opaque-session",
      instance_id: "host-a",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://host-a:7680/api/gallery/source-media/print%20one.png/reuse-sessions",
      expect.objectContaining({
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify({
          target_request: request,
          member_ids: ["opaque-member"],
        }),
      }),
    );
    const headers = fetchMock.mock.calls[0]![1]!.headers as Headers;
    expect(headers.get("x-api-key")).toBe("secret");
    expect(headers.get("content-type")).toBe("application/json");
  });

  it("stays quiet about an unavailable answer for a print that shipped no bytes", () => {
    // A text-to-image print. The server can only answer `unavailable_legacy`
    // here — "Reattach it before developing" for a source that never existed.
    // The probe still runs (the server is the authority); only the toast is
    // withheld.
    expect(retainedSourceMediaDisclosable({})).toBe(false);
    expect(retainedSourceMediaDisclosable(null)).toBe(false);
    expect(retainedSourceMediaDisclosable(undefined)).toBe(false);
    // An empty array is not evidence either.
    expect(retainedSourceMediaDisclosable({ references: [] })).toBe(false);
    expect(retainedSourceMediaDisclosable({ keyframes: [] })).toBe(false);
  });

  it("excludes a bare source name, which carries no restorable bytes", () => {
    // MiniMax H3 records `source_image_name` alone. Its retained members are
    // all filtered by the server's `downloadable_role`, so probing on it only
    // ever produced an empty list.
    expect(
      retainedSourceMediaDisclosable({
        source_image_name: "closing-boundary.png",
      } as Record<string, unknown>),
    ).toBe(false);
  });

  it("discloses for every recorded marker of a downloadable role", () => {
    for (const metadata of [
      { extend_overlap_frames: 8 },
      { source_image_sha256: "a".repeat(64) },
      { id_image_sha256: "b".repeat(64) },
      { source_video_path: "/clip.mp4" },
      { audio_file_path: "/voice.wav" },
      { extend_video_path: "/tail.mp4" },
      { edit_image_sha256s: ["c".repeat(64)] },
      { id_image_sha256s: ["d".repeat(64)] },
      { references: [{ media: { authority: "descriptor" } }] },
      { keyframes: [{ frame: 0 }] },
    ]) {
      expect(retainedSourceMediaDisclosable(metadata)).toBe(true);
    }
  });

  it("reports a middleware 401 as the auth state instead of throwing", async () => {
    // A host that enforces API keys refuses before the handler answers. Without
    // this, the caller's catch swallowed it and the one case the "connect this
    // machine with an API key" sentence is about showed nothing at all.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          error: "missing X-Api-Key header",
          code: "UNAUTHORIZED",
        }),
        { status: 401, headers: { "content-type": "application/json" } },
      ),
    );
    await expect(
      retainedSourceMediaInventory("print.png", {
        baseUrl: "http://host-a:7680",
        apiKey: null,
      }),
    ).resolves.toEqual({ availability: "unavailable_auth", members: [] });
  });

  it("still throws for a failure that is not an auth refusal", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("boom", { status: 500 }),
    );
    await expect(
      retainedSourceMediaInventory("print.png", {
        baseUrl: "http://host-a:7680",
        apiKey: "secret",
      }),
    ).rejects.toThrow();
  });

  it("maps every unavailable state to an explicit reuse disclosure", () => {
    expect(retainedSourceMediaDisclosure("unavailable_legacy")).toContain(
      "older print",
    );
    expect(
      retainedSourceMediaDisclosure("unavailable_missing_or_corrupt"),
    ).toContain("missing or damaged");
    expect(retainedSourceMediaDisclosure("unavailable_auth")).toContain(
      "API key",
    );
    expect(retainedSourceMediaDisclosure("available")).toBeNull();
  });

  it("relays authenticated retained bytes into a cross-host request by role", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(new Uint8Array([1, 2, 3]), { status: 200 }),
      )
      .mockResolvedValueOnce(
        new Response(new Uint8Array([4, 5]), { status: 200 }),
      );
    const request = { model: "flux", prompt: "relay" };
    const relayed = await relayRetainedSourceMedia(
      "print.png",
      [
        {
          member_id: "source",
          role: "source_image",
          display_name: "source.png",
          size_bytes: 3,
        },
        {
          member_id: "mask",
          role: "mask_image",
          display_name: "mask.png",
          size_bytes: 2,
        },
      ],
      request,
      { baseUrl: "http://origin:7680", apiKey: "secret" },
    );
    expect(relayed).toEqual({
      model: "flux",
      prompt: "relay",
      source_image: "AQID",
      mask_image: "BAU=",
    });
  });

  it("refuses to overwrite a request that already owns the selected role", async () => {
    await expect(
      relayRetainedSourceMedia(
        "print.png",
        [
          {
            member_id: "source",
            role: "source_image",
            display_name: "source.png",
            size_bytes: 3,
          },
        ],
        { source_image: "existing" },
        { baseUrl: "http://origin:7680", apiKey: "secret" },
      ),
    ).rejects.toThrow("already contains source_image");
  });

  it("relays every reusable backend role without paths or store identities", async () => {
    const encoded = (value: string) => new TextEncoder().encode(value);
    const payloads = [
      encoded("source"),
      encoded("identity-one"),
      encoded("identity-two"),
      encoded("edit-one"),
      encoded("mask"),
      encoded("control"),
      encoded("audio"),
      encoded("source-video"),
      encoded("extend-video"),
      encoded(JSON.stringify({ frame: 1 })),
      encoded("reference"),
    ];
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      const bytes = payloads.shift();
      return new Response(bytes, { status: bytes ? 200 : 404 });
    });
    const members = [
      "source_image",
      "identity_images",
      "identity_images",
      "edit_images",
      "mask_image",
      "control_image",
      "audio_file_path",
      "source_video_path",
      "extend_video_path",
      "keyframes",
      "references",
    ].map((role, index) => ({
      member_id: `member-${index}`,
      role,
      display_name: `${role}-${index}`,
      size_bytes: 1,
    }));
    const request = {
      model: "h3",
      references: [{ kind: "image", media: { authority: "descriptor" } }],
    };
    const relayed = await relayRetainedSourceMedia(
      "print.png",
      members,
      request,
      { baseUrl: "http://origin:7680", apiKey: "secret" },
    );
    expect(relayed).toMatchObject({
      source_image: "c291cmNl",
      id_images: ["aWRlbnRpdHktb25l", "aWRlbnRpdHktdHdv"],
      edit_images: ["ZWRpdC1vbmU="],
      mask_image: "bWFzaw==",
      control_image: "Y29udHJvbA==",
      audio_file: "YXVkaW8=",
      source_video: "c291cmNlLXZpZGVv",
      extend_video: "ZXh0ZW5kLXZpZGVv",
      keyframes: [{ frame: 1 }],
      references: [
        { kind: "image", media: { authority: "inline", data: "cmVmZXJlbmNl" } },
      ],
    });
    expect(JSON.stringify(relayed)).not.toMatch(
      /queue-media|server_path|pin_id/,
    );
  });
});

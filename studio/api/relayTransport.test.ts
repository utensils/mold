import { webcrypto } from "node:crypto";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  relayFetch,
  clearRelayInfoCache,
  validateRelayObjectUrl,
} from "./relayTransport";
beforeEach(() => {
  clearRelayInfoCache();
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(() => vi.unstubAllGlobals());
it("hashes empty mutation bodies without probing ordinary requests", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", fetch);
  await relayFetch("https://machine.example/api/status", {
    method: "POST",
    headers: { "x-api-key": "secret" },
  });
  expect(fetch).toHaveBeenCalledOnce();
  expect(
    new Headers(fetch.mock.calls[0]?.[1]?.headers).get("x-amz-content-sha256"),
  ).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
});
it("preserves encoded request targets and repeated query values", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", fetch);
  await relayFetch(
    "https://machine.example/api/gallery/image/a%20b.png?media_token=x%2By&part=1&part=2",
    { headers: { "x-api-key": "secret" } },
  );
  expect(
    new Headers(fetch.mock.calls[0]?.[1]?.headers).get("x-mold-request-target"),
  ).toBe("/api/gallery/image/a%20b.png?media_token=x%2By&part=1&part=2");
});
it.each(["PATCH", "DELETE"])(
  "stages large %s bodies without Mold credentials to S3",
  async (method) => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url, init) => {
        calls.push({ url: String(url), init });
        if (String(url).endsWith("/info"))
          return Response.json({
            protocol: 2,
            object_origin:
              "https://mold-relay-123456789012-us-east-1.s3.dualstack.us-east-1.amazonaws.com",
            upload_threshold: 2097152,
            max_body_bytes: 67108864,
          });
        if (String(url).endsWith("/uploads"))
          return Response.json({
            id: "up1",
            url: "https://mold-relay-123456789012-us-east-1.s3.us-east-1.amazonaws.com/object?signature=short",
            headers: {},
            expires_at: 9999999999,
          });
        return new Response("original");
      }),
    );
    const body = new Uint8Array(2097153).fill(7);
    const response = await relayFetch(
      "https://machine.example/api/arbitrary?query=1",
      {
        method,
        headers: {
          "x-api-key": "secret",
          "content-type": "application/octet-stream",
        },
        body,
      },
    );
    expect(await response.text()).toBe("original");
    expect(calls.map((c) => c.url)).toEqual([
      "https://machine.example/_mold/relay/info",
      "https://machine.example/_mold/relay/uploads",
      "https://mold-relay-123456789012-us-east-1.s3.us-east-1.amazonaws.com/object?signature=short",
      "https://machine.example/_mold/relay/request",
    ]);
    expect(new Headers(calls[2]?.init?.headers).has("x-api-key")).toBe(false);
    expect(calls[2]?.init?.redirect).toBe("error");
    const grant = JSON.parse(String(calls[1]?.init?.body));
    expect(grant.path).toBe("/api/arbitrary?query=1");
    expect(grant.method).toBe(method);
    expect(grant.size).toBe(body.length);
    expect(JSON.parse(String(calls[3]?.init?.body))).toEqual({ id: "up1" });
  },
);
it("follows only explicit same-origin object envelopes without a key", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          url: "https://machine.example/_mold/objects/file?Signature=short",
          status: 206,
          headers: { "content-range": "bytes 0-2/10" },
        }),
        { headers: { "x-mold-relay-object": "1" } },
      ),
    )
    .mockResolvedValueOnce(new Response("abc"));
  vi.stubGlobal("fetch", fetch);
  const response = await relayFetch("https://machine.example/api/file", {
    headers: { "x-api-key": "secret" },
  });
  expect(response.status).toBe(206);
  expect(response.headers.get("content-range")).toBe("bytes 0-2/10");
  expect(await response.text()).toBe("abc");
  expect(new Headers(fetch.mock.calls[1]?.[1]?.headers).has("x-api-key")).toBe(
    false,
  );
});
it("refuses foreign and nonreserved object locations", () => {
  for (const url of [
    "https://evil.example/_mold/objects/x",
    "https://machine.example/api/x",
    "http://machine.example/_mold/objects/x",
  ])
    expect(() =>
      validateRelayObjectUrl(url, "https://machine.example"),
    ).toThrow();
});
it("accepts only signed objects from the pinned relay S3 bucket", () => {
  const objectOrigin =
    "https://mold-relay-123456789012-us-east-1.s3.dualstack.us-east-1.amazonaws.com";
  const signed = `${objectOrigin}/_mold/objects/a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=${"a".repeat(64)}&X-Amz-Expires=900`;
  expect(
    validateRelayObjectUrl(signed, "https://machine.example", objectOrigin),
  ).toBe(signed);
  for (const url of [
    signed.replace("123456789012", "999999999999"),
    signed.replace("X-Amz-Signature=", "unsigned="),
    signed.replace("Expires=900", "Expires=901"),
    signed + "#fragment",
    signed.replace("dualstack.us-east-1", "dualstack.us-west-2"),
  ])
    expect(() =>
      validateRelayObjectUrl(url, "https://machine.example", objectOrigin),
    ).toThrow();
});
it("resolves a cross-origin signed object explicitly without any API headers", async () => {
  const object_origin =
    "https://mold-relay-123456789012-us-east-1.s3.dualstack.us-east-1.amazonaws.com";
  const url = `${object_origin}/_mold/objects/a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=${"a".repeat(64)}&X-Amz-Expires=900`;
  const fetch = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    if (String(input).endsWith("/info"))
      return Response.json({
        protocol: 2,
        upload_threshold: 2097152,
        max_body_bytes: 67108864,
        object_origin,
      });
    if (String(input) === url) return new Response("media");
    return new Response(
      JSON.stringify({
        url,
        status: 200,
        headers: { "content-type": "video/mp4" },
      }),
      { headers: { "x-mold-relay-object": "1" } },
    );
  });
  vi.stubGlobal("fetch", fetch);
  const response = await relayFetch(
    "https://machine.example/api/gallery/image/a.mp4",
    { headers: { "x-api-key": "secret" } },
  );
  expect(await response.text()).toBe("media");
  const download = fetch.mock.calls.find((call) => call[0] === url);
  expect(download).toBeDefined();
  expect(new Headers(download?.[1]?.headers).has("x-api-key")).toBe(false);
  expect(new Headers(download?.[1]?.headers).has("x-mold-request-target")).toBe(
    false,
  );
});
it("caches only exact-origin discovery and falls back only on 404", async () => {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url) => {
      calls.push(String(url));
      return String(url).endsWith("/info")
        ? new Response(null, { status: 404 })
        : new Response("direct");
    }),
  );
  const body = new Uint8Array(2097153);
  await relayFetch("https://one.example/api/x", { method: "POST", body });
  await relayFetch("https://one.example/api/y", { method: "POST", body });
  await relayFetch("https://two.example/api/x", { method: "POST", body });
  expect(calls.filter((url) => url.endsWith("/info"))).toEqual([
    "https://one.example/_mold/relay/info",
    "https://two.example/_mold/relay/info",
  ]);
  clearRelayInfoCache();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response(null, { status: 503 })),
  );
  await expect(
    relayFetch("https://one.example/api/x", { method: "POST", body }),
  ).rejects.toThrow("discovery failed");
});
it("refuses bodies above relay limits before issuing an upload grant", async () => {
  const fetch = vi.fn().mockResolvedValue(
    Response.json({
      protocol: 2,
      object_origin:
        "https://mold-relay-123456789012-us-east-1.s3.dualstack.us-east-1.amazonaws.com",
      upload_threshold: 2097152,
      max_body_bytes: 2097152,
    }),
  );
  vi.stubGlobal("fetch", fetch);
  await expect(
    relayFetch("https://one.example/api/x", {
      method: "POST",
      body: new Uint8Array(2097153),
    }),
  ).rejects.toThrow("64 MiB");
  expect(fetch).toHaveBeenCalledOnce();
});

it("preserves direct JSON body representation while adding its digest", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", fetch);
  const body = JSON.stringify({ enabled: false });
  await relayFetch("https://machine.example/api/devices/1", {
    method: "PATCH",
    body,
  });
  expect(fetch.mock.calls[0]?.[1]?.body).toBe(body);
  expect(
    new Headers(fetch.mock.calls[0]?.[1]?.headers).get("x-amz-content-sha256"),
  ).toMatch(/^[a-f0-9]{64}$/);
});

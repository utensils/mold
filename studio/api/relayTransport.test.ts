import { webcrypto } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
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
  expect(fetch.mock.calls[0]?.[1]?.body).toBeNull();
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
          authorization: "Bearer credential",
          cookie: "credential=secret",
          connection: "keep-alive, x-hop-secret",
          "x-hop-secret": "connection-private",
          "x-mold-viewer-secret": "private",
          "x-mold-operation-id": "operation-1",
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
    expect(new Headers(calls[2]?.init?.headers).has("authorization")).toBe(
      false,
    );
    expect(new Headers(calls[2]?.init?.headers).has("cookie")).toBe(false);
    expect(Object.fromEntries(new Headers(calls[0]?.init?.headers))).toEqual({
      authorization: "Bearer credential",
      "x-api-key": "secret",
    });
    expect(calls[0]?.init?.redirect).toBe("error");
    expect(calls[2]?.init?.redirect).toBe("error");
    const grant = JSON.parse(String(calls[1]?.init?.body));
    expect(grant.path).toBe("/api/arbitrary?query=1");
    expect(grant.method).toBe(method);
    expect(grant.size).toBe(body.length);
    expect(grant.headers).toEqual({
      "content-type": "application/octet-stream",
      "x-mold-operation-id": "operation-1",
    });
    const validatorPath = [
      "relay/aws/transfers.mjs",
      "../relay/aws/transfers.mjs",
    ]
      .map((path) => resolve(process.cwd(), path))
      .find(existsSync)!;
    const validator = pathToFileURL(validatorPath).href;
    expect(
      execFileSync(
        process.execPath,
        [
          "--input-type=module",
          "-e",
          `import {validateUpload} from ${JSON.stringify(validator)}; process.stdout.write(JSON.stringify(validateUpload(JSON.parse(process.argv[1]))));`,
          JSON.stringify(grant),
        ],
        { encoding: "utf8" },
      ),
    ).toBe(JSON.stringify(grant));
    expect(new Headers(calls[1]?.init?.headers).get("x-api-key")).toBe(
      "secret",
    );
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

it("recognizes only the bounded legacy Mold SPA as direct discovery", async () => {
  const shellPath = ["web/index.html", "../web/index.html"]
    .map((path) => resolve(process.cwd(), path))
    .find(existsSync)!;
  const shell = readFileSync(shellPath, "utf8");
  const calls: string[] = [];
  const fetch = vi.fn(async (url: RequestInfo | URL, _init?: RequestInit) => {
    calls.push(String(url));
    if (
      String(url).endsWith("/info") &&
      !new Headers(_init?.headers).has("x-api-key")
    )
      return new Response(null, { status: 401 });
    return String(url).endsWith("/info")
      ? new Response(shell, {
          headers: { "content-type": "text/html; charset=utf-8" },
        })
      : new Response("direct");
  });
  vi.stubGlobal("fetch", fetch);
  const body = new Uint8Array(2097153);
  await relayFetch("https://legacy.example/api/upload", {
    method: "POST",
    body,
    headers: { "x-api-key": "secret" },
  });
  await relayFetch("https://legacy.example/api/upload", {
    method: "POST",
    body,
  });
  expect(calls).toEqual([
    "https://legacy.example/_mold/relay/info",
    "https://legacy.example/api/upload",
    "https://legacy.example/api/upload",
  ]);
  expect(new Headers(fetch.mock.calls[0]?.[1]?.headers).get("x-api-key")).toBe(
    "secret",
  );
  expect(new Headers(fetch.mock.calls[1]?.[1]?.headers).get("x-api-key")).toBe(
    "secret",
  );
  expect(fetch.mock.calls[1]?.[1]?.body).toBe(body);
});
it.each([
  ["<html>gateway login</html>", { "content-type": "text/html" }],
  [
    "<title>mold</title><h1>mold is running</h1>This binary was built without the web gallery UI bundled.",
    { "content-type": "text/html", "x-mold-relay-protocol": "2" },
  ],
  [
    '<title>mold — studio</title><div id="app"></div>',
    { "content-type": "text/html", "x-mold-relay-protocol": "2" },
  ],
  [
    '<title>mold — studio</title><div id="app"></div>',
    { "content-type": "application/json" },
  ],
  [
    '<title>mold — studio</title><div id="app"></div>' + " ".repeat(65536),
    { "content-type": "text/html" },
  ],
])(
  "refuses unrecognized or oversized discovery responses %#",
  async (body, headers) => {
    const fetch = vi.fn().mockResolvedValue(new Response(body, { headers }));
    vi.stubGlobal("fetch", fetch);
    await expect(
      relayFetch("https://unknown.example/api/upload", {
        method: "POST",
        body: new Uint8Array(2097153),
      }),
    ).rejects.toThrow();
    expect(fetch).toHaveBeenCalledOnce();
  },
);

it("keeps HTTPS mutations usable from an insecure LAN page without WebCrypto", async () => {
  vi.stubGlobal("crypto", {});
  const fetch = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", fetch);
  await relayFetch("https://machine.example/api/status", {
    method: "POST",
    body: "abc",
  });
  expect(fetch).toHaveBeenCalledOnce();
  expect(
    new Headers(fetch.mock.calls[0]?.[1]?.headers).get("x-amz-content-sha256"),
  ).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
});

it("isolates a shared discovery from the first upload cancellation", async () => {
  let release!: (response: Response) => void;
  const discovery = new Promise<Response>((resolve) => {
    release = resolve;
  });
  const fetch = vi.fn((url: RequestInfo | URL, _init?: RequestInit) =>
    String(url).endsWith("/info")
      ? discovery
      : Promise.resolve(new Response("direct")),
  );
  vi.stubGlobal("fetch", fetch);
  const controller = new AbortController();
  const body = new Uint8Array(2097153);
  const first = relayFetch("https://shared.example/api/x", {
    method: "POST",
    body,
    signal: controller.signal,
  }).then(
    () => "success",
    (error: Error) => error.name,
  );
  const second = relayFetch("https://shared.example/api/y", {
    method: "POST",
    body,
  });
  await vi.waitFor(() => expect(fetch).toHaveBeenCalledOnce());
  expect(fetch.mock.calls[0]?.[1]?.signal).not.toBe(controller.signal);
  controller.abort();
  release(new Response(null, { status: 404 }));
  expect(await first).toBe("AbortError");
  expect(await (await second).text()).toBe("direct");
});

it("recognizes the known unbundled Mold stub and refuses a title-only page", async () => {
  const stub =
    "<title>mold</title><h1>mold is running</h1>This binary was built without the web gallery UI bundled.";
  const fetch = vi.fn(async (url: RequestInfo | URL) =>
    String(url).endsWith("/info")
      ? new Response(stub, { headers: { "content-type": "text/html" } })
      : new Response("direct"),
  );
  vi.stubGlobal("fetch", fetch);
  expect(
    await (
      await relayFetch("https://stub.example/api/upload", {
        method: "POST",
        body: new Uint8Array(2097153),
      })
    ).text(),
  ).toBe("direct");
  clearRelayInfoCache();
  fetch.mockImplementation(
    async () =>
      new Response("<title>mold</title>", {
        headers: { "content-type": "text/html" },
      }),
  );
  await expect(
    relayFetch("https://stub.example/api/upload", {
      method: "POST",
      body: new Uint8Array(2097153),
    }),
  ).rejects.toThrow();
});

import { failureCategory } from "./router-core.mjs";
import { randomUUID } from "node:crypto";
import { pipeline } from "node:stream/promises";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { store } from "./aws-store.mjs";
import { openRequest, cleanHeaders } from "./guest.mjs";
import {
  s3,
  prepareUpload,
  consumeUpload,
  stageObject,
  objectURL,
  credentialDigest,
  MAX_BODY,
  THRESHOLD,
  OBJECT_THRESHOLD,
} from "./transfers.mjs";
const lambda = new LambdaClient({});
const objectOrigin = () =>
  process.env.OBJECT_ORIGIN ??
  `https://${process.env.BUCKET_NAME}.s3.dualstack.${process.env.AWS_REGION}.amazonaws.com`;
const safeTarget = (path) =>
  typeof path === "string" &&
  path.startsWith("/") &&
  !path.startsWith("//") &&
  !/[\r\n]/.test(path) &&
  path.length <= 8192;
export function normalizeEvent(event) {
  const headers = Object.fromEntries(
    Object.entries(event.headers ?? {}).map(([k, v]) => [k.toLowerCase(), v]),
  );
  // Regional API Gateway preserves Authorization; never restore caller-supplied aliases.
  delete headers["x-mold-viewer-authorization"];
  const method =
    event.requestContext?.http?.method ?? event.httpMethod ?? "GET";
  const route = event.rawPath ?? event.path ?? "/";
  const parameters =
    event.multiValueQueryStringParameters ??
    Object.fromEntries(
      Object.entries(event.queryStringParameters ?? {}).map(([k, v]) => [
        k,
        [v],
      ]),
    );
  const query =
    event.rawQueryString ??
    new URLSearchParams(
      Object.entries(parameters).flatMap(([k, values]) =>
        (values ?? []).map((v) => [k, v]),
      ),
    ).toString();
  const canonical = `${route}${query ? "?" + query : ""}`;
  const target = headers["x-mold-request-target"] ?? canonical;
  if (!safeTarget(target) || !safeTarget(route))
    throw new Error("Invalid request target");
  const body = Buffer.from(
    event.body ?? "",
    event.isBase64Encoded ? "base64" : "utf8",
  );
  if (body.length > THRESHOLD)
    throw new Error("Use staged upload for this request");
  return { method, route, path: target, headers, body, size: body.length };
}
const cors = {
  "access-control-allow-origin": "*",
  "access-control-expose-headers":
    "x-mold-relay-object,x-mold-relay-protocol,content-range,content-length,x-mold-seed-used,x-mold-generation-time-ms,x-mold-gpu",
  "x-mold-relay-protocol": "2",
  "cache-control": "no-store",
};
export function createFrontend(dependencies = {}) {
  const deps = {
    store,
    request: openRequest,
    prepareUpload,
    consumeUpload,
    stage: stageObject,
    objectURL,
    shell: async (key) =>
      s3.send(
        new GetObjectCommand({
          Bucket: process.env.BUCKET_NAME,
          Key: `shell/${key}`,
        }),
      ),
    invoke: async (payload) =>
      lambda.send(
        new InvokeCommand({
          FunctionName: process.env.AWS_LAMBDA_FUNCTION_NAME,
          InvocationType: "Event",
          Payload: Buffer.from(JSON.stringify(payload)),
        }),
      ),
    ...dependencies,
  };
  const output = (raw, status, headers) => {
    const out =
      typeof globalThis.awslambda?.HttpResponseStream?.from === "function"
        ? awslambda.HttpResponseStream.from(raw, {
            statusCode: status,
            headers: { ...cors, ...headers },
          })
        : (raw.setMetadata?.(status, { ...cors, ...headers }), raw);
    out.write("");
    return out;
  };
  const json = (raw, status, value, headers = {}) => {
    const out = output(raw, status, {
      "content-type": "application/json",
      ...headers,
    });
    out.end(status === 204 ? undefined : JSON.stringify(value));
    return out;
  };
  async function authenticated(headers) {
    credentialDigest(headers);
    const result = await deps.request({ path: "/api/status", headers });
    try {
      let bytes = 0;
      for await (const chunk of result.response) {
        bytes += chunk.length;
        if (bytes > 1048576) throw new Error("Invalid host status response");
      }
      if (result.response.statusCode !== 200)
        throw new Error("Mold authentication refused");
      return result.sid;
    } finally {
      result.close();
    }
  }
  async function createMedia(request) {
    const sid = await authenticated(request.headers);
    const id = randomUUID(),
      expiresAt = Math.floor(Date.now() / 1000) + 900;
    await deps.store.put(`media#${id}`, {
      state: "pending",
      credential: credentialDigest(request.headers),
      sid,
      expiresAt,
    });
    await deps.invoke({
      kind: "stage",
      id,
      sid,
      request: { method: "GET", path: request.path, headers: request.headers },
    });
    return { relay: { id, state: "pending" }, expires_at: expiresAt };
  }
  async function stageWorker(event) {
    const job = await deps.store.get(`media#${event.id}`);
    if (
      !job ||
      job.state !== "pending" ||
      job.expiresAt <= Math.floor(Date.now() / 1000) ||
      job.sid !== event.sid
    )
      return;
    if (
      !(await deps.store.cas(`media#${event.id}`, job.revision, {
        ...job,
        state: "working",
      }))
    )
      return;
    let result;
    try {
      result = await deps.request(event.request);
      if (result.sid !== event.sid || result.response.statusCode !== 200)
        throw new Error("Media refused");
      const object = await deps.stage(
        result.response,
        cleanHeaders(result.response.headers, { response: true }),
      );
      const { url: unusedURL, ...facts } = object;
      await deps.store.put(`media#${event.id}`, {
        ...job,
        state: "ready",
        ...facts,
        expiresAt: object.expires_at,
      });
    } catch {
      await deps.store.put(`media#${event.id}`, { ...job, state: "failed" });
    } finally {
      result?.close();
    }
  }
  return async (event, raw, context = {}) => {
    if (event.kind === "stage") {
      await stageWorker(event);
      raw.end();
      return;
    }
    let result,
      cleanup,
      hasOutput = false;
    let mutationMayHaveExecuted = false;
    try {
      const request = normalizeEvent(event);
      if (request.method === "OPTIONS") {
        const requested =
          request.headers["access-control-request-headers"] ??
          "authorization,x-api-key,content-type,x-amz-content-sha256,x-mold-request-target";
        if (!/^[a-zA-Z0-9, _-]+$/.test(requested))
          throw new Error("Invalid CORS headers");
        json(
          raw,
          204,
          {},
          {
            "access-control-allow-methods":
              "GET,HEAD,POST,PUT,PATCH,DELETE,OPTIONS",
            "access-control-allow-headers": requested,
            "access-control-max-age": "600",
          },
        );
        return;
      }
      if (
        [request.route, request.path.split("?")[0]].some((path) =>
          decodeURIComponent(path).replace(/\/+$/, "").startsWith("/metrics"),
        )
      ) {
        json(raw, 404, { error: "Not found" });
        return;
      }
      if (request.route === "/_mold/relay/info") {
        json(raw, 200, {
          protocol: 2,
          upload_threshold: THRESHOLD,
          max_body_bytes: MAX_BODY,
          object_origin: objectOrigin(),
        });
        return;
      }
      const readTicket =
        ["GET", "HEAD"].includes(request.method) &&
        /^\/api\/gallery\/(image|thumbnail|preview|assets|source-media)\//.test(
          request.route,
        ) &&
        new URL(request.path, "https://mold.invalid").searchParams.has(
          "media_token",
        );
      const publicClaim =
        request.route === "/api/pairing/claim" && request.method === "POST";
      if (
        (request.route.startsWith("/api/") ||
          request.route.startsWith("/_mold/relay/")) &&
        !request.headers["x-api-key"] &&
        !readTicket &&
        !publicClaim
      ) {
        json(raw, 401, { error: "missing X-Api-Key header" });
        return;
      }
      if (
        request.route === "/_mold/relay/uploads" &&
        request.method === "POST"
      ) {
        const sid = await authenticated(request.headers);
        json(
          raw,
          200,
          await deps.prepareUpload(
            JSON.parse(request.body.toString()),
            request.headers,
            sid,
          ),
        );
        return;
      }
      if (
        request.route === "/_mold/relay/request" &&
        request.method === "POST"
      ) {
        const sid = await authenticated(request.headers),
          entry = await deps.consumeUpload(
            JSON.parse(request.body.toString()).id,
            request.headers,
            sid,
          );
        cleanup = entry.cleanup;
        mutationMayHaveExecuted = !["GET", "HEAD", "OPTIONS"].includes(
          entry.method,
        );
        result = await deps.request({
          method: entry.method,
          path: entry.path,
          headers: {
            ...entry.headers,
            ...Object.fromEntries(
              Object.entries(request.headers).filter(([k]) =>
                ["authorization", "x-api-key"].includes(k),
              ),
            ),
          },
          body: entry.body,
          size: entry.size,
        });
        if (result.sid !== sid) throw new Error("Host session changed");
      } else if (
        request.route === "/_mold/relay/media" &&
        request.method === "POST"
      ) {
        const { path } = JSON.parse(request.body.toString());
        if (!safeTarget(path) || !path.startsWith("/api/"))
          throw new Error("Invalid media path");
        json(raw, 202, await createMedia({ ...request, path }));
        return;
      } else if (
        request.route.startsWith("/_mold/relay/media/") &&
        request.method === "GET"
      ) {
        const id = request.route.slice("/_mold/relay/media/".length);
        if (!/^[a-f0-9-]{36}$/.test(id))
          throw new Error("Invalid media identity");
        const job = await deps.store.get(`media#${id}`);
        if (
          !job ||
          job.expiresAt <= Math.floor(Date.now() / 1000) ||
          job.credential !== credentialDigest(request.headers)
        )
          throw new Error("Media grant refused");
        json(raw, 200, {
          state: job.state === "working" ? "pending" : job.state,
          ...(job.state === "ready"
            ? {
                url: await deps.objectURL(
                  job.key,
                  Math.max(
                    1,
                    Math.min(
                      900,
                      job.expiresAt - Math.floor(Date.now() / 1000),
                    ),
                  ),
                ),
                expires_at: job.expires_at,
              }
            : {}),
        });
        return;
      } else if (request.route.startsWith("/_mold/")) {
        json(raw, 404, { error: "Not found" });
        return;
      } else if (
        ["GET", "HEAD"].includes(request.method) &&
        !request.route.startsWith("/api/")
      ) {
        const asset =
          /^\/(assets|fonts)\//.test(request.route) ||
          /^\/(logo\.png|favicon\.ico|icon.*\.png)$/.test(request.route);
        const shell = await deps.shell(
          asset ? request.route.slice(1) : "index.html",
        );
        const out = output(raw, 200, {
          "content-type":
            shell.ContentType ??
            (asset ? "application/octet-stream" : "text/html"),
        });
        hasOutput = true;
        if (request.method === "HEAD") out.end();
        else await pipeline(shell.Body, out);
        return;
      } else {
        mutationMayHaveExecuted = !["GET", "HEAD", "OPTIONS"].includes(
          request.method,
        );
        result = await deps.request(request);
      }
      const response = result.response,
        headers = cleanHeaders(response.headers, { response: true });
      if (
        request.route === "/api/gallery/media-token" &&
        request.method === "POST" &&
        response.statusCode === 200
      ) {
        const chunks = [];
        let bytes = 0;
        for await (const b of response) {
          bytes += b.length;
          if (bytes > 65536) throw new Error("Invalid media ticket response");
          chunks.push(b);
        }
        const ticket = JSON.parse(Buffer.concat(chunks).toString()),
          original = JSON.parse(request.body.toString());
        if (
          ticket.auth_required !== false &&
          ticket.token &&
          safeTarget(original.path)
        ) {
          const path = new URL(original.path, "https://mold.invalid");
          path.searchParams.set("media_token", ticket.token);
          path.searchParams.set("expires", String(ticket.expires_at));
          const pending = await createMedia({
            ...request,
            path: path.pathname + path.search,
          });
          json(raw, 200, { ...ticket, ...pending });
          return;
        }
        json(raw, 200, ticket);
        return;
      }
      const length = Number(headers["content-length"]);
      const eventStream = String(headers["content-type"] ?? "").startsWith(
        "text/event-stream",
      );
      if (
        request.method !== "HEAD" &&
        ![204, 304].includes(response.statusCode) &&
        !eventStream &&
        (!Number.isFinite(length) || length > OBJECT_THRESHOLD)
      ) {
        const object = await deps.stage(response, headers);
        json(
          raw,
          200,
          { url: object.url, status: response.statusCode, headers },
          { "x-mold-relay-object": "1" },
        );
        return;
      }
      if (eventStream) delete headers["content-length"];
      const out = output(raw, response.statusCode ?? 502, headers);
      hasOutput = true;
      if (
        request.method === "HEAD" ||
        [204, 304].includes(response.statusCode)
      ) {
        response.resume();
        out.end();
        return;
      }
      if (eventStream) {
        let deadline = false;
        const timer = setTimeout(
          () => {
            deadline = true;
            out.end(": relay reconnect\n\n");
            result.close();
          },
          Math.min(
            828000,
            Math.max(
              1000,
              (context.getRemainingTimeInMillis?.() ?? 850000) - 20000,
            ),
          ),
        );
        timer.unref();
        try {
          await pipeline(response, out);
        } catch (error) {
          if (!deadline) throw error;
        } finally {
          clearTimeout(timer);
        }
      } else await pipeline(response, out);
    } catch (error) {
      console.error("Mold relay frontend failure", failureCategory(error));
      if (!hasOutput)
        json(
          raw,
          503,
          {
            error: "Relay request unavailable or refused",
            request_state: mutationMayHaveExecuted
              ? "outcome-unknown"
              : "not-forwarded",
          },
          {
            "x-mold-relay-request-state": mutationMayHaveExecuted
              ? "outcome-unknown"
              : "not-forwarded",
          },
        );
      else raw.destroy?.(new Error("Relay stream interrupted"));
    } finally {
      result?.close();
      await cleanup?.().catch(() => {});
    }
  };
}
const serve = createFrontend();
export const handler =
  typeof globalThis.awslambda?.streamifyResponse === "function"
    ? awslambda.streamifyResponse(serve)
    : serve;

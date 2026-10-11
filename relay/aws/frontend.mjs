// Protocol 3 has no plaintext application ingress. This Lambda returns public
// discovery metadata, a trusted Studio redirect, or an explicit upgrade error.
// It has no DynamoDB, S3, SSM, Lambda invocation or host API dependencies.
const STUDIO = "https://utensils.io/mold/studio/";
function configuration(value) {
  const config = value ?? {
    websocket_url: process.env.WS_ENDPOINT,
    host_domain: process.env.MANAGED_HOST_DOMAIN,
    studio_url: STUDIO,
  };
  const ws = new URL(config.websocket_url);
  if (
    ws.protocol !== "wss:" ||
    ws.username ||
    ws.password ||
    ws.search ||
    ws.hash ||
    !ws.hostname.endsWith(".amazonaws.com") ||
    !/^[a-z0-9][a-z0-9.-]*[a-z0-9]$/.test(config.host_domain ?? "") ||
    config.host_domain.includes("..") ||
    config.studio_url !== STUDIO
  )
    throw new Error("Invalid public relay configuration");
  return {
    protocol: 3,
    websocket_url: ws.toString(),
    host_domain: config.host_domain,
    studio_url: STUDIO,
  };
}
const headers = {
  "content-type": "application/json",
  "cache-control": "no-store",
  "x-mold-relay-protocol": "3",
  "access-control-allow-origin": "*",
  "access-control-expose-headers": "x-mold-relay-protocol",
  "x-content-type-options": "nosniff",
  "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
};
function output(raw, status, extra, body, method) {
  const metadata = { statusCode: status, headers: { ...headers, ...extra } };
  const stream =
    typeof globalThis.awslambda?.HttpResponseStream?.from === "function"
      ? globalThis.awslambda.HttpResponseStream.from(raw, metadata)
      : (raw.setMetadata?.(status, metadata.headers), raw);
  stream.end(method === "HEAD" || status === 204 ? undefined : body);
}
export function createFrontend({ config } = {}) {
  const info = configuration(config);
  return async (event, raw) => {
    const method =
      event.requestContext?.http?.method ?? event.httpMethod ?? "GET";
    const path = event.rawPath ?? event.path ?? "/";
    const discovery = path === "/_mold/relay/info" && method === "GET";
    const enrollment = path === "/_mold/relay/enroll" && method === "POST";
    if (discovery || enrollment)
      return output(raw, 200, {}, JSON.stringify(info), method);
    if (
      method === "OPTIONS" &&
      (path === "/_mold/relay/info" || path === "/_mold/relay/enroll")
    )
      return output(
        raw,
        204,
        {
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers": "content-type",
        },
        undefined,
        method,
      );
    if (path === "/" && (method === "GET" || method === "HEAD"))
      return output(raw, 303, { location: STUDIO }, "", method);
    return output(
      raw,
      426,
      {},
      JSON.stringify({
        error: "secure-client-required",
        protocol: 3,
        message:
          "Update Mold and pair this device using the encrypted connection.",
      }),
      method,
    );
  };
}
const dispatch = async (event, raw) => {
  try {
    return await createFrontend()(event, raw);
  } catch {
    return output(
      raw,
      503,
      {},
      JSON.stringify({ error: "relay-unavailable" }),
      event.httpMethod,
    );
  }
};
export const handler =
  typeof globalThis.awslambda?.streamifyResponse === "function"
    ? globalThis.awslambda.streamifyResponse(dispatch)
    : dispatch;

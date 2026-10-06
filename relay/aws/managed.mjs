import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { isIP } from "node:net";
export const ROSTER = "managed-roster";
export const validNamespace = (value) =>
  typeof value === "string" && /^[a-f0-9]{32}$/.test(value);
export const namespaceKey = (kind, namespace = "") =>
  namespace ? `${kind}#tenant#${namespace}` : kind;
const digest = (value) => createHash("sha256").update(value).digest("hex");
export function managedConfig(env = process.env) {
  if (!env.MANAGED_HOST_DOMAIN || !env.PUBLIC_ORIGIN || !env.WS_ENDPOINT)
    return undefined;
  const origin = new URL(env.PUBLIC_ORIGIN),
    ws = new URL(env.WS_ENDPOINT),
    domain = env.MANAGED_HOST_DOMAIN;
  if (
    origin.protocol !== "https:" ||
    origin.username ||
    origin.password ||
    origin.pathname !== "/" ||
    origin.search ||
    origin.hash ||
    origin.port ||
    !/^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$/.test(domain) ||
    domain.length > 220 ||
    ws.protocol !== "wss:" ||
    ws.username ||
    ws.password ||
    ws.search ||
    ws.hash ||
    ws.port
  )
    throw new Error("Invalid managed relay configuration");
  return { domain, origin: origin.origin, relayURL: ws.href };
}
export async function activeEnrollment(
  store,
  id,
  now = Math.floor(Date.now() / 1000),
) {
  if (!validNamespace(id)) return undefined;
  const entry = (await store.get(ROSTER))?.hosts?.[id];
  return entry?.expiresAt > now ? entry : undefined;
}
export async function ownerAllowed(store, id, authorization, now) {
  const entry = await activeEnrollment(store, id, now);
  if (
    !entry ||
    typeof authorization !== "string" ||
    !/^Bearer [A-Za-z0-9_-]{43}$/.test(authorization)
  )
    return false;
  const actual = digest(authorization.slice(7));
  return (
    typeof entry.verifier === "string" &&
    entry.verifier.length === 64 &&
    timingSafeEqual(Buffer.from(actual), Buffer.from(entry.verifier))
  );
}
export async function resolveNamespace({ domainName }, store, config, now) {
  if (!config) return "";
  if (domainName === new URL(config.origin).hostname) return "";
  if (
    typeof domainName !== "string" ||
    !domainName.endsWith("." + config.domain)
  )
    throw new Error("Unknown relay origin");
  const id = domainName.slice(0, -config.domain.length - 1);
  if (!(await activeEnrollment(store, id, now)))
    throw new Error("Unknown relay host");
  return id;
}
export function createManagedEnrollment({
  store,
  config,
  now = () => Math.floor(Date.now() / 1000),
}) {
  return async (request) => {
    if (!config)
      return {
        status: 503,
        value: {
          error:
            "Managed phone pairing is unavailable; configure the managed gateway",
        },
      };
    if (request.domainName !== new URL(config.origin).hostname)
      return { status: 404, value: { error: "Not found" } };
    const match = /^\/_mold\/relay\/enroll(?:\/([a-f0-9]{32}))?$/.exec(
      request.route,
    );
    if (
      !match ||
      !(match[1] ? ["POST", "DELETE"] : ["POST"]).includes(request.method)
    )
      return { status: 404, value: { error: "Not found" } };
    const id = match[1],
      time = now();
    if (!isIP(request.sourceIP ?? ""))
      return {
        status: 403,
        value: { error: "Trusted source address required" },
      };
    const source = digest(request.sourceIP),
      fresh = id
        ? undefined
        : {
            id: randomBytes(16).toString("hex"),
            token: randomBytes(32).toString("base64url"),
          };
    for (let attempt = 0; attempt < 64; attempt++) {
      const old = await store.get(ROSTER);
      const hosts = Object.fromEntries(
        Object.entries(old?.hosts ?? {}).filter(([, v]) => v.expiresAt > time),
      );
      const sources = Object.fromEntries(
        Object.entries(old?.sources ?? {}).filter(
          ([, v]) => v.expiresAt > time,
        ),
      );
      let status, value;
      if (id) {
        const host = hosts[id];
        if (
          !host ||
          !(await ownerAllowed(
            { get: async () => ({ hosts }) },
            id,
            request.headers.authorization,
            time,
          ))
        )
          return { status: 403, value: { error: "Enrollment owner refused" } };
        if (request.method === "DELETE") {
          delete hosts[id];
          status = 204;
          value = {};
        } else {
          hosts[id] = { ...host, expiresAt: time + 30 * 86400 };
          status = 200;
          value = {
            host_id: id,
            token: request.headers.authorization.slice(7),
            public_url: `https://${id}.${config.domain}`,
            relay_url: config.relayURL,
            expires_at: hosts[id].expiresAt,
          };
        }
      } else {
        if (
          (sources[source]?.count ?? 0) >= 5 ||
          (!sources[source] && Object.keys(sources).length >= 1024)
        )
          return {
            status: 429,
            value: { error: "Enrollment rate limit reached" },
          };
        if (Object.keys(hosts).length >= 32)
          return {
            status: 503,
            value: { error: "Managed relay capacity reached" },
          };
        hosts[fresh.id] = {
          verifier: digest(fresh.token),
          expiresAt: time + 30 * 86400,
        };
        sources[source] = {
          count: (sources[source]?.count ?? 0) + 1,
          expiresAt: sources[source]?.expiresAt ?? time + 3600,
        };
        status = 201;
        value = {
          host_id: fresh.id,
          token: fresh.token,
          public_url: `https://${fresh.id}.${config.domain}`,
          relay_url: config.relayURL,
          expires_at: hosts[fresh.id].expiresAt,
        };
      }
      const expiresAt = Math.max(
        time + 3600,
        ...Object.values(hosts).map((v) => v.expiresAt),
        ...Object.values(sources).map((v) => v.expiresAt),
      );
      if (
        await store.cas(ROSTER, old?.revision ?? 0, {
          hosts,
          sources,
          expiresAt,
        })
      )
        return { status, value };
    }
    return { status: 503, value: { error: "Managed enrollment busy; retry" } };
  };
}

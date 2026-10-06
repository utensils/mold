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
export const OWNER_PREFIX = "managed-owner#";
const PROVISIONAL = 120,
  LIVE = 90,
  OWNER_LEASE = 30 * 86400;
const BURST = 8,
  REFILL_SECONDS = 30;
const liveSlots = (roster, time) =>
  Object.fromEntries(
    Object.entries(roster?.hosts ?? {}).filter(
      ([, value]) => value.expiresAt > time,
    ),
  );
async function ownerRecord(store, id, time) {
  if (!validNamespace(id)) return undefined;
  let owner = await store.get(OWNER_PREFIX + id);
  // Migrate the previously shipped inline owner verifier without changing identity.
  if (!owner) {
    const legacy = (await store.get(ROSTER))?.hosts?.[id];
    if (legacy?.verifier && legacy.expiresAt > time) {
      await store.cas(OWNER_PREFIX + id, 0, {
        verifier: legacy.verifier,
        established: true,
        expiresAt: legacy.expiresAt,
      });
      owner = await store.get(OWNER_PREFIX + id);
    }
  }
  return !owner?.revoked && owner?.expiresAt > time ? owner : undefined;
}
export async function activeEnrollment(
  store,
  id,
  now = Math.floor(Date.now() / 1000),
) {
  const owner = await ownerRecord(store, id, now);
  if (!owner) return undefined;
  const slot = (await store.get(ROSTER))?.hosts?.[id];
  if (!slot || slot.expiresAt <= now) return undefined;
  // Old inline owners held30-day slots. Only a still-live host can retain one.
  if (slot.verifier) {
    const host = await store.get(namespaceKey("host", id));
    if (!host || host.expiresAt <= now) return undefined;
  }
  return owner;
}
export async function ownerAllowed(
  store,
  id,
  authorization,
  now = Math.floor(Date.now() / 1000),
) {
  const entry = await ownerRecord(store, id, now);
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
async function capacitySlots(store, old, time) {
  const hosts = liveSlots(old, time);
  for (const [id, slot] of Object.entries(hosts)) {
    if (!slot.verifier) continue;
    // Preserve old owner secrets in individual TTL records before reclaiming slots.
    await ownerRecord(store, id, time);
    const host = await store.get(namespaceKey("host", id));
    if (!host || host.expiresAt <= time) delete hosts[id];
    else hosts[id] = { expiresAt: Math.min(host.expiresAt, time + LIVE) };
  }
  return hosts;
}
export async function reserveManagedSlot(
  store,
  id,
  time = Math.floor(Date.now() / 1000),
) {
  if (!(await ownerRecord(store, id, time))) return false;
  for (let attempt = 0; attempt < 64; attempt++) {
    const old = await store.get(ROSTER),
      hosts = await capacitySlots(store, old, time);
    if (!hosts[id] && Object.keys(hosts).length >= 32) return false;
    // Repeated handshakes cannot extend an unproven reservation.
    hosts[id] ??= { expiresAt: time + PROVISIONAL };
    if (
      await store.cas(ROSTER, old?.revision ?? 0, {
        ...old,
        hosts,
        expiresAt: Math.max(old?.expiresAt ?? 0, time + 3600),
      })
    )
      return true;
  }
  return false;
}
export async function promoteManagedHost(
  store,
  id,
  time = Math.floor(Date.now() / 1000),
) {
  for (let attempt = 0; attempt < 64; attempt++) {
    const owner = await ownerRecord(store, id, time);
    if (!owner) return false;
    // Refresh identity at most once a day; heartbeats renew capacity every30s.
    if (owner.established && owner.expiresAt > time + OWNER_LEASE - 86400)
      break;
    if (
      await store.cas(OWNER_PREFIX + id, owner.revision, {
        ...owner,
        established: true,
        expiresAt: time + OWNER_LEASE,
      })
    )
      break;
    if (attempt === 63) return false;
  }
  for (let attempt = 0; attempt < 64; attempt++) {
    const old = await store.get(ROSTER),
      hosts = await capacitySlots(store, old, time);
    if (!hosts[id]) return false;
    hosts[id] = { expiresAt: time + LIVE };
    if (
      await store.cas(ROSTER, old?.revision ?? 0, {
        ...old,
        hosts,
        expiresAt: Math.max(old?.expiresAt ?? 0, time + 3600),
      })
    )
      return true;
  }
  return false;
}
async function releaseManagedSlot(store, id, time) {
  for (let attempt = 0; attempt < 64; attempt++) {
    const old = await store.get(ROSTER),
      hosts = await capacitySlots(store, old, time);
    delete hosts[id];
    const sources = Object.fromEntries(
      Object.entries(old?.sources ?? {}).filter(([, v]) => v.expiresAt > time),
    );
    if (
      await store.cas(ROSTER, old?.revision ?? 0, {
        ...old,
        hosts,
        sources,
        expiresAt: Math.max(old?.expiresAt ?? 0, time + 3600),
      })
    )
      return true;
  }
  return false;
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
    if (id) {
      for (let attempt = 0; attempt < 64; attempt++) {
        const owner = await ownerRecord(store, id, time);
        if (
          !owner ||
          !(await ownerAllowed(store, id, request.headers.authorization, time))
        )
          return { status: 403, value: { error: "Enrollment owner refused" } };
        const expiresAt =
          request.method === "DELETE" ? time + OWNER_LEASE : owner.expiresAt;
        if (
          !(await store.cas(OWNER_PREFIX + id, owner.revision, {
            ...owner,
            expiresAt,
            ...(request.method === "DELETE" ? { revoked: true } : {}),
          }))
        )
          continue;
        if (
          !(await releaseManagedSlotOnExpiry(
            store,
            id,
            time,
            request.method === "DELETE",
          ))
        )
          return {
            status: 503,
            value: { error: "Managed enrollment busy; retry" },
          };
        return request.method === "DELETE"
          ? { status: 204, value: {} }
          : {
              status: 200,
              value: {
                host_id: id,
                token: request.headers.authorization.slice(7),
                public_url: `https://${id}.${config.domain}`,
                relay_url: config.relayURL,
                expires_at: expiresAt,
              },
            };
      }
      return {
        status: 503,
        value: { error: "Managed enrollment busy; retry" },
      };
    }
    for (let attempt = 0; attempt < 64; attempt++) {
      const old = await store.get(ROSTER),
        hosts = await capacitySlots(store, old, time);
      const sources = Object.fromEntries(
        Object.entries(old?.sources ?? {}).filter(
          ([, v]) => v.expiresAt > time,
        ),
      );
      const budget = {
        tokens: Math.min(
          BURST,
          (old?.budget?.tokens ?? BURST) +
            Math.floor(
              Math.max(0, time - (old?.budget?.updatedAt ?? time)) /
                REFILL_SECONDS,
            ),
        ),
        updatedAt: time,
      };
      if (
        (sources[source]?.count ?? 0) >= 5 ||
        (!sources[source] && Object.keys(sources).length >= 1024) ||
        budget.tokens < 1
      )
        return {
          status: 429,
          value: { error: "Enrollment rate limit reached; retry shortly" },
        };
      if (Object.keys(hosts).length >= 32)
        return {
          status: 503,
          value: { error: "Managed relay capacity reached" },
        };
      budget.tokens--;
      // Preserve fractional refill time rather than discarding it on each admission.
      budget.updatedAt =
        budget.tokens === BURST - 1
          ? time
          : (old?.budget?.updatedAt ?? time) +
            Math.floor(
              Math.max(0, time - (old?.budget?.updatedAt ?? time)) /
                REFILL_SECONDS,
            ) *
              REFILL_SECONDS;
      hosts[fresh.id] = { expiresAt: time + PROVISIONAL };
      sources[source] = {
        count: (sources[source]?.count ?? 0) + 1,
        expiresAt: sources[source]?.expiresAt ?? time + 3600,
      };
      const expiresAt = Math.max(
        time + 3600,
        ...Object.values(hosts).map((v) => v.expiresAt),
        ...Object.values(sources).map((v) => v.expiresAt),
      );
      if (
        !(await store.cas(ROSTER, old?.revision ?? 0, {
          hosts,
          sources,
          budget,
          expiresAt,
        }))
      )
        continue;
      // Failed persistence leaves only a short provisional slot and no returned credentials.
      if (
        !(await store.cas(OWNER_PREFIX + fresh.id, 0, {
          verifier: digest(fresh.token),
          established: false,
          expiresAt: time + PROVISIONAL,
        }))
      )
        return {
          status: 503,
          value: { error: "Managed enrollment busy; retry" },
        };
      return {
        status: 201,
        value: {
          host_id: fresh.id,
          token: fresh.token,
          public_url: `https://${fresh.id}.${config.domain}`,
          relay_url: config.relayURL,
          expires_at: time + PROVISIONAL,
        },
      };
    }
    return { status: 503, value: { error: "Managed enrollment busy; retry" } };
  };
}
async function releaseManagedSlotOnExpiry(store, id, time, remove) {
  if (remove) return releaseManagedSlot(store, id, time);
  for (let attempt = 0; attempt < 64; attempt++) {
    const old = await store.get(ROSTER),
      hosts = await capacitySlots(store, old, time);
    const sources = Object.fromEntries(
      Object.entries(old?.sources ?? {}).filter(([, v]) => v.expiresAt > time),
    );
    if (
      await store.cas(ROSTER, old?.revision ?? 0, {
        ...old,
        hosts,
        sources,
        expiresAt: Math.max(
          time + 3600,
          ...Object.values(hosts).map((v) => v.expiresAt),
          ...Object.values(sources).map((v) => v.expiresAt),
        ),
      })
    )
      return true;
  }
  return false;
}

import { randomBytes } from "node:crypto";
import { hex, integer } from "./secure-admission.mjs";

export const OBJECT_LIMITS = Object.freeze({
  version: 1,
  maxBytes: 1048576,
  maxCount: 64,
  maxInFlightBytes: 1048576,
  maxInFlightObjects: 4,
});
const rowKey = (c) =>
  `v3-objects#${c.namespace}#${c.sid}#${c.rid}#${c.direction}`;
// Raw S3 key: the SDK encodes its URL path exactly once. Pre-encoding here
// would turn API Gateway's '=' connection IDs into '%253D' in presigned URLs.
export const objectPath = (c, id) =>
  `ciphertext/v3/${c.namespace}/${c.sid}/${c.rid}/${c.direction}/${id}`;
const validSize = (f) =>
  integer(f.seq) &&
  integer(f.count) &&
  f.count >= 1 &&
  f.count <= OBJECT_LIMITS.maxCount &&
  Number.isSafeInteger(f.seq + f.count) &&
  integer(f.bytes) &&
  f.bytes >= f.count * 5 &&
  f.bytes <= Math.min(OBJECT_LIMITS.maxBytes, f.count * 16388) &&
  hex(f.sha256, 32);
export function objectEnvelope(frame) {
  const o = frame?.o;
  if (
    frame.d !== undefined ||
    !o ||
    !hex(o.id, 16) ||
    !validSize({ ...o, seq: frame.seq }) ||
    typeof o.url !== "string" ||
    o.url.length > 4096
  )
    return;
  return {
    a: "data",
    v: 3,
    sid: frame.sid,
    rid: frame.rid,
    seq: frame.seq,
    o: {
      id: o.id,
      bytes: o.bytes,
      sha256: o.sha256,
      url: o.url,
      count: o.count,
    },
  };
}
export function createObjectBroker({
  store,
  storage,
  now = () => Math.floor(Date.now() / 1000),
}) {
  const policy = storage
    ? { version: 1, origin: storage.origin, ...OBJECT_LIMITS }
    : undefined;
  const response = (f, op, fields = {}) => ({
    a: "object",
    v: 3,
    op,
    requestId: f.requestId,
    sid: f.sid,
    rid: f.rid,
    ...fields,
  });
  const error = (f, code) => response(f, "error", { code });
  async function mutate(c, change, initial) {
    for (let n = 0; n < 32; n++) {
      const old = n === 0 && initial ? initial : await store.get(rowKey(c));
      const next = change(old ?? { members: {}, next: 0 });
      if (!next) return;
      if (
        await store.cas(rowKey(c), old?.revision ?? 0, {
          ...next,
          expiresAt: now() + 3600,
        })
      )
        return next;
    }
    throw new Error("ObjectContention");
  }
  async function cleanup(c, predicate, ack) {
    const next = ack?.next;
    const row = await mutate(c, (old) => {
      const members = { ...old.members };
      let changed = false;
      for (const [id, m] of Object.entries(members))
        if (predicate(m) || m.deleting) {
          members[id] = { ...m, deleting: true };
          changed = true;
        }
      if (!changed && (next === undefined || next <= old.next)) return;
      return {
        ...old,
        members,
        next: Math.max(old.next, next ?? 0),
        ...(next !== undefined && next >= old.next ? { ack } : {}),
      };
    });
    if (!row) return;
    const ids = Object.entries(row.members)
      .filter(([, m]) => m.deleting)
      .map(([id]) => id);
    await Promise.all(ids.map((id) => storage.clear(objectPath(c, id))));
    if (ids.length)
      await mutate(c, (old) => {
        const members = { ...old.members };
        for (const id of ids) if (members[id]?.deleting) delete members[id];
        return { ...old, members };
      });
  }
  async function command(c, f) {
    if (!hex(f.requestId, 16) || f.sid !== c.sid || f.rid !== c.rid)
      return undefined;
    if (!storage) return error(f, "disabled");
    if (f.op === "release") {
      if (!hex(f.id, 16)) return;
      const row = await store.get(rowKey(c)),
        m = row?.members?.[f.id];
      if (m && (m.sender !== c.sender || m.forwarded))
        return error(f, "conflict");
      await cleanup(
        c,
        (x) => x.id === f.id && !x.forwarded && x.sender === c.sender,
      );
      if ((await store.get(rowKey(c)))?.members?.[f.id])
        return error(f, "conflict");
      return response(f, "released", { id: f.id });
    }
    if (f.op !== "create" || !validSize(f)) return;
    // Presigning is local CPU, not an S3 operation. Store the exact reply in
    // the same conditional write that reserves bytes, so four parallel
    // allocations contend once instead of twice on their shared quota row.
    const member = {
      id: randomBytes(16).toString("hex"),
      requestId: f.requestId,
      sender: c.sender,
      seq: f.seq,
      count: f.count,
      bytes: f.bytes,
      sha256: f.sha256,
      expiresAt: now() + 300,
    };
    const grants = await storage.grant(objectPath(c, member.id), member);
    const reply = response(f, "created", {
      id: member.id,
      seq: member.seq,
      count: member.count,
      bytes: member.bytes,
      sha256: member.sha256,
      ...grants,
      expiresAt: member.expiresAt,
    });
    for (let attempt = 0; attempt < 3; attempt++) {
      let result,
        needsCleanup = false,
        allocated = false;
      await mutate(c, (old) => {
        // Reset on a failed CAS retry: only its winning snapshot is authority.
        allocated = false;
        needsCleanup = false;
        const entries = Object.values(old.members);
        if (entries.some((m) => m.deleting || m.expiresAt <= now())) {
          needsCleanup = true;
          return;
        }
        const existing = entries.find((m) => m.requestId === f.requestId);
        if (existing) {
          result =
            existing.sender === c.sender &&
            ["seq", "count", "bytes", "sha256"].every(
              (k) => existing[k] === f[k],
            )
              ? existing.reply
              : error(f, "conflict");
          return;
        }
        if (
          f.seq < old.next ||
          entries.some(
            (m) => f.seq < m.seq + m.count && m.seq < f.seq + f.count,
          )
        ) {
          result = error(f, "conflict");
          return;
        }
        if (
          entries.length >= 4 ||
          entries.reduce((sum, m) => sum + m.bytes, 0) + f.bytes > 1048576
        ) {
          result = error(f, "capacity");
          return;
        }
        allocated = true;
        result = reply;
        return {
          ...old,
          members: { ...old.members, [member.id]: { ...member, reply } },
        };
      });
      if (needsCleanup) {
        await cleanup(c, (m) => m.expiresAt <= now());
        continue;
      }
      if (allocated)
        console.info(
          "Mold secure relay object allocated",
          c.namespace,
          member.id,
          c.direction,
          member.bytes,
          member.count,
        );
      return result ?? error(f, "unavailable");
    }
    return error(f, "unavailable");
  }

  async function forward(c, frame) {
    if (!storage) return { code: "disabled" };
    const f = objectEnvelope(frame);
    if (!f) return { code: "conflict" };
    const row = await store.get(rowKey(c)),
      m = row?.members?.[f.o.id];
    // A lost cumulative ACK can cause a legitimate reference retry after its
    // object payload was cleared. Resend the recipient's exact recorded ACK; never
    // fetch or re-forward an already consumed object, or invent new ack_seq.
    if (f.seq + f.o.count <= row?.next && row.ack) return { ack: row.ack };
    if (
      !m ||
      m.sender !== c.sender ||
      m.deleting ||
      m.seq !== f.seq ||
      !["count", "bytes", "sha256"].every((k) => m[k] === f.o[k]) ||
      m.reply?.getUrl !== f.o.url
    )
      return { code: "conflict" };
    if (m.expiresAt <= now()) return { code: "expired" };
    if (m.forwarded && m.verified) return { frame: f };
    const metadata = await storage.head(objectPath(c, m.id));
    if (!metadata) return { code: "not_ready" };
    if (metadata.bytes !== m.bytes || metadata.sha256 !== m.sha256)
      return { code: "conflict" };
    let valid = false;
    await mutate(
      c,
      (old) => {
        valid = false;
        const current = old.members[m.id];
        if (!current || current.deleting || current.expiresAt <= now()) return;
        valid = true;
        if (current.forwarded && current.verified) return;
        return {
          ...old,
          members: {
            ...old.members,
            [m.id]: { ...current, forwarded: true, verified: true },
          },
        };
      },
      row,
    );
    if (!valid) return { code: "expired" };
    console.info(
      "Mold secure relay object verified",
      c.namespace,
      m.id,
      c.direction,
      m.bytes,
      m.count,
    );
    return { frame: f };
  }
  return {
    policy,
    async command(c, f) {
      try {
        return await command(c, f);
      } catch {
        return error(f, "unavailable");
      }
    },
    forward,
    async ack(c, ack) {
      if (storage && (await store.get(rowKey(c))))
        await cleanup(
          c,
          (m) => m.forwarded && m.seq + m.count <= ack.next,
          ack,
        );
    },
    async cancel(c) {
      if (storage) await cleanup(c, () => true);
    },
  };
}

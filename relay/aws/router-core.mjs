import { randomUUID, timingSafeEqual } from "node:crypto";
export const FRAME_LIMIT = 24 * 1024;
export function failureCategory(error) {
  return /^[A-Za-z][A-Za-z0-9]{0,63}$/.test(error?.name ?? "")
    ? error.name
    : "Error";
}
const LIVE = 90,
  CAPACITY = 32;
const equal = (a, b) =>
  typeof a === "string" &&
  typeof b === "string" &&
  Buffer.byteLength(a) === Buffer.byteLength(b) &&
  timingSafeEqual(Buffer.from(a), Buffer.from(b));
const validInt = (n) => Number.isSafeInteger(n) && n >= 0;
export function validFrame(f) {
  if (f?.v !== 2 || typeof f.sid !== "string" || typeof f.rid !== "string")
    return false;
  if (f.a === "data")
    return (
      validInt(f.seq) &&
      typeof f.d === "string" &&
      Buffer.byteLength(f.d) <= 21848 &&
      Buffer.from(f.d, "base64").length <= 16384 &&
      Buffer.from(f.d, "base64").toString("base64") === f.d
    );
  if (f.a === "eof") return validInt(f.seq);
  if (f.a === "ack")
    return (
      validInt(f.next) &&
      validInt(f.credit) &&
      f.credit <= 4 &&
      validInt(f.ack_seq)
    );
  return f.a === "accept" || f.a === "cancel";
}
export function createRouter({
  store,
  tokens,
  post,
  close,
  now = () => Math.floor(Date.now() / 1000),
}) {
  async function updateHost(change) {
    for (let attempt = 0; attempt < 64; attempt++) {
      const old = await store.get("host");
      const next = change(old);
      if (!next) return false;
      if (await store.cas("host", old?.revision ?? 0, next)) return next;
    }
    return false;
  }
  async function connect(id, headers) {
    const role = headers["x-mold-relay-role"];
    const expected = (await tokens())[role];
    if (
      !["host", "frontend"].includes(role) ||
      !equal(headers.authorization, `Bearer ${expected}`)
    )
      return 403;
    if (role === "host") {
      const host = await updateHost((old) =>
        old?.expiresAt > now()
          ? null
          : {
              connectionId: id,
              sid: randomUUID(),
              guests: {},
              expiresAt: now() + LIVE,
            },
      );
      if (!host) return 409;
      await store.put(`connection#${id}`, {
        role,
        sid: host.sid,
        expiresAt: now() + LIVE,
      });
      return 200;
    }
    const host = await updateHost((old) => {
      if (!old || old.expiresAt <= now()) return null;
      const guests = Object.fromEntries(
        Object.entries(old.guests ?? {}).filter(
          ([, expires]) => expires > now(),
        ),
      );
      if (Object.keys(guests).length >= CAPACITY) return null;
      guests[id] = now() + LIVE;
      return { ...old, guests };
    });
    if (!host) return 503;
    try {
      await store.put(`connection#${id}`, {
        role,
        sid: host.sid,
        expiresAt: now() + LIVE,
      });
    } catch (error) {
      await updateHost((old) => {
        if (old?.sid !== host.sid) return null;
        const guests = { ...old.guests };
        delete guests[id];
        return { ...old, guests };
      });
      throw error;
    }
    return 200;
  }
  async function leave(id) {
    const connection = await store.get(`connection#${id}`);
    await store.remove(`connection#${id}`);
    if (!connection) return;
    let affected;
    await updateHost((old) => {
      if (old?.sid !== connection.sid) return null;
      if (connection.role === "host" && old.connectionId === id) {
        affected = Object.keys(old.guests ?? {});
        return { ...old, expiresAt: 0, guests: {} };
      }
      if (connection.role === "frontend") {
        const guests = { ...old.guests };
        delete guests[id];
        affected = [old.connectionId];
        return { ...old, guests };
      }
      return null;
    });
    for (const peer of affected ?? []) {
      try {
        await post(peer, {
          a: "cancel",
          v: 2,
          sid: connection.sid,
          rid: connection.role === "frontend" ? id : peer,
        });
      } catch {
        /* Stale peer cleanup is bounded by leases. */
      }
    }
  }
  async function message(id, body) {
    if (typeof body !== "string" || Buffer.byteLength(body) > FRAME_LIMIT)
      return 400;
    let frame;
    try {
      frame = JSON.parse(body);
    } catch {
      return 400;
    }
    const connection = await store.get(`connection#${id}`);
    let host = await store.get("host");
    if (
      !connection ||
      !host ||
      host.expiresAt <= now() ||
      connection.expiresAt <= now() ||
      host.sid !== connection.sid
    ) {
      console.warn(
        "Mold relay frame refusal",
        !connection
          ? "missing-connection"
          : !host
            ? "missing-host"
            : host.expiresAt <= now()
              ? "host-expired"
              : connection.expiresAt <= now()
                ? "connection-expired"
                : "epoch-mismatch",
        connection?.role ?? "unknown",
      );
      return 403;
    }
    if (frame.v !== 2) {
      console.warn("Mold relay frame refusal", "version");
      return 400;
    }
    if (frame.a === "heartbeat") {
      const updated = await updateHost((old) => {
        if (old?.sid !== connection.sid || old.expiresAt <= now()) return null;
        if (connection.role === "host" && old.connectionId === id)
          return { ...old, expiresAt: now() + LIVE };
        if (connection.role === "frontend" && (old.guests?.[id] ?? 0) > now())
          return { ...old, guests: { ...old.guests, [id]: now() + LIVE } };
        return null;
      });
      if (!updated) return 403;
      for (let attempt = 0; attempt < 8; attempt++) {
        const current = await store.get(`connection#${id}`);
        if (!current || current.sid !== connection.sid) return 403;
        if (
          await store.cas(`connection#${id}`, current.revision, {
            ...current,
            expiresAt: now() + LIVE,
          })
        )
          break;
        if (attempt === 7) return 503;
      }
      await post(id, { a: "heartbeat", v: 2, sid: connection.sid });
      return 200;
    }
    if (frame.a === "hello") {
      if (connection.hello) return 409;
      if (
        !(await store.cas(`connection#${id}`, connection.revision, {
          ...connection,
          hello: true,
        }))
      )
        return 409;
      await post(id, {
        a: "ready",
        v: 2,
        sid: host.sid,
        rid: id,
        role: connection.role,
      });
      if (connection.role === "frontend")
        await post(host.connectionId, {
          a: "open",
          v: 2,
          sid: host.sid,
          rid: id,
        });
      return 200;
    }
    if (!validFrame(frame) || frame.sid !== connection.sid) {
      console.warn(
        "Mold relay frame refusal",
        "schema-or-epoch",
        connection.role,
      );
      return 400;
    }
    let target;
    if (connection.role === "host") {
      if (host.connectionId !== id) {
        console.warn("Mold relay frame refusal", "host-owner");
        return 403;
      }
      // A completed frontend can disconnect while its final ACK/EOF is in flight.
      // Drop that stale request frame without evicting the shared host session.
      if ((host.guests?.[frame.rid] ?? 0) <= now()) return 200;
      const guest = await store.get(`connection#${frame.rid}`);
      if (
        guest?.role !== "frontend" ||
        guest.sid !== host.sid ||
        guest.expiresAt <= now()
      )
        return 200;
      target = frame.rid;
    } else {
      if (frame.rid !== id || (host.guests?.[id] ?? 0) <= now()) return 403;
      if (frame.a === "accept") return 403;
      target = host.connectionId;
    }
    const { from: ignored, to: ignoredTo, ...safe } = frame;
    try {
      await post(target, { ...safe, from: id });
    } catch (error) {
      if (error?.$metadata?.httpStatusCode === 410) {
        await leave(target);
        return connection.role === "host" ? 200 : 503;
      }
      throw error;
    }
    return 200;
  }
  return async (event) => {
    const id = event.requestContext?.connectionId;
    const route = event.requestContext?.routeKey;
    if (typeof id !== "string") return { statusCode: 400 };
    try {
      if (route === "$connect")
        return {
          statusCode: await connect(
            id,
            Object.fromEntries(
              Object.entries(event.headers ?? {}).map(([k, v]) => [
                k.toLowerCase(),
                v,
              ]),
            ),
          ),
        };
      if (route === "$disconnect") {
        await leave(id);
        return { statusCode: 200 };
      }
      const status = await message(id, event.body);
      if (status === 400 || status === 403) {
        const membership = await store.get(`connection#${id}`);
        console.warn(
          "Mold relay connection close",
          status,
          membership?.role ?? "unknown",
          id,
        );
        await close(id);
      }
      return { statusCode: status };
    } catch (error) {
      console.error("Mold relay router failure", failureCategory(error));
      return { statusCode: 503 };
    }
  };
}

import { createHash, randomBytes, randomUUID } from "node:crypto";
import { createObjectBroker, objectEnvelope } from "./opaque-objects.mjs";
import {
  hex,
  integer,
  namespaceFor,
  verifyBinding,
  verifyChallenge,
  verifyGrant,
  verifyInvite,
} from "./secure-admission.mjs";

const LIVE = 90,
  CHALLENGE = 60,
  IDLE = 600,
  PEERS = 32,
  HOSTS = 32,
  MAX_GRANT = 30 * 86400;
const connectionKey = (id) => `v3-connection#${id}`;
const hostKey = (namespace) => `v3-host#${namespace}`;
const policyKey = (namespace) => `v3-policy#${namespace}`;
const category = (error) =>
  /^[A-Za-z][A-Za-z0-9]{0,63}$/.test(error?.name ?? "") ? error.name : "Error";
const bindingOnly = (b) => ({
  signingKey: b.signingKey,
  noiseKey: b.noiseKey,
  signature: b.signature,
});

// Allowlist the outer envelope. Application methods, paths, headers, status and
// stream IDs must remain inside Noise-authenticated ciphertext.
export function opaqueFrame(frame) {
  if (
    frame?.v !== 3 ||
    typeof frame.sid !== "string" ||
    frame.sid.length > 128 ||
    typeof frame.rid !== "string" ||
    frame.rid.length > 256
  )
    return;
  const result = { a: frame.a, v: 3, sid: frame.sid, rid: frame.rid };
  if (frame.a === "data") {
    if (frame.o !== undefined) return objectEnvelope(frame);
    if (
      !integer(frame.seq) ||
      typeof frame.d !== "string" ||
      frame.d.length > 21848
    )
      return;
    const bytes = Buffer.from(frame.d, "base64");
    if (bytes.length > 16384 || bytes.toString("base64") !== frame.d) return;
    return { ...result, seq: frame.seq, d: frame.d };
  }
  if (frame.a === "eof")
    return integer(frame.seq) ? { ...result, seq: frame.seq } : undefined;
  if (frame.a === "ack")
    return integer(frame.next) &&
      integer(frame.credit) &&
      frame.credit <= 64 &&
      integer(frame.ack_seq)
      ? {
          ...result,
          next: frame.next,
          credit: frame.credit,
          ack_seq: frame.ack_seq,
        }
      : undefined;
  if (frame.a === "accept" || frame.a === "cancel") return result;
}

export function createSecureRouter({
  store,
  post,
  close,
  checkConnection,
  storage,
  now = () => Math.floor(Date.now() / 1000),
}) {
  const objects = createObjectBroker({ store, storage, now });
  const objectContext = (connection, rid, direction, sender) => ({
    namespace: connection.namespace,
    sid: connection.sid,
    rid,
    direction,
    sender,
  });
  async function cancelObjects(connection, rid) {
    // Storage cleanup must never prevent immediate socket/grant revocation.
    const outcomes = await Promise.allSettled(
      ["host", "device"].map((direction) =>
        objects.cancel(objectContext(connection, rid, direction)),
      ),
    );
    if (outcomes.some((result) => result.status === "rejected"))
      console.error("Mold secure relay object cleanup", "unavailable");
  }
  async function update(key, change) {
    for (let attempt = 0; attempt < 32; attempt++) {
      const old = await store.get(key),
        next = change(old);
      if (!next) return;
      if (await store.cas(key, old?.revision ?? 0, next)) return next;
    }
  }
  async function reserve(key, id, limit, until) {
    return update(key, (old) => {
      const members = Object.fromEntries(
        Object.entries(old?.members ?? {}).filter(
          ([, expiry]) => expiry > now(),
        ),
      );
      if (!members[id] && Object.keys(members).length >= limit) return;
      members[id] = until;
      return {
        members,
        expiresAt: Math.max(...Object.values(members), now() + IDLE),
      };
    });
  }
  // Quota leases namespaces, while the host row alone owns the current socket.
  // A disconnect cannot safely release a namespace across two independent rows:
  // a reconnect may already have replaced its owner. Let the short lease expire.
  async function reserveHost(namespace, until) {
    return update("v3-hosts", (old) => {
      const members = Object.fromEntries(
        Object.entries(old?.members ?? {}).filter(
          ([, member]) => member.expiresAt > now(),
        ),
      );
      if (!members[namespace] && Object.keys(members).length >= HOSTS) return;
      members[namespace] = {
        expiresAt: Math.max(until, members[namespace]?.expiresAt ?? 0),
      };
      return {
        members,
        expiresAt: Math.max(
          ...Object.values(members).map((member) => member.expiresAt),
          now() + LIVE,
        ),
      };
    });
  }
  async function release(key, id) {
    await update(key, (old) => {
      if (!old?.members?.[id]) return;
      const members = { ...old.members };
      delete members[id];
      return { ...old, members };
    });
  }
  async function releasePending(id, connection) {
    await Promise.all([
      release("v3-pending", id),
      connection?.source
        ? release(`v3-pending#${connection.source}`, id)
        : undefined,
    ]);
  }
  async function leave(id, expectedSid) {
    const connection = await store.get(connectionKey(id));
    if (!connection || (expectedSid && connection.sid !== expectedSid)) return;
    await store.remove(connectionKey(id));
    if (connection.state !== "active") {
      await releasePending(id, connection);
      return;
    }
    const namespace = connection.namespace;
    let peers = [];
    await update(hostKey(namespace), (old) => {
      if (old?.sid !== connection.sid) return;
      if (connection.role === "host" && old.connectionId === id) {
        peers = Object.keys(old.guests ?? {});
        return { ...old, expiresAt: 0, guests: {} };
      }
      if (old.guests?.[id]) {
        const guests = { ...old.guests };
        delete guests[id];
        peers = [old.connectionId];
        return { ...old, guests };
      }
    });
    for (const peer of peers) {
      await cancelObjects(connection, connection.role === "host" ? peer : id);
      try {
        await post(peer, {
          a: "cancel",
          v: 3,
          sid: connection.sid,
          rid: connection.role === "host" ? peer : id,
        });
      } catch {}
      if (connection.role === "host") {
        await store.remove(connectionKey(peer));
        try {
          await close(peer);
        } catch {}
      }
    }
  }
  async function connect(id, event) {
    if (event.queryStringParameters?.v !== "3") return 426;
    // API Gateway's trusted socket source, never a caller forwarding header.
    const sourceIp = event.requestContext?.identity?.sourceIp;
    if (
      typeof sourceIp !== "string" ||
      !sourceIp.length ||
      sourceIp.length > 64
    )
      return 403;
    const source = createHash("sha256").update(sourceIp).digest("hex");
    // Hold capacity for the gateway idle window, even if the client never
    // sends hello. DynamoDB TTL is cleanup, not an authorization boundary.
    if (
      !(await reserve(`v3-pending#${source}`, id, 4, now() + IDLE + CHALLENGE))
    )
      return 429;
    if (!(await reserve("v3-pending", id, 128, now() + IDLE + CHALLENGE))) {
      await release(`v3-pending#${source}`, id);
      return 429;
    }
    try {
      await store.put(connectionKey(id), {
        state: "pending",
        source,
        expiresAt: now() + CHALLENGE,
      });
    } catch (error) {
      await releasePending(id, { source });
      throw error;
    }
    return 200;
  }
  async function hello(id, connection, frame) {
    if (
      connection.state !== "pending" ||
      connection.expiresAt <= now() ||
      !["host", "device", "invite"].includes(frame.role) ||
      !hex(frame.namespace, 16) ||
      !hex(frame.hostKey, 32) ||
      namespaceFor(frame.hostKey) !== frame.namespace
    )
      return 403;
    const nonce = randomBytes(32).toString("hex");
    const next = {
      ...connection,
      state: "challenge",
      namespace: frame.namespace,
      hostKey: frame.hostKey,
      role: frame.role,
      objects: frame.objects === 1,
      nonce,
    };
    if (!(await store.cas(connectionKey(id), connection.revision, next)))
      return 409;
    await post(id, {
      a: "challenge",
      v: 3,
      connectionId: id,
      nonce,
      expiresAt: connection.expiresAt,
    });
    return 200;
  }
  function certificate(connection, frame, host) {
    if (connection.role === "device") {
      const g = frame.grant;
      if (
        !g ||
        !hex(g.id, 16) ||
        !integer(g.expiresAt) ||
        g.expiresAt <= now() ||
        g.expiresAt > now() + MAX_GRANT ||
        g.epoch !== host.epoch ||
        !verifyGrant(
          {
            ...g,
            namespace: connection.namespace,
            signingKey: frame.binding.signingKey,
            noiseKey: frame.binding.noiseKey,
          },
          g.signature,
          connection.hostKey,
        )
      )
        return;
      return {
        grant: {
          id: g.id,
          expiresAt: g.expiresAt,
          epoch: g.epoch,
          signature: g.signature,
        },
        signer: frame.binding.signingKey,
      };
    }
    const i = frame.invite;
    if (
      !i ||
      !hex(i.id, 16) ||
      !hex(i.signingKey, 32) ||
      !integer(i.expiresAt) ||
      i.expiresAt <= now() ||
      i.expiresAt > now() + IDLE ||
      i.epoch !== host.epoch ||
      !verifyInvite(
        { ...i, namespace: connection.namespace },
        i.signature,
        connection.hostKey,
      )
    )
      return;
    return {
      invite: {
        id: i.id,
        signingKey: i.signingKey,
        expiresAt: i.expiresAt,
        epoch: i.epoch,
        signature: i.signature,
      },
      signer: i.signingKey,
    };
  }
  async function registration(namespace, signingKey) {
    const value = await store.get(`v3-registration#${namespace}`);
    return validRegistration(value, namespace, signingKey);
  }
  function validRegistration(value, namespace, signingKey) {
    return value?.enabled === true &&
      value.namespace === namespace &&
      value.signingKey === signingKey &&
      typeof value.principalId === "string" &&
      /^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$/.test(value.principalId)
      ? value
      : undefined;
  }
  async function join(id, connection, frame) {
    if (
      connection.state !== "challenge" ||
      connection.expiresAt <= now() ||
      !verifyBinding(frame.binding)
    )
      return 403;
    const registered = await registration(
      connection.namespace,
      connection.hostKey,
    );
    if (!registered) return 403;
    let host, authority;
    if (connection.role === "host") {
      if (frame.binding.signingKey !== connection.hostKey) return 403;
      authority = { signer: connection.hostKey };
    } else {
      host = await store.get(hostKey(connection.namespace));
      if (!host || host.expiresAt <= now()) return 503;
      authority = certificate(connection, frame, host);
      if (!authority) return 403;
    }
    if (
      !verifyChallenge(
        {
          namespace: connection.namespace,
          role: connection.role,
          connectionId: id,
          nonce: connection.nonce,
          expiresAt: connection.expiresAt,
          signingKey: frame.binding.signingKey,
        },
        frame.signature,
        authority.signer,
      )
    )
      return 403;
    // Consume exactly this challenge before installing any namespace membership.
    if (
      !(await store.cas(connectionKey(id), connection.revision, {
        ...connection,
        state: "joining",
      }))
    )
      return 409;
    const binding = bindingOnly(frame.binding);
    if (connection.role === "host") {
      const epoch = frame.epoch ?? 0;
      if (!integer(epoch)) return 400;
      const policy = await update(policyKey(connection.namespace), (old) => {
        if (old && epoch < old.epoch) return;
        return { epoch, expiresAt: now() + MAX_GRANT + LIVE };
      });
      if (!policy) return 403;
      if (!(await reserveHost(connection.namespace, now() + LIVE))) return 503;
      const previous = await store.get(hostKey(connection.namespace));
      // Valid owner proof permits deliberate reconnect. Fence the old epoch
      // before notifying or closing any old peer; no request is resumed.
      host = await update(hostKey(connection.namespace), () => ({
        connectionId: id,
        sid: randomUUID(),
        binding,
        objects: connection.objects,
        principalId: registered.principalId,
        epoch,
        guests: {},
        expiresAt: now() + LIVE,
      }));
      if (!host) return 503;
      if (previous) {
        for (const oldId of [
          previous.connectionId,
          ...Object.keys(previous.guests ?? {}),
        ]) {
          if (oldId === id) continue;
          await store.remove(connectionKey(oldId));
          try {
            await close(oldId);
          } catch {}
        }
      }
    } else {
      if (authority.invite) {
        const key = `v3-invite#${connection.namespace}#${authority.invite.id}`;
        const used = await update(key, (old) =>
          old
            ? undefined
            : { connectionId: id, expiresAt: authority.invite.expiresAt },
        );
        if (!used) return 403;
      }
      const cert = authority.grant ?? authority.invite;
      const expectedSid = host.sid;
      host = await update(hostKey(connection.namespace), (old) => {
        if (
          old?.sid !== expectedSid ||
          old.expiresAt <= now() ||
          old.epoch !== cert.epoch
        )
          return;
        const guests = Object.fromEntries(
          Object.entries(old.guests ?? {}).filter(
            ([, g]) => g.expiresAt > now(),
          ),
        );
        if (
          Object.keys(guests).length >= PEERS ||
          (connection.role === "invite" &&
            Object.values(guests).filter((g) => g.role === "invite").length >=
              2)
        )
          return;
        guests[id] = {
          role: connection.role,
          epoch: cert.epoch,
          expiresAt: Math.min(now() + LIVE, cert.expiresAt),
          grantExpires: cert.expiresAt,
        };
        return { ...old, guests };
      });
      if (!host) return 503;
    }
    const cert = authority.grant ?? authority.invite;
    const active = {
      state: "active",
      role: connection.role,
      namespace: connection.namespace,
      sid: host.sid,
      binding,
      epoch: host.epoch,
      objects:
        connection.role !== "invite" &&
        connection.objects &&
        (connection.role === "host" || host.objects === true),
      expiresAt: Math.min(now() + LIVE, cert?.expiresAt ?? Infinity),
      ...(cert ? { grantExpires: cert.expiresAt } : {}),
      ...(authority.grant ? { grant: authority.grant } : {}),
      ...(authority.invite ? { invite: authority.invite } : {}),
    };
    await store.put(connectionKey(id), active);
    await releasePending(id, connection);
    if (connection.role !== "host")
      await post(host.connectionId, {
        a: "open",
        v: 3,
        sid: host.sid,
        rid: id,
        role: connection.role,
        binding,
        ...(active.grant ? { grant: active.grant } : {}),
        ...(active.invite ? { invite: active.invite } : {}),
        ...(active.objects && objects.policy
          ? { objects: objects.policy }
          : {}),
      });
    await post(id, {
      a: "ready",
      v: 3,
      sid: host.sid,
      rid: id,
      role: connection.role,
      epoch: host.epoch,
      ...(active.objects && objects.policy ? { objects: objects.policy } : {}),
    });
    return 200;
  }
  async function deliver(target, frame, sid) {
    try {
      await post(target, frame);
      return 200;
    } catch (error) {
      if (error?.$metadata?.httpStatusCode !== 410) throw error;
      console.error("Mold secure relay post gone", "observed", target);
      let alive = false;
      try {
        if (checkConnection) {
          await checkConnection(target);
          alive = true;
          console.error(
            "Mold secure relay connection verification",
            "alive",
            target,
          );
        }
      } catch (checkError) {
        if (checkError?.$metadata?.httpStatusCode === 410) {
          console.error(
            "Mold secure relay connection verification",
            "gone",
            target,
          );
          await leave(target, sid);
          return 200;
        }
        console.error(
          "Mold secure relay connection verification",
          "unavailable",
          target,
        );
      }
      if (alive)
        try {
          await post(target, frame);
          return 200;
        } catch {}
      // Sender retries the SAME sequence/ciphertext; never create a new tunnel.
      return 503;
    }
  }
  async function activeMessage(id, connection, frame) {
    const key = hostKey(connection.namespace);
    const isHost = connection.role === "host";
    const isObject = frame.a === "object";
    const safe =
      isObject &&
      typeof frame.sid === "string" &&
      typeof frame.rid === "string" &&
      frame.rid.length <= 256
        ? frame
        : opaqueFrame(frame);
    // Independent consistent reads overlap network latency. Every message still
    // revalidates fresh host, registration and recipient authority; no cache.
    const [host, registeredValue, peer] = await Promise.all([
      store.get(key),
      store.get(`v3-registration#${connection.namespace}`),
      isHost && safe && safe.sid === connection.sid
        ? store.get(connectionKey(safe.rid))
        : undefined,
    ]);
    if (
      !host ||
      host.sid !== connection.sid ||
      host.expiresAt <= now() ||
      connection.expiresAt <= now() ||
      host.epoch !== connection.epoch
    )
      return 403;
    const registered = validRegistration(
      registeredValue,
      connection.namespace,
      host.binding.signingKey,
    );
    if (!registered || registered.principalId !== host.principalId) return 403;
    if (
      isHost
        ? host.connectionId !== id
        : host.guests?.[id]?.expiresAt <= now() || !host.guests?.[id]
    )
      return 403;
    if (frame.a === "policy") {
      if (!isHost || !integer(frame.epoch) || frame.epoch < host.epoch)
        return 403;
      if (frame.epoch === host.epoch) return 200;
      if (
        !(await update(policyKey(connection.namespace), (old) =>
          old && old.epoch > frame.epoch
            ? undefined
            : { epoch: frame.epoch, expiresAt: now() + MAX_GRANT + LIVE },
        ))
      )
        return 409;
      const changed = await update(key, (old) =>
        old?.sid === connection.sid && old.epoch <= frame.epoch
          ? { ...old, epoch: frame.epoch, guests: {} }
          : undefined,
      );
      if (!changed) return 409;
      await update(connectionKey(id), (old) =>
        old?.sid === connection.sid
          ? { ...old, epoch: frame.epoch }
          : undefined,
      );
      for (const peer of Object.keys(host.guests ?? {})) {
        await cancelObjects(connection, peer);
        await store.remove(connectionKey(peer));
        try {
          await close(peer);
        } catch {}
      }
      return 200;
    }
    if (frame.a === "heartbeat") {
      const expiresAt = Math.min(
        now() + LIVE,
        connection.grantExpires ?? Infinity,
      );
      const renewed = await update(key, (old) => {
        if (
          old?.sid !== connection.sid ||
          old.epoch !== connection.epoch ||
          old.expiresAt <= now()
        )
          return;
        if (isHost && old.connectionId === id) return { ...old, expiresAt };
        if (old.guests?.[id]?.expiresAt > now())
          return {
            ...old,
            guests: { ...old.guests, [id]: { ...old.guests[id], expiresAt } },
          };
      });
      if (!renewed) return 403;
      if (isHost && !(await reserveHost(connection.namespace, expiresAt)))
        return 503;
      if (
        !(await update(connectionKey(id), (old) =>
          old?.sid === connection.sid && old.epoch === connection.epoch
            ? { ...old, expiresAt }
            : undefined,
        ))
      )
        return 403;
      await post(id, { a: "heartbeat", v: 3, sid: connection.sid });
      return 200;
    }
    if (!safe || safe.sid !== connection.sid) return 400;
    let target;
    if (isHost) {
      if (
        !peer ||
        peer.state !== "active" ||
        peer.role === "host" ||
        peer.namespace !== connection.namespace ||
        peer.sid !== host.sid ||
        peer.epoch !== host.epoch ||
        peer.expiresAt <= now() ||
        !(host.guests?.[safe.rid]?.expiresAt > now())
      ) {
        await post(id, { a: "cancel", v: 3, sid: host.sid, rid: safe.rid });
        return 200;
      }
      target = safe.rid;
    } else {
      if (safe.rid !== id || safe.a === "accept") return 403;
      target = host.connectionId;
    }
    const context = objectContext(
      connection,
      safe.rid,
      isHost ? "host" : "device",
      id,
    );
    if (safe.a === "accept") {
      const accepted = await update(connectionKey(safe.rid), (old) =>
        old?.state === "active" &&
        old.sid === host.sid &&
        old.namespace === connection.namespace &&
        old.epoch === host.epoch
          ? { ...old, accepted: true }
          : undefined,
      );
      if (!accepted)
        return deliver(
          id,
          { a: "cancel", v: 3, sid: host.sid, rid: safe.rid },
          host.sid,
        );
    }
    if (isObject) {
      if (isHost ? !peer.accepted : !connection.accepted) return 403;
      if (!connection.objects || (isHost && !peer.objects)) return 403;
      if (connection.role === "invite" || (isHost && peer.role === "invite"))
        return 403;
      const reply = await objects.command(context, safe);
      if (!reply) return 400;
      await post(id, reply);
      return 200;
    }
    if (safe.a === "data" && safe.o) {
      if (isHost ? !peer.accepted : !connection.accepted) return 403;
      if (!connection.objects || (isHost && !peer.objects)) return 403;
      if (connection.role === "invite" || (isHost && peer.role === "invite"))
        return 403;
      const result = await objects.forward(context, safe);
      if (result.ack) return deliver(id, result.ack, host.sid);
      if (result.code) {
        if (result.code === "conflict" || result.code === "expired") return 409;
        // Exact reference retry is driven by the unchanged reliable window.
        return 503;
      }
    }
    if (safe.a === "ack" && connection.objects && (!isHost || peer.objects))
      await objects.ack(
        objectContext(connection, safe.rid, isHost ? "device" : "host", target),
        safe,
      );
    if (safe.a === "cancel") {
      const result = await deliver(target, safe, host.sid);
      const peerId = isHost ? safe.rid : id;
      await leave(peerId, host.sid);
      await close(peerId);
      return result;
    }
    return deliver(target, safe, host.sid);
  }
  return async (event) => {
    const id = event.requestContext?.connectionId,
      route = event.requestContext?.routeKey;
    if (typeof id !== "string" || !id.length || id.length > 256)
      return { statusCode: 400 };
    try {
      if (route === "$connect") return { statusCode: await connect(id, event) };
      if (route === "$disconnect") {
        await leave(id);
        return { statusCode: 200 };
      }
      let status = 400;
      if (
        typeof event.body === "string" &&
        Buffer.byteLength(event.body) <= 24 * 1024
      ) {
        let frame;
        try {
          frame = JSON.parse(event.body);
        } catch {}
        const connection = await store.get(connectionKey(id));
        if (!connection || connection.expiresAt <= now()) status = 403;
        else if (frame?.v !== 3) status = 400;
        else if (connection.state === "active")
          status = await activeMessage(id, connection, frame);
        else if (frame.a === "hello")
          status = await hello(id, connection, frame);
        else if (frame.a === "join") status = await join(id, connection, frame);
      }
      if (status === 400 || status === 403 || status === 409) {
        console.error("Mold secure relay refusal", String(status), id);
        await close(id);
        await leave(id);
      }
      return { statusCode: status };
    } catch (error) {
      console.error("Mold secure relay failure", category(error), id);
      return { statusCode: 503 };
    }
  };
}

import init, {
  Identity,
  Session,
  decodeRecord,
  encodeRecord,
  namespace,
  verifySignature,
} from "./generated/secure/mold_secure";
export {
  Identity,
  Session,
  decodeRecord,
  encodeRecord,
  namespace,
  verifySignature,
};
let loading: Promise<unknown> | undefined;
export async function loadSecureCrypto(): Promise<void> {
  loading ??= init().catch((error: unknown) => {
    loading = undefined;
    throw error;
  });
  await loading;
}

export interface SecureBinding {
  signingKey: string;
  noiseKey: string;
  signature: string;
}
export interface SecureGrant {
  id: string;
  expiresAt: number;
  epoch: number;
  signature: string;
}
export interface SecureInvite extends SecureGrant {
  signingKey: string;
}
export interface SecureHostIdentity {
  namespace: string;
  host_key: string;
  host_noise_key: string;
  host_binding_signature: string;
  websocket_url: string | null;
}
export interface SecureHostProfile extends SecureHostIdentity {
  grant: SecureGrant;
  instance_id: string;
  endpoints: import("./connectionRoutes").ConnectionEndpoint[];
  /** Address aliases authenticated by this host, never inferred by name. */
  origins: string[];
  apiKey: string;
}

export const hex = (bytes: Uint8Array): string =>
  Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
export function unhex(value: string, count: number): Uint8Array {
  if (
    typeof value !== "string" ||
    value.length !== count * 2 ||
    !/^[0-9a-f]+$/.test(value)
  )
    throw new Error("Invalid secure identity.");
  return Uint8Array.from(value.match(/../g)!, (byte) =>
    Number.parseInt(byte, 16),
  );
}
export function concat(...parts: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(
    parts.reduce((sum, bytes) => sum + bytes.length, 0),
  );
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}
const ascii = (value: string) => new TextEncoder().encode(value);
export function u64(value: number): Uint8Array {
  if (!Number.isSafeInteger(value) || value < 0)
    throw new Error("Invalid secure counter.");
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(value));
  return bytes;
}
export function binding(identity: Identity): SecureBinding {
  const bytes = identity.binding();
  return {
    signingKey: hex(bytes.subarray(0, 32)),
    noiseKey: hex(bytes.subarray(32, 64)),
    signature: hex(bytes.subarray(64)),
  };
}
export function checkHost(host: SecureHostIdentity): void {
  const signing = unhex(host.host_key, 32);
  const noise = unhex(host.host_noise_key, 32);
  if (
    namespace(signing) !== host.namespace ||
    !verifySignature(
      signing,
      concat(ascii("MOLD3-BIND\0"), signing, noise),
      unhex(host.host_binding_signature, 64),
    )
  )
    throw new Error(
      "The host's secure identity is invalid. Create a new pairing code on the host.",
    );
  if (host.websocket_url !== null) {
    const url = new URL(host.websocket_url);
    if (
      url.protocol !== "wss:" ||
      url.username ||
      url.password ||
      url.search ||
      url.hash ||
      !url.hostname ||
      url.port === "0"
    )
      throw new Error("Invalid secure relay address.");
  }
}
export function grantBytes(
  host: SecureHostIdentity,
  device: SecureBinding,
  grant: SecureGrant,
): Uint8Array {
  return concat(
    ascii("MOLD3-GRANT\0"),
    unhex(host.namespace, 16),
    unhex(device.signingKey, 32),
    unhex(device.noiseKey, 32),
    unhex(grant.id, 16),
    u64(grant.expiresAt),
    u64(grant.epoch),
  );
}
export function checkGrant(
  host: SecureHostIdentity,
  identity: Identity,
  grant: SecureGrant,
): void {
  if (
    grant.expiresAt <= Date.now() / 1000 ||
    !verifySignature(
      unhex(host.host_key, 32),
      grantBytes(host, binding(identity), grant),
      unhex(grant.signature, 64),
    )
  )
    throw new Error(
      "This device's pairing has expired or is invalid. Pair it again from the host.",
    );
}
export function inviteBytes(
  host: SecureHostIdentity,
  invite: SecureInvite,
): Uint8Array {
  return concat(
    ascii("MOLD3-INVITE\0"),
    unhex(host.namespace, 16),
    unhex(invite.id, 16),
    unhex(invite.signingKey, 32),
    u64(invite.expiresAt),
    u64(invite.epoch),
  );
}
export function checkInvite(
  host: SecureHostIdentity,
  invite: SecureInvite,
  seed: string,
): Identity {
  checkHost(host);
  if (
    invite.expiresAt <= Date.now() / 1000 ||
    invite.expiresAt > Date.now() / 1000 + 600 ||
    !verifySignature(
      unhex(host.host_key, 32),
      inviteBytes(host, invite),
      unhex(invite.signature, 64),
    )
  )
    throw new Error(
      "This pairing code has expired or is invalid. Create a new code on the host.",
    );
  const secret = concat(
    unhex(seed, 32),
    crypto.getRandomValues(new Uint8Array(32)),
  );
  const identity = Identity.fromBytes(secret);
  secret.fill(0);
  if (hex(identity.publicSigningKey()) !== invite.signingKey) {
    identity.close();
    identity.free();
    throw new Error("Invalid pairing invitation.");
  }
  return identity;
}
export function joinBytes(
  host: SecureHostIdentity,
  device: SecureBinding,
  role: "device" | "invite",
  challenge: { connectionId: string; nonce: string; expiresAt: number },
): Uint8Array {
  const id = ascii(challenge.connectionId);
  if (
    !id.length ||
    id.length > 128 ||
    !/^[\x21-\x7e]+$/.test(challenge.connectionId) ||
    challenge.expiresAt <= Date.now() / 1000 ||
    challenge.expiresAt > Date.now() / 1000 + 65
  )
    throw new Error("Invalid relay challenge.");
  const length = new Uint8Array(2);
  new DataView(length.buffer).setUint16(0, id.length);
  return concat(
    ascii("MOLD3-JOIN\0"),
    new Uint8Array([3]),
    unhex(host.namespace, 16),
    new Uint8Array([role === "device" ? 2 : 3]),
    length,
    id,
    unhex(challenge.nonce, 32),
    unhex(device.signingKey, 32),
    u64(challenge.expiresAt),
  );
}
export function claimBytes(
  host: SecureHostIdentity,
  invite: SecureInvite,
  handshake: Uint8Array,
  identity: Identity,
): Uint8Array {
  if (handshake.length !== 32)
    throw new Error("Secure handshake is incomplete.");
  return concat(
    ascii("MOLD3-CLAIM\0"),
    unhex(host.namespace, 16),
    unhex(invite.id, 16),
    handshake,
    identity.publicSigningKey(),
    identity.publicNoiseKey(),
  );
}

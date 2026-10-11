// Canonical public-key admission. This module never accepts a Mold API credential.
import { createHash, createPublicKey, verify } from "node:crypto";
export const hex = (value, bytes) =>
  typeof value === "string" &&
  new RegExp(`^[0-9a-f]{${bytes * 2}}$`).test(value);
export const integer = (value) => Number.isSafeInteger(value) && value >= 0;
const raw = (value, bytes) => {
  if (!hex(value, bytes)) throw new Error("Invalid public field");
  return Buffer.from(value, "hex");
};
const u64 = (value) => {
  if (!integer(value)) throw new Error("Invalid integer");
  const bytes = Buffer.alloc(8);
  bytes.writeBigUInt64BE(BigInt(value));
  return bytes;
};
const text = (value) => {
  if (
    typeof value !== "string" ||
    !value.length ||
    Buffer.byteLength(value) > 256
  )
    throw new Error("Invalid routing identifier");
  const bytes = Buffer.from(value);
  const prefix = Buffer.alloc(2);
  prefix.writeUInt16BE(bytes.length);
  return Buffer.concat([prefix, bytes]);
};
export function namespaceFor(signingKey) {
  return createHash("sha256")
    .update(raw(signingKey, 32))
    .digest("hex")
    .slice(0, 32);
}
export const bindingBytes = (binding) =>
  Buffer.concat([
    Buffer.from("MOLD3-BIND\0"),
    raw(binding.signingKey, 32),
    raw(binding.noiseKey, 32),
  ]);
export function challengeBytes(challenge) {
  const role = { host: 1, device: 2, invite: 3 }[challenge.role];
  if (!role) throw new Error("Invalid role");
  return Buffer.concat([
    Buffer.from("MOLD3-JOIN\0"),
    Buffer.from([3]),
    raw(challenge.namespace, 16),
    Buffer.from([role]),
    text(challenge.connectionId),
    raw(challenge.nonce, 32),
    raw(challenge.signingKey, 32),
    u64(challenge.expiresAt),
  ]);
}
export const grantBytes = (grant) =>
  Buffer.concat([
    Buffer.from("MOLD3-GRANT\0"),
    raw(grant.namespace, 16),
    raw(grant.signingKey, 32),
    raw(grant.noiseKey, 32),
    raw(grant.id, 16),
    u64(grant.expiresAt),
    u64(grant.epoch),
  ]);
export const inviteBytes = (invite) =>
  Buffer.concat([
    Buffer.from("MOLD3-INVITE\0"),
    raw(invite.namespace, 16),
    raw(invite.id, 16),
    raw(invite.signingKey, 32),
    u64(invite.expiresAt),
    u64(invite.epoch),
  ]);
function signatureValid(bytes, signature, publicKey) {
  const key = createPublicKey({
    key: Buffer.concat([
      Buffer.from("302a300506032b6570032100", "hex"),
      raw(publicKey, 32),
    ]),
    format: "der",
    type: "spki",
  });
  return verify(null, bytes, key, raw(signature, 64));
}
const safely = (fn) => {
  try {
    return fn();
  } catch {
    return false;
  }
};
export const verifyBinding = (binding) =>
  safely(() =>
    signatureValid(
      bindingBytes(binding),
      binding.signature,
      binding.signingKey,
    ),
  );
export const verifyChallenge = (challenge, signature, publicKey) =>
  safely(() => signatureValid(challengeBytes(challenge), signature, publicKey));
export const verifyGrant = (grant, signature, publicKey) =>
  safely(() => signatureValid(grantBytes(grant), signature, publicKey));
export const verifyInvite = (invite, signature, publicKey) =>
  safely(() => signatureValid(inviteBytes(invite), signature, publicKey));

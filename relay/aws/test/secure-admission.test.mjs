import test from "node:test";
import assert from "node:assert/strict";
import { generateKeyPairSync, sign } from "node:crypto";
import {
  bindingBytes,
  challengeBytes,
  grantBytes,
  inviteBytes,
  namespaceFor,
  verifyBinding,
  verifyChallenge,
  verifyGrant,
  verifyInvite,
} from "../secure-admission.mjs";
export function identity() {
  const { publicKey, privateKey } = generateKeyPairSync("ed25519");
  const signingKey = publicKey
    .export({ format: "der", type: "spki" })
    .subarray(-32)
    .toString("hex");
  const noiseKey = generateKeyPairSync("x25519")
    .publicKey.export({ format: "der", type: "spki" })
    .subarray(-32)
    .toString("hex");
  const binding = { signingKey, noiseKey };
  binding.signature = sign(null, bindingBytes(binding), privateKey).toString(
    "hex",
  );
  return { binding, privateKey, namespace: namespaceFor(signingKey) };
}
test("identity binding and self-certifying namespace reject a substituted Noise key", () => {
  const a = identity(),
    b = identity();
  assert.ok(verifyBinding(a.binding));
  assert.equal(a.namespace.length, 32);
  assert.equal(
    verifyBinding({ ...a.binding, noiseKey: b.binding.noiseKey }),
    false,
  );
  assert.equal(
    verifyBinding({
      ...a.binding,
      signingKey: a.binding.signingKey.toUpperCase(),
    }),
    false,
  );
});
test("challenge proof binds every routing dimension and is not a reusable bearer", () => {
  const a = identity();
  const c = {
    namespace: a.namespace,
    role: "device",
    connectionId: "opaque-ID=",
    nonce: "22".repeat(32),
    signingKey: a.binding.signingKey,
    expiresAt: 12345,
  };
  const signature = sign(null, challengeBytes(c), a.privateKey).toString("hex");
  assert.ok(verifyChallenge(c, signature, a.binding.signingKey));
  for (const change of [
    { namespace: "11".repeat(16) },
    { role: "host" },
    { connectionId: "other" },
    { nonce: "33".repeat(32) },
    { expiresAt: 12346 },
  ])
    assert.equal(
      verifyChallenge({ ...c, ...change }, signature, a.binding.signingKey),
      false,
    );
  assert.throws(() => challengeBytes({ ...c, expiresAt: 1.5 }));
});
test("host grants and invitations bind distinct purposes, public keys and epoch", () => {
  const h = identity(),
    d = identity();
  const g = {
    namespace: h.namespace,
    signingKey: d.binding.signingKey,
    noiseKey: d.binding.noiseKey,
    id: "ab".repeat(16),
    expiresAt: 500,
    epoch: 1,
  };
  const signature = sign(null, grantBytes(g), h.privateKey).toString("hex");
  assert.ok(verifyGrant(g, signature, h.binding.signingKey));
  assert.equal(
    verifyGrant({ ...g, epoch: 2 }, signature, h.binding.signingKey),
    false,
  );
  assert.equal(verifyInvite(g, signature, h.binding.signingKey), false);
  const i = { ...g };
  const proof = sign(null, inviteBytes(i), h.privateKey).toString("hex");
  assert.ok(verifyInvite(i, proof, h.binding.signingKey));
});

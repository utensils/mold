import test from "node:test";
import assert from "node:assert/strict";
import {
  validateUpload,
  credentialDigest,
  validateObjectURL,
  verifyBodyHash,
  claimUpload,
} from "../transfers.mjs";
const good = {
  method: "POST",
  path: "/api/generate?q=1",
  headers: { "content-type": "application/json" },
  size: 3000000,
  sha256: "a".repeat(64),
};
test("uploads are bounded original requests, not arbitrary destinations", () => {
  assert.deepEqual(validateUpload(good), good);
  for (const patch of [
    { path: "https://bad.invalid/" },
    { path: "/_mold/relay/request" },
    { size: 67108865 },
    { sha256: "bad" },
    { headers: { authorization: "secret" } },
    { method: "CONNECT" },
  ])
    assert.throws(() => validateUpload({ ...good, ...patch }));
});
test("credentials bind transfer identity and signed object URLs are exact-origin only", () => {
  assert.notEqual(
    credentialDigest({ "x-api-key": "a" }),
    credentialDigest({ "x-api-key": "b" }),
  );
  assert.throws(() => credentialDigest({}));
  assert.equal(
    validateObjectURL(
      "https://mold-link.urandom.io/_mold/objects/a?Signature=x",
      "https://mold-link.urandom.io",
    ).pathname,
    "/_mold/objects/a",
  );
  for (const url of [
    "https://evil.invalid/_mold/objects/a",
    "https://mold-link.urandom.io/api/x",
    "http://mold-link.urandom.io/_mold/objects/a",
  ])
    assert.throws(() => validateObjectURL(url, "https://mold-link.urandom.io"));
});
test("OAC request hash rejects altered bodies, including empty mutation bodies", () => {
  const headers = {
    "x-amz-content-sha256":
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  };
  assert.equal(verifyBodyHash(headers, Buffer.alloc(0)), true);
  assert.equal(verifyBodyHash(headers, Buffer.from("changed")), false);
});

test("grant consumption is atomic, credential-bound and epoch-bound", async () => {
  const id = "00000000-0000-4000-8000-000000000000",
    headers = { "x-api-key": "fixture" };
  let entry = {
    ...good,
    revision: 1,
    state: "prepared",
    expiresAt: 100,
    credential: credentialDigest(headers),
    sid: "epoch",
  };
  const storage = {
    get: async () => ({ ...entry }),
    cas: async (key, revision, next) => {
      if (entry.revision !== revision) return false;
      entry = { ...next, revision: revision + 1 };
      return true;
    },
  };
  await assert.rejects(
    claimUpload(id, { "x-api-key": "different" }, "epoch", storage, 1),
  );
  await assert.rejects(claimUpload(id, headers, "different", storage, 1));
  const results = await Promise.allSettled([
    claimUpload(id, headers, "epoch", storage, 1),
    claimUpload(id, headers, "epoch", storage, 1),
  ]);
  assert.equal(results.filter((r) => r.status === "fulfilled").length, 1);
  await assert.rejects(claimUpload(id, headers, "epoch", storage, 1));
});

test("upload checksum header is signed rather than hoisted to query", async () => {
  const { uploadURL } = await import("../transfers.mjs");
  const { S3Client } = await import("@aws-sdk/client-s3");
  const client = new S3Client({
    region: "us-west-2",
    credentials: { accessKeyId: "fixture", secretAccessKey: "fixture" },
    useDualstackEndpoint: true,
  });
  const url = new URL(
    await uploadURL(
      "uploads/fixture",
      "a".repeat(44),
      client,
      "mold-relay-042506291754-us-west-2",
    ),
  );
  assert.ok(
    url.searchParams
      .get("X-Amz-SignedHeaders")
      .split(";")
      .includes("x-amz-checksum-sha256"),
  );
  assert.equal(url.searchParams.has("x-amz-checksum-sha256"), false);
});

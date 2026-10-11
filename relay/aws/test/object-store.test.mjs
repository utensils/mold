import test from "node:test";
import assert from "node:assert/strict";
import { createObjectStorage } from "../object-store.mjs";
import { objectPath } from "../opaque-objects.mjs";
test("ACK cleanup keeps an empty conditional-PUT tombstone rather than reopening the key", async () => {
  const commands = [];
  const storage = createObjectStorage(
    { AWS_REGION: "us-west-2", CIPHERTEXT_BUCKET: "mold-test-bucket" },
    undefined,
    {
      async send(command) {
        commands.push(command);
        return {};
      },
    },
  );
  await storage.clear("ciphertext/v3/test");
  assert.equal(commands.length, 1);
  assert.equal(commands[0].constructor.name, "PutObjectCommand");
  assert.equal(commands[0].input.Key, "ciphertext/v3/test");
  assert.equal(commands[0].input.ContentLength, 0);
  assert.equal(commands[0].input.Body.length, 0);
  assert.equal(commands[0].input.Metadata["mold-tombstone"], "1");
});
test("SDK presigns encode raw session segments exactly once and bind PUT length/checksum/condition", async () => {
  const storage = createObjectStorage(
    { AWS_REGION: "us-west-2", CIPHERTEXT_BUCKET: "mold-test-bucket" },
    {
      accessKeyId: "TESTACCESSKEY",
      secretAccessKey: "test-only-not-a-real-secret",
    },
  );
  const context = {
      namespace: "a".repeat(32),
      sid: "session+1",
      rid: "device=2",
      direction: "host",
    },
    id = "b".repeat(32);
  const grants = await storage.grant(objectPath(context, id), {
    bytes: 100,
    sha256: "c".repeat(64),
    expiresAt: Math.floor(Date.now() / 1000) + 300,
  });
  const expected = `/ciphertext/v3/${context.namespace}/session%2B1/device%3D2/host/${id}`;
  for (const url of [grants.putUrl, grants.getUrl])
    assert.equal(new URL(url).pathname, expected);
  const signed = new URL(grants.putUrl).searchParams.get("X-Amz-SignedHeaders");
  for (const header of [
    "content-length",
    "content-type",
    "if-none-match",
    "x-amz-checksum-sha256",
  ])
    assert.ok(signed.split(";").includes(header));
  assert.equal(grants.putHeaders["if-none-match"], "*");
});

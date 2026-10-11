import test from "node:test";
import assert from "node:assert/strict";
import { deflateSync } from "node:zlib";
import { encodeDocument, decodeDocument } from "../store-codec.mjs";

const key = "v3-objects#namespace#session#peer#host";
test("quota documents retain exact grants in bounded binary storage", () => {
  const value = {
    next: 64,
    members: Object.fromEntries(
      Array.from({ length: 4 }, (_, i) => [
        String(i),
        {
          seq: i * 16,
          reply: {
            putUrl: `https://s3.test/put/${i}?token=${"secret-token".repeat(90)}`,
            getUrl: `https://s3.test/get/${i}?token=${"secret-token".repeat(90)}`,
          },
        },
      ]),
    ),
  };
  const encoded = encodeDocument(key, value);
  assert.equal(encoded.codec.S, "deflate-v1");
  assert.ok(encoded.document.B.length < JSON.stringify(value).length / 4);
  assert.deepEqual(decodeDocument(key, encoded), value);
});
test("existing v3 JSON rows remain readable and authority rows are not compressed", () => {
  const value = { namespace: "a".repeat(32), enabled: true };
  assert.deepEqual(
    decodeDocument(key, { document: { S: JSON.stringify(value) } }),
    value,
  );
  assert.deepEqual(encodeDocument("v3-registration#host", value), {
    document: { S: JSON.stringify(value) },
  });
  assert.throws(() =>
    decodeDocument("v3-registration#host", encodeDocument(key, value)),
  );
});
test("unknown codecs, invalid JSON and compressed inflation beyond 64KiB fail closed", () => {
  const document = { B: deflateSync(Buffer.from("{}")) };
  assert.throws(() =>
    decodeDocument(key, { document, codec: { S: "deflate-v2" } }),
  );
  assert.throws(() => decodeDocument(key, { document }));
  assert.throws(() =>
    decodeDocument(key, {
      document: { B: Buffer.from("invalid") },
      codec: { S: "deflate-v1" },
    }),
  );
  for (const input of ["[1]", "null", "{", `{"data":"${"x".repeat(65536)}"}`]) {
    assert.throws(() =>
      decodeDocument(key, {
        document: { B: deflateSync(Buffer.from(input)) },
        codec: { S: "deflate-v1" },
      }),
    );
  }
  assert.throws(() => encodeDocument(key, { data: "x".repeat(65536) }));
});

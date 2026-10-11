import { deflateSync, inflateSync } from "node:zlib";

const LIMIT = 65536;
const quota = (id) => id.startsWith("v3-objects#");
const invalid = () => new Error("InvalidQuotaDocument");

// Repeated signed-URL credentials otherwise make every quota CAS rewrite tens
// of KiB. Compression preserves the exact reply, without reissuing capabilities.
// Revisions and DynamoDB TTL remain independent top-level item attributes.
export function encodeDocument(id, value) {
  const json = JSON.stringify(value);
  if (!quota(id)) return { document: { S: json } };
  const raw = Buffer.from(json);
  if (raw.length > LIMIT) throw invalid();
  const binary = deflateSync(raw);
  if (binary.length > LIMIT) throw invalid();
  return { document: { B: binary }, codec: { S: "deflate-v1" } };
}

export function decodeDocument(id, item) {
  let json;
  if (item.codec !== undefined || item.document?.B !== undefined) {
    if (
      !quota(id) ||
      item.codec?.S !== "deflate-v1" ||
      item.document?.S !== undefined ||
      !item.document?.B ||
      item.document.B.length > LIMIT
    )
      throw invalid();
    // Bound during inflation, not after allocating a potentially huge result.
    json = inflateSync(item.document.B, { maxOutputLength: LIMIT }).toString(
      "utf8",
    );
  } else {
    json = item.document?.S;
    if (
      typeof json !== "string" ||
      (quota(id) && Buffer.byteLength(json) > LIMIT)
    )
      throw invalid();
  }
  const value = JSON.parse(json);
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw invalid();
  return value;
}

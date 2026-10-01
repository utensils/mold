import { createHash, randomUUID } from "node:crypto";
import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  HeadObjectCommand,
  DeleteObjectCommand,
  CreateMultipartUploadCommand,
  UploadPartCommand,
  CompleteMultipartUploadCommand,
  AbortMultipartUploadCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { mkdtemp, open, rm } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { store } from "./aws-store.mjs";
export const MAX_BODY = 67108864,
  THRESHOLD = 2097152,
  OBJECT_THRESHOLD = 16 * 1024 * 1024;
export const s3 = new S3Client({ useDualstackEndpoint: true });
const bucket = () => process.env.BUCKET_NAME;
export function credentialDigest(headers) {
  const authorization = headers.authorization ?? "",
    key = headers["x-api-key"] ?? "";
  if (!authorization && !key) throw new Error("Mold authentication required");
  return createHash("sha256").update(`${authorization}\0${key}`).digest("hex");
}
export function verifyBodyHash(headers, body) {
  return (
    headers["x-amz-content-sha256"] ===
    createHash("sha256").update(body).digest("hex")
  );
}
export function validateUpload(value) {
  if (
    !value ||
    !["POST", "PUT", "PATCH", "DELETE"].includes(value.method) ||
    typeof value.path !== "string" ||
    !value.path.startsWith("/") ||
    value.path.startsWith("//") ||
    value.path.startsWith("/_mold/") ||
    value.path.split("?")[0].startsWith("/metrics") ||
    /[\r\n]/.test(value.path) ||
    value.path.length > 8192 ||
    !Number.isSafeInteger(value.size) ||
    value.size < 0 ||
    value.size > MAX_BODY ||
    !/^([a-f0-9]{64})$/.test(value.sha256)
  )
    throw new Error("Invalid staged request");
  if (
    !value.headers ||
    typeof value.headers !== "object" ||
    Array.isArray(value.headers) ||
    Buffer.byteLength(JSON.stringify(value.headers)) > 16384
  )
    throw new Error("Invalid staged headers");
  for (const [name, v] of Object.entries(value.headers))
    if (
      !/^[a-zA-Z0-9!#$%&'*+.^_`|~-]+$/.test(name) ||
      typeof v !== "string" ||
      /[\r\n]/.test(v) ||
      [
        "authorization",
        "x-api-key",
        "cookie",
        "host",
        "content-length",
        "transfer-encoding",
      ].includes(name.toLowerCase()) ||
      name.toLowerCase().startsWith("x-mold-viewer-")
    )
      throw new Error("Unsafe staged headers");
  return value;
}
export function validateObjectURL(value, origin) {
  const url = new URL(value);
  if (
    url.protocol !== "https:" ||
    url.origin !== new URL(origin).origin ||
    !url.pathname.startsWith("/_mold/objects/") ||
    url.username ||
    url.password ||
    url.hash
  )
    throw new Error("Invalid media object URL");
  return url;
}
export async function uploadURL(
  key,
  checksum,
  client = s3,
  bucketName = bucket(),
) {
  return getSignedUrl(
    client,
    new PutObjectCommand({
      Bucket: bucketName,
      Key: key,
      ContentType: "application/octet-stream",
      ChecksumSHA256: checksum,
    }),
    { expiresIn: 900, unhoistableHeaders: new Set(["x-amz-checksum-sha256"]) },
  );
}
export async function prepareUpload(value, headers, sid) {
  const spec = validateUpload(value),
    credential = credentialDigest(headers),
    id = randomUUID(),
    key = `uploads/${id}`,
    expiresAt = Math.floor(Date.now() / 1000) + 900;
  const entry = { ...spec, credential, sid, key, state: "prepared", expiresAt };
  // Bound outstanding grants independently of Lambda invocation concurrency.
  for (let attempt = 0; attempt < 32; attempt++) {
    const old = await store.get("upload-admission"),
      leases = Object.fromEntries(
        Object.entries(old?.leases ?? {}).filter(
          ([, until]) => until > Math.floor(Date.now() / 1000),
        ),
      );
    if (Object.keys(leases).length >= 64)
      throw new Error("Upload admission full");
    leases[id] = expiresAt;
    if (
      await store.cas("upload-admission", old?.revision ?? 0, {
        leases,
        expiresAt,
      })
    )
      break;
    if (attempt === 31) throw new Error("Upload admission busy");
  }
  await store.put(`upload#${id}`, entry);
  const checksum = Buffer.from(spec.sha256, "hex").toString("base64");
  const url = await uploadURL(key, checksum);
  return {
    id,
    url,
    headers: {
      "content-type": "application/octet-stream",
      "x-amz-checksum-sha256": checksum,
    },
    expires_at: expiresAt,
  };
}
export async function claimUpload(
  id,
  headers,
  sid,
  storage = store,
  now = Math.floor(Date.now() / 1000),
) {
  if (!/^[a-f0-9-]{36}$/.test(id ?? ""))
    throw new Error("Invalid upload identity");
  const entry = await storage.get(`upload#${id}`);
  if (
    !entry ||
    entry.state !== "prepared" ||
    entry.expiresAt <= now ||
    entry.credential !== credentialDigest(headers) ||
    entry.sid !== sid
  )
    throw new Error(
      "Upload is expired, consumed or belongs to another request",
    );
  if (
    !(await storage.cas(`upload#${id}`, entry.revision, {
      ...entry,
      state: "consumed",
    }))
  )
    throw new Error("Upload already consumed");
  return entry;
}
export async function releaseUpload(id, storage = store) {
  for (let attempt = 0; attempt < 32; attempt++) {
    const old = await storage.get("upload-admission");
    if (!old || !old.leases?.[id]) return;
    const leases = { ...old.leases };
    delete leases[id];
    if (await storage.cas("upload-admission", old.revision, { ...old, leases }))
      return;
  }
  throw new Error("Upload admission release busy");
}
export async function consumeUpload(id, headers, sid) {
  const entry = await claimUpload(id, headers, sid);
  await releaseUpload(id);
  const directory = await mkdtemp(join(tmpdir(), "mold-upload-")),
    path = join(directory, "body");
  let file;
  try {
    const head = await s3.send(
      new HeadObjectCommand({ Bucket: bucket(), Key: entry.key }),
    );
    if (head.ContentLength !== entry.size)
      throw new Error("Staged body length mismatch");
    const object = await s3.send(
      new GetObjectCommand({ Bucket: bucket(), Key: entry.key }),
    );
    file = await open(path, "wx", 0o600);
    const hash = createHash("sha256");
    let size = 0;
    for await (const chunk of object.Body) {
      size += chunk.length;
      if (size > entry.size || size > MAX_BODY)
        throw new Error("Staged body exceeds limit");
      hash.update(chunk);
      await file.writeFile(chunk);
    }
    await file.close();
    file = undefined;
    if (size !== entry.size || hash.digest("hex") !== entry.sha256)
      throw new Error("Staged body checksum mismatch");
    return {
      ...entry,
      body: createReadStream(path),
      cleanup: async () => {
        await rm(directory, { recursive: true, force: true });
        await s3.send(
          new DeleteObjectCommand({ Bucket: bucket(), Key: entry.key }),
        );
      },
    };
  } catch (error) {
    await file?.close();
    await rm(directory, { recursive: true, force: true });
    await s3
      .send(new DeleteObjectCommand({ Bucket: bucket(), Key: entry.key }))
      .catch(() => {});
    throw error;
  }
}
export async function objectURL(key, expiresIn = 900) {
  if (
    typeof key !== "string" ||
    !/^_mold\/objects\/[a-f0-9-]+$/.test(key) ||
    expiresIn <= 0 ||
    expiresIn > 900
  )
    throw new Error("Invalid staged object identity");
  return getSignedUrl(
    s3,
    new GetObjectCommand({ Bucket: bucket(), Key: key }),
    { expiresIn },
  );
}
export async function stageObject(
  stream,
  headers,
  key = `_mold/objects/${randomUUID()}`,
) {
  const created = await s3.send(
    new CreateMultipartUploadCommand({
      Bucket: bucket(),
      Key: key,
      ContentType: headers["content-type"] ?? "application/octet-stream",
      CacheControl: "private,no-store",
      ...(headers["content-encoding"]
        ? { ContentEncoding: headers["content-encoding"] }
        : {}),
      ...(headers["content-disposition"]
        ? { ContentDisposition: headers["content-disposition"] }
        : {}),
    }),
  );
  const uploadId = created.UploadId;
  const parts = [];
  let chunks = [],
    length = 0,
    total = 0;
  const flush = async () => {
    const body = Buffer.concat(chunks, length);
    chunks = [];
    length = 0;
    const part = await s3.send(
      new UploadPartCommand({
        Bucket: bucket(),
        Key: key,
        UploadId: uploadId,
        PartNumber: parts.length + 1,
        Body: body,
      }),
    );
    parts.push({ ETag: part.ETag, PartNumber: parts.length + 1 });
  };
  try {
    for await (const chunk of stream) {
      total += chunk.length;
      if (total > 8 * 1024 * 1024 * 1024)
        throw new Error("Staged response exceeds 8 GiB");
      chunks.push(chunk);
      length += chunk.length;
      if (length >= 8 * 1024 * 1024) await flush();
    }
    if (length || parts.length === 0) await flush();
    await s3.send(
      new CompleteMultipartUploadCommand({
        Bucket: bucket(),
        Key: key,
        UploadId: uploadId,
        MultipartUpload: { Parts: parts },
      }),
    );
  } catch (error) {
    await s3
      .send(
        new AbortMultipartUploadCommand({
          Bucket: bucket(),
          Key: key,
          UploadId: uploadId,
        }),
      )
      .catch(() => {});
    throw error;
  }
  const url = await getSignedUrl(
    s3,
    new GetObjectCommand({ Bucket: bucket(), Key: key }),
    { expiresIn: 900 },
  );
  return {
    url,
    key,
    expires_at: Math.floor(Date.now() / 1000) + 900,
    size: total,
  };
}

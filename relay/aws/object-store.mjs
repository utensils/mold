import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  HeadObjectCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

export function createObjectStorage(env = process.env, credentials, transport) {
  if (!env.CIPHERTEXT_BUCKET) return;
  const bucket = env.CIPHERTEXT_BUCKET,
    region = env.AWS_REGION;
  if (
    !/^[a-z0-9][a-z0-9.-]{2,62}$/.test(bucket) ||
    !/^[a-z]{2}-[a-z]+-\d$/.test(region ?? "")
  )
    throw new Error("ObjectConfiguration");
  const origin = `https://${bucket}.s3.${region}.amazonaws.com`;
  const s3 =
    transport ??
    new S3Client({
      region,
      maxAttempts: 2,
      ...(credentials ? { credentials } : {}),
    });
  const input = (key) => {
    if (!key.startsWith("ciphertext/v3/")) throw new Error("ObjectPrefix");
    return { Bucket: bucket, Key: key };
  };
  return {
    origin,
    async grant(key, value) {
      const checksum = Buffer.from(value.sha256, "hex").toString("base64");
      const expiresIn = Math.max(
        1,
        Math.min(300, value.expiresAt - Math.floor(Date.now() / 1000)),
      );
      const [putUrl, getUrl] = await Promise.all([
        getSignedUrl(
          s3,
          new PutObjectCommand({
            ...input(key),
            ContentLength: value.bytes,
            ContentType: "application/octet-stream",
            ChecksumSHA256: checksum,
            IfNoneMatch: "*",
          }),
          {
            expiresIn,
            unhoistableHeaders: new Set(["x-amz-checksum-sha256"]),
            signableHeaders: new Set([
              "content-type",
              "content-length",
              "if-none-match",
            ]),
          },
        ),
        getSignedUrl(s3, new GetObjectCommand(input(key)), { expiresIn }),
      ]);
      return {
        putUrl,
        getUrl,
        putHeaders: {
          "content-type": "application/octet-stream",
          "x-amz-checksum-sha256": checksum,
          "if-none-match": "*",
        },
      };
    },
    async head(key) {
      try {
        const value = await s3.send(
          new HeadObjectCommand({ ...input(key), ChecksumMode: "ENABLED" }),
          { abortSignal: AbortSignal.timeout(2000) },
        );
        return {
          bytes: value.ContentLength,
          sha256: value.ChecksumSHA256
            ? Buffer.from(value.ChecksumSHA256, "base64").toString("hex")
            : undefined,
        };
      } catch (error) {
        if (error.$metadata?.httpStatusCode === 404) return;
        throw error;
      }
    },
    async clear(key) {
      // Keep the key occupied until lifecycle expiry: deleting it would let a
      // still-valid conditional upload grant recreate bytes after quota release.
      await s3.send(
        new PutObjectCommand({
          ...input(key),
          Body: Buffer.alloc(0),
          ContentLength: 0,
          ContentType: "application/octet-stream",
          Metadata: { "mold-tombstone": "1" },
        }),
        { abortSignal: AbortSignal.timeout(2000) },
      );
    },
  };
}

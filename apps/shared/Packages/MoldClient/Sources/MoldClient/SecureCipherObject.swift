import Foundation
import CryptoKit

/// A bulk carrier for already encrypted, consecutive reliable frames. It does
/// not mint cryptographic material or inspect an application request.
enum SecureCipherObject {
    static let maxBytes = 1_048_576
    static let maxCount = 64
    static func canAppend(_ chunks: [Data], _ next: Data) -> Bool {
        chunks.count < maxCount && (1...16_384).contains(next.count) && chunks.reduce(next.count + 4, { $0 + 4 + $1.count }) <= maxBytes
    }
    static func checksum(_ bytes: Data) -> String { SecureHex.encode(Data(SHA256.hash(data: bytes))) }
    static func pack(_ chunks: [Data]) throws -> Data {
        guard (1...maxCount).contains(chunks.count) else { throw SecureConnectionError.protocolViolation }
        var bytes = Data()
        for chunk in chunks {
            guard (1...16_384).contains(chunk.count), bytes.count + 4 + chunk.count <= maxBytes else { throw SecureConnectionError.protocolViolation }
            bytes.append(SecureHTTPRecord.u32(UInt32(chunk.count))); bytes.append(chunk)
        }
        return bytes
    }
    static func unpack(_ input: Data, count: Int, checksum: String) throws -> [Data] {
        guard (1...maxCount).contains(count), (5...maxBytes).contains(input.count), self.checksum(input) == checksum
        else { throw SecureConnectionError.protocolViolation }
        let bytes = Data(input)
        var offset = 0, chunks: [Data] = []
        for _ in 0..<count {
            guard offset + 4 <= bytes.count else { throw SecureConnectionError.protocolViolation }
            let length = Int(bytes[offset..<(offset + 4)].reduce(UInt32(0)) { $0 << 8 | UInt32($1) })
            offset += 4
            guard (1...16_384).contains(length), offset + length <= bytes.count else { throw SecureConnectionError.protocolViolation }
            chunks.append(bytes.subdata(in: offset..<(offset + length))); offset += length
        }
        guard offset == bytes.count else { throw SecureConnectionError.protocolViolation }
        return chunks
    }
}

struct SecureObjectPolicy: Sendable {
    static let productionOrigin = URL(string: "https://mold-relay-042506291754-us-west-2.s3.us-west-2.amazonaws.com")!
    let origin: URL

    static func accept(_ value: Any?, trusted: URL) -> Self? {
        guard let value = value as? [String: Any], value["origin"] as? String == trusted.absoluteString,
              SecureWebSocketConnection.number(value["version"]) == 1,
              SecureWebSocketConnection.number(value["maxBytes"]) == UInt64(SecureCipherObject.maxBytes),
              SecureWebSocketConnection.number(value["maxCount"]) == UInt64(SecureCipherObject.maxCount),
              SecureWebSocketConnection.number(value["maxInFlightBytes"]) == 1_048_576,
              SecureWebSocketConnection.number(value["maxInFlightObjects"]) == 4,
              trusted.scheme == "https", trusted.host != nil, trusted.user == nil, trusted.password == nil,
              trusted.query == nil, trusted.fragment == nil, trusted.path.isEmpty
        else { return nil }
        return Self(origin: trusted)
    }

    func path(namespace: String, sid: String, rid: String, direction: String, id: String) -> String {
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        return "/ciphertext/v3/" + [namespace, sid, rid, direction, id].map {
            $0.addingPercentEncoding(withAllowedCharacters: allowed)!
        }.joined(separator: "/")
    }

    func validate(_ value: String, path: String) throws -> URL {
        guard value.utf8.count <= 8192, let parts = URLComponents(string: value), let url = parts.url,
              parts.scheme == "https", parts.host == origin.host, parts.port == origin.port,
              parts.user == nil, parts.password == nil, parts.fragment == nil, parts.percentEncodedPath == path
        else { throw SecureConnectionError.protocolViolation }
        return url
    }
}

struct SecureObjectReference: Equatable, Sendable {
    let id: String, bytes: Int, checksum: String, url: URL, count: Int
    init(_ value: [String: Any], policy: SecureObjectPolicy, namespace: String, sid: String, rid: String, direction: String) throws {
        guard let id = value["id"] as? String, (try? SecureHex.decode(id, count: 16)) != nil,
              let length = SecureWebSocketConnection.number(value["bytes"]), (5...UInt64(SecureCipherObject.maxBytes)).contains(length),
              let checksum = value["sha256"] as? String, (try? SecureHex.decode(checksum, count: 32)) != nil,
              let count = SecureWebSocketConnection.number(value["count"]), (1...UInt64(SecureCipherObject.maxCount)).contains(count),
              length >= count * 5, length <= count * 16_388,
              let url = value["url"] as? String else { throw SecureConnectionError.protocolViolation }
        self.id = id; self.bytes = Int(length); self.checksum = checksum; self.count = Int(count)
        self.url = try policy.validate(url, path: policy.path(namespace: namespace, sid: sid, rid: rid, direction: direction, id: id))
    }
    var json: [String: Any] { ["id": id, "bytes": bytes, "sha256": checksum, "url": url.absoluteString, "count": count] }
}

struct SecureObjectBatch: Sendable {
    let sequence: UInt64, chunks: [Data], bytes: Data, checksum: String
    init(sequence: UInt64, chunks: [Data]) throws {
        self.sequence = sequence; self.chunks = chunks
        bytes = try SecureCipherObject.pack(chunks); checksum = SecureCipherObject.checksum(bytes)
    }
}

struct SecureObjectGrant: Sendable {
    let reference: SecureObjectReference, putURL: URL, headers: [String: String], expiresAt: UInt64

    init(_ value: [String: Any], batch: SecureObjectBatch, policy: SecureObjectPolicy,
         namespace: String, sid: String, rid: String, now: UInt64 = UInt64(Date().timeIntervalSince1970)) throws {
        guard value["sid"] as? String == sid, value["rid"] as? String == rid,
              SecureWebSocketConnection.number(value["seq"]) == batch.sequence,
              SecureWebSocketConnection.number(value["count"]) == UInt64(batch.chunks.count),
              SecureWebSocketConnection.number(value["bytes"]) == UInt64(batch.bytes.count),
              value["sha256"] as? String == batch.checksum,
              let expiry = SecureWebSocketConnection.number(value["expiresAt"]), expiry > now, expiry <= now + 330,
              let get = value["getUrl"] as? String, let put = value["putUrl"] as? String,
              let headers = value["putHeaders"] as? [String: String], headers.count == 3,
              headers["content-type"] == "application/octet-stream", headers["if-none-match"] == "*",
              headers["x-amz-checksum-sha256"] == (try SecureHex.decode(batch.checksum, count: 32)).base64EncodedString()
        else { throw SecureConnectionError.protocolViolation }
        var object = value; object["url"] = get
        reference = try SecureObjectReference(object, policy: policy, namespace: namespace, sid: sid, rid: rid, direction: "device")
        putURL = try policy.validate(put, path: policy.path(namespace: namespace, sid: sid, rid: rid, direction: "device", id: reference.id))
        self.headers = headers; expiresAt = expiry
    }
}

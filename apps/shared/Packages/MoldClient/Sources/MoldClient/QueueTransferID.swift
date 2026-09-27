import CryptoKit
import Foundation

/// The destination batch id for one transfer, derived and never minted.
///
/// `sha256(["mold.queue-transfer.v1", sourceInstance, jobId, destInstance])`
/// with the v4 variant/version bits set, formatted as a UUID -- the studio's
/// `queueTransferId` (`studio/api/queueTransfer.ts:27-48`), ported
/// byte-compatibly. Deriving rather than minting is the whole idempotency
/// fence: a re-run after a dropped connection asks the destination about the
/// same id and finds the prior attempt instead of admitting a second job
/// (design M6 fact 16, decision 16).
public enum QueueTransferID {
    public static func derive(source: String, jobId: String, destination: String) -> String {
        // `MoldJSON.encoder` on a bare `[String]` produces exactly what
        // `JSON.stringify` writes for a string array -- no spaces, no key
        // strategy to diverge on, since there are no keys.
        let json = (try? MoldJSON.encoder.encode([
            "mold.queue-transfer.v1", source, jobId, destination,
        ])) ?? Data()
        var digest = Array(SHA256.hash(data: json))
        digest[6] = (digest[6] & 0x0F) | 0x50
        digest[8] = (digest[8] & 0x3F) | 0x80
        let hex = digest[0 ..< 16].map { String(format: "%02x", $0) }.joined()
        func part(_ range: Range<Int>) -> Substring {
            let start = hex.index(hex.startIndex, offsetBy: range.lowerBound)
            let end = hex.index(hex.startIndex, offsetBy: range.upperBound)
            return hex[start ..< end]
        }
        return "\(part(0 ..< 8))-\(part(8 ..< 12))-\(part(12 ..< 16))-\(part(16 ..< 20))-\(part(20 ..< 32))"
    }
}

import Foundation
import Testing

@testable import MoldClient

/// `QueueTransferID.derive` ports the studio's `queueTransferId`
/// (`studio/api/queueTransfer.ts:27-48`) byte-compatibly, because deriving
/// rather than minting is the whole idempotency fence: a re-run after a
/// dropped connection has to land on the SAME destination batch id the other
/// app would have derived, or the two apps would each find their own prior
/// attempt and never each other's (design M6 fact 16, decision 16).
///
/// The vector below was computed straight from the TypeScript with Node's
/// built-in `crypto` (SHA-256 has one definition; `@noble/hashes` is just a
/// pure-JS implementation of it):
///
/// ```
/// const digest = createHash("sha256").update(Buffer.from(JSON.stringify([
///   "mold.queue-transfer.v1", "plato-instance-0001", "job-abc-123",
///   "hal9000-instance-0002",
/// ]), "utf8")).digest();
/// digest[6] = (digest[6] & 15) | 80;
/// digest[8] = (digest[8] & 63) | 128;
/// // => "db83f081-c42a-5cad-a003-7aa89b86b5c4"
/// ```
@Test func aTransferIdMatchesTheOneTheWebAppDerives() {
    let id = QueueTransferID.derive(
        source: "plato-instance-0001", jobId: "job-abc-123", destination: "hal9000-instance-0002")
    #expect(id == "db83f081-c42a-5cad-a003-7aa89b86b5c4")
}

/// Two different jobs between the same pair of machines must derive two
/// different ids -- the id names the TRANSFER, not just the machine pair.
@Test func aDifferentJobDerivesADifferentId() {
    let a = QueueTransferID.derive(source: "s", jobId: "job-1", destination: "d")
    let b = QueueTransferID.derive(source: "s", jobId: "job-2", destination: "d")
    #expect(a != b)
}

/// Source and destination are not interchangeable: sending the same job the
/// other direction is a different transfer and must not collide.
@Test func swappingSourceAndDestinationChangesTheId() {
    let outbound = QueueTransferID.derive(source: "a", jobId: "job-1", destination: "b")
    let inbound = QueueTransferID.derive(source: "b", jobId: "job-1", destination: "a")
    #expect(outbound != inbound)
}

/// The v4 variant/version bits the studio sets by hand, so a byte-shaped
/// UUID parser (this app never parses it back, but a log line or a picker
/// might) never chokes on it.
@Test func theDerivedIdIsShapedLikeAVersionFourUUID() throws {
    let id = QueueTransferID.derive(source: "s", jobId: "j", destination: "d")
    let parts = id.split(separator: "-")
    #expect(parts.map(\.count) == [8, 4, 4, 4, 12])
    #expect(parts[2].first == "5")
    let variantNibble = try #require(UInt8(String(parts[3].first!), radix: 16))
    #expect(variantNibble & 0b1100 == 0b1000)
}

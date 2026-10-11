import Foundation
import Testing
@testable import MoldClient

struct SecureRelayWindowTests {
    @Test func reordersBeforeDecryptionAndAcknowledgesIdenticalDuplicateOnly() throws {
        var window = SecureRelayWindow()
        #expect(try !window.insert(sequence: 1, data: Data([2])))
        #expect(window.drain(capacity: 10).isEmpty)
        #expect(window.hasGap)
        #expect(try !window.insert(sequence: 0, data: Data([1])))
        #expect(window.drain(capacity: 10) == Data([1, 2]))
        #expect(window.next == 2)
        #expect(try window.insert(sequence: 0, data: Data([1])))
        #expect(throws: SecureConnectionError.protocolViolation) { try window.insert(sequence: 0, data: Data([3])) }
        #expect(throws: SecureConnectionError.protocolViolation) { try window.insert(sequence: 66, data: Data([3])) }
    }
    @Test func consumerCapacityDoesNotDiscardAcceptedBytes() throws {
        var window = SecureRelayWindow()
        for seq in UInt64(0)..<64 { _ = try window.insert(sequence: seq, data: Data(repeating: 1, count: 16_384)) }
        #expect(window.credit == 0)
        #expect(window.drain(capacity: 16_383).isEmpty)
        #expect(window.next == 0)
        #expect(window.drain(capacity: 16_384).count == 16_384)
        #expect(window.next == 1 && window.credit == 1)
    }
    @Test func advertisedCreditAccountsForFramesAlreadyInFlight() async throws {
        let host = SecureHost(binding: .init(signingKey: "", noiseKey: "", signature: ""), namespace: String(repeating: "a", count: 32),
            websocketURL: URL(string: "wss://fixture.invalid")!, grant: .init(id: "", expiresAt: 0, epoch: 0, signature: ""))
        let connection = try SecureWebSocketConnection(host: host, identity: SecureIdentity())
        try await connection.checkRelativeCredit()
        await connection.close()
    }

    @Test func delayedDuplicateOutsideDigestHistoryIsAcknowledgedWithoutDecryption() throws {
        var window = SecureRelayWindow()
        for seq in UInt64(0)..<256 {
            #expect(try !window.insert(sequence: seq, data: Data([UInt8(truncatingIfNeeded: seq)])))
            #expect(window.drain(capacity: 1) == Data([UInt8(truncatingIfNeeded: seq)]))
        }
        #expect(try window.insert(sequence: 0, data: Data([0])))
        #expect(window.drain(capacity: 100).isEmpty)
        #expect(window.next == 256 && window.credit == 64)
        #expect(throws: SecureConnectionError.protocolViolation) { try window.insert(sequence: 255, data: Data([0])) }
    }
}

private extension SecureWebSocketConnection {
    func checkRelativeCredit() async throws {
        outgoing = 64
        try await receiveFrame(["a":"ack","v":3,"sid":sid,"rid":rid,"ack_seq":0,"next":15,"credit":49])
        #expect(remoteCredit == 0, "All 49 advertised slots already contain unacknowledged frames")
        try await receiveFrame(["a":"ack","v":3,"sid":sid,"rid":rid,"ack_seq":1,"next":64,"credit":64])
        #expect(remoteCredit == 64)
    }
}

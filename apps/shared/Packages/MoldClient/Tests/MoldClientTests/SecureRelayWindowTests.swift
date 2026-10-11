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
        #expect(throws: SecureConnectionError.protocolViolation) { try window.insert(sequence: 18, data: Data([3])) }
    }
    @Test func consumerCapacityDoesNotDiscardAcceptedBytes() throws {
        var window = SecureRelayWindow()
        for seq in UInt64(0)..<16 { _ = try window.insert(sequence: seq, data: Data(repeating: 1, count: 16_384)) }
        #expect(window.credit == 0)
        #expect(window.drain(capacity: 16_383).isEmpty)
        #expect(window.next == 0)
        #expect(window.drain(capacity: 16_384).count == 16_384)
        #expect(window.next == 1 && window.credit == 1)
    }
    @Test func delayedDuplicateOutsideDigestHistoryIsAcknowledgedWithoutDecryption() throws {
        var window = SecureRelayWindow()
        for seq in UInt64(0)..<64 {
            #expect(try !window.insert(sequence: seq, data: Data([UInt8(seq)])))
            #expect(window.drain(capacity: 1) == Data([UInt8(seq)]))
        }
        #expect(try window.insert(sequence: 0, data: Data([0])))
        #expect(window.drain(capacity: 100).isEmpty)
        #expect(window.next == 64 && window.credit == 16)
        #expect(throws: SecureConnectionError.protocolViolation) { try window.insert(sequence: 63, data: Data([0])) }
    }
}

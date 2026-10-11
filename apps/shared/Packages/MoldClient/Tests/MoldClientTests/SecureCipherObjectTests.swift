import Foundation
import CryptoKit
import Testing
@testable import MoldClient

struct SecureCipherObjectTests {
    @Test func completeCiphertextFramesRoundTripAndStayInsideTheObjectBound() throws {
        let chunks = (0..<16).map { Data(repeating: UInt8($0), count: 16_320) }
        let bytes = try SecureCipherObject.pack(chunks)
        #expect(bytes.count == 261_184)
        #expect(try SecureCipherObject.unpack(bytes, count: 16, checksum: SecureCipherObject.checksum(bytes)) == chunks)
        #expect(throws: SecureConnectionError.protocolViolation) { try SecureCipherObject.pack(Array(repeating: Data(repeating: 1, count: 16_384), count: 16)) }
        let full = Data(repeating: 1, count: 16_384)
        #expect(SecureCipherObject.canAppend(Array(repeating: full, count: 14), full))
        #expect(!SecureCipherObject.canAppend(Array(repeating: full, count: 15), full), "Split before a sixteenth full chunk exceeds the byte cap")
    }
    @Test func malformedObjectNeverExposesAPartialCiphertextFrame() throws {
        let bytes = try SecureCipherObject.pack([Data([1, 2]), Data([3, 4])])
        for bad in [Data(bytes.dropLast()), bytes + Data([0]), Data([0, 0, 0, 0]), Data()] {
            #expect(throws: SecureConnectionError.protocolViolation) { try SecureCipherObject.unpack(bad, count: 2, checksum: SecureCipherObject.checksum(bad)) }
        }
        #expect(throws: SecureConnectionError.protocolViolation) { try SecureCipherObject.unpack(bytes, count: 1, checksum: SecureCipherObject.checksum(bytes)) }
        #expect(throws: SecureConnectionError.protocolViolation) { try SecureCipherObject.unpack(bytes, count: 2, checksum: String(repeating: "0", count: 64)) }
    }
    @Test func storagePolicyPinsOriginAndEverySessionPathSegment() throws {
        let policy = SecureObjectPolicy(origin: SecureObjectPolicy.productionOrigin)
        let path = policy.path(namespace: String(repeating: "a", count: 32), sid: "host/id", rid: "device? id", direction: "host", id: String(repeating: "b", count: 32))
        #expect(path.contains("/host%2Fid/device%3F%20id/host/"))
        let valid = URL(string: policy.origin.absoluteString + path + "?opaque=signature")!
        #expect(try policy.validate(valid.absoluteString, path: path) == valid)
        for value in [valid.absoluteString.replacingOccurrences(of: "https:", with: "http:"),
                      valid.absoluteString.replacingOccurrences(of: policy.origin.host!, with: "evil.example"),
                      valid.absoluteString + "#fragment", valid.absoluteString.replacingOccurrences(of: "/host/", with: "/device/"),
                      valid.absoluteString.replacingOccurrences(of: "https://", with: "https://user@") ] {
            #expect(throws: SecureConnectionError.protocolViolation) { try policy.validate(value, path: path) }
        }
    }
    @Test func cloudAdvertisementCannotIntroduceAnotherTrustedOrigin() {
        let offered: [String: Any] = ["version": 1, "origin": "https://evil.example", "maxBytes": 262_144,
            "maxCount": 16, "maxInFlightBytes": 1_048_576, "maxInFlightObjects": 4]
        #expect(SecureObjectPolicy.accept(offered, trusted: SecureObjectPolicy.productionOrigin) == nil)
        var trusted = offered; trusted["origin"] = SecureObjectPolicy.productionOrigin.absoluteString
        #expect(SecureObjectPolicy.accept(trusted, trusted: SecureObjectPolicy.productionOrigin) != nil)
        trusted["maxCount"] = 17
        #expect(SecureObjectPolicy.accept(trusted, trusted: SecureObjectPolicy.productionOrigin) == nil)
    }

    @Test func allocationMustEchoTheExactBatchAndOnlyAllowSignedUploadHeaders() throws {
        let batch = try SecureObjectBatch(sequence: 42, chunks: [Data([1, 2, 3])])
        let policy = SecureObjectPolicy(origin: SecureObjectPolicy.productionOrigin)
        let namespace = String(repeating: "a", count: 32), id = String(repeating: "b", count: 32)
        let url = policy.origin.absoluteString + policy.path(namespace: namespace, sid: "host", rid: "device", direction: "device", id: id)
        let value: [String: Any] = ["sid": "host", "rid": "device", "seq": 42, "count": 1, "bytes": batch.bytes.count,
            "sha256": batch.checksum, "id": id, "getUrl": url + "?get", "putUrl": url + "?put", "expiresAt": 1200,
            "putHeaders": ["content-type": "application/octet-stream", "if-none-match": "*",
                "x-amz-checksum-sha256": try SecureHex.decode(batch.checksum, count: 32).base64EncodedString()]]
        let grant = try SecureObjectGrant(value, batch: batch, policy: policy, namespace: namespace, sid: "host", rid: "device", now: 1000)
        #expect(grant.reference.count == 1)
        for (field, bad): (String, Any) in [("seq", 41), ("count", 2), ("bytes", 999), ("sha256", String(repeating: "0", count: 64)),
            ("sid", "other"), ("expiresAt", 1000), ("expiresAt", 1331), ("putUrl", "https://evil.example/path"),
            ("putHeaders", ["content-type": "application/octet-stream", "Authorization": "must-not-send"])] {
            var invalid = value; invalid[field] = bad
            #expect(throws: SecureConnectionError.protocolViolation) {
                try SecureObjectGrant(invalid, batch: batch, policy: policy, namespace: namespace, sid: "host", rid: "device", now: 1000)
            }
        }
    }

    @Test func fullyConsumedObjectIsAcknowledgedWithoutFetchingItsPossiblyDeletedBody() async throws {
        let host = SecureHost(binding: .init(signingKey: "", noiseKey: "", signature: ""), namespace: String(repeating: "a", count: 32),
            websocketURL: URL(string: "wss://fixture.invalid")!, grant: .init(id: "", expiresAt: 0, epoch: 0, signature: ""))
        let connection = try SecureWebSocketConnection(host: host, identity: SecureIdentity())
        try await connection.checkRetiredObject()
        await connection.close()
    }

    @Test func duplicateDisabledBrokerReplyDoesNotInterruptInlineFallback() async throws {
        let host = SecureHost(binding: .init(signingKey: "", noiseKey: "", signature: ""), namespace: String(repeating: "a", count: 32),
            websocketURL: URL(string: "wss://fixture.invalid")!, grant: .init(id: "", expiresAt: 0, epoch: 0, signature: ""))
        let connection = try SecureWebSocketConnection(host: host, identity: SecureIdentity())
        await connection.checkDuplicateDisabledReply()
        await connection.close()
    }

    @Test func shortCiphertextTailsStayInAnExistingBulkBatch() async throws {
        let connection = try makeConnection()
        await connection.checkShortTailBatching()
        await connection.close()
    }

    @Test func aRetiredObjectDownloadFailureDoesNotInterruptTheChannel() async throws {
        let connection = try makeConnection()
        await connection.checkRetiredDownloadFailure()
        await connection.close()
    }

    @Test func fourOutstandingObjectsLeaveTheFifthBatchInlineUntilAcknowledged() async throws {
        let connection = try makeConnection()
        await connection.checkObjectReservationBudget()
        await connection.close()
    }

    @Test func finishingAnEarlierUploadDoesNotFlushTheNextPartialBatchEarly() async throws {
        let connection = try makeConnection()
        await connection.checkCompletionKeepsBatchDeadline()
        await connection.close()
    }

    @Test func foundationHTTPSUpgradeMetadataDoesNotDisableWSSObjectTransfer() async throws {
        let connection = try makeConnection()
        await connection.checkObjectNegotiation()
        await connection.close()
    }
    @Test func gatewayThrottlePreservesPendingCiphertextWithoutReopeningHTTP() async throws {
        let connection = try makeConnection()
        try await connection.checkGatewayRetry()
        await connection.close()
    }

    private func makeConnection() throws -> SecureWebSocketConnection {
        let host = SecureHost(binding: .init(signingKey: "", noiseKey: "", signature: ""), namespace: String(repeating: "a", count: 32),
            websocketURL: URL(string: "wss://fixture.invalid")!, grant: .init(id: "", expiresAt: 0, epoch: 0, signature: ""))
        return try SecureWebSocketConnection(host: host, identity: SecureIdentity())
    }
}

private extension SecureWebSocketConnection {
    func checkCompletionKeepsBatchDeadline() {
        objectPolicy = SecureObjectPolicy(origin: SecureObjectPolicy.productionOrigin)
        stagedFrames = [(17, Data(repeating: 1, count: 16_349))]
        objectFlushTask = Task { try? await Task.sleep(for: .seconds(30)) }
        completedObjectSend(sequence: 0)
        #expect(stagedFrames.count == 1, "Finishing an unrelated PUT must preserve the existing batching deadline")
        #expect(objectSends.isEmpty, "A partial batch is not published early")
    }

    func checkObjectNegotiation() {
        #expect(socket.originalRequest?.url?.scheme == "https", "Foundation rewrites the WSS upgrade request")
        configureObjectPolicy(["version":1,"origin":SecureObjectPolicy.productionOrigin.absoluteString,
            "maxBytes":262_144,"maxCount":16,"maxInFlightBytes":1_048_576,"maxInFlightObjects":4])
        #expect(objectPolicy != nil, "The originally selected WSS route supports objects")
    }
    func checkGatewayRetry() async throws {
        let ciphertext = Data([1,2,3])
        pending[1] = PendingFrame(bytes: ciphertext, sent: .now, tries: 1)
        try await receiveFrame(["connectionId":"fixture","requestId":"fixture","message":"Internal server error"])
        #expect(pending[1]?.bytes == ciphertext && pending[1]?.tries == 1)
        #expect(failure == nil)
        #expect(!Self.isGatewayBusy(["connectionId":" ", "requestId":"fixture", "message":"Internal server error"]))
        #expect(!Self.isGatewayBusy(["connectionId":"é", "requestId":"fixture", "message":"Internal server error"]))
        #expect(!Self.isGatewayBusy(["connectionId":"fixture", "requestId":"fixture", "message":"Internal server error", "v":2]))
        await #expect(throws: SecureConnectionError.protocolViolation) { try await receiveFrame(["message":"Internal server error"]) }
    }

    func checkObjectReservationBudget() {
        for index in 0..<4 { #expect(reserveObject(sequence: UInt64(index * 15), count: 15)) }
        #expect(!reserveObject(sequence: 60, count: 4), "64 full wire chunks need five objects after length prefixes; the tail must stay inline")
        retireObjects(next: 14)
        #expect(!reserveObject(sequence: 60, count: 4), "Partial ACK cannot release an object")
        retireObjects(next: 15)
        #expect(reserveObject(sequence: 60, count: 4))
    }

    func checkShortTailBatching() {
        objectPolicy = SecureObjectPolicy(origin: SecureObjectPolicy.productionOrigin)
        #expect(!shouldStagePacket(bulk: false), "Initial handshakes and small controls stay inline")
        stagedFrames = (0..<4).map { (UInt64($0), Data(repeating: 1, count: 16_349)) }
        #expect(shouldStagePacket(bulk: false), "A 64KiB body-read tail must not flush its bulk prefix inline")
        objectAllocationDisabled = true
        #expect(!shouldStagePacket(bulk: true))
        stagedFrames.removeAll()
    }

    func checkRetiredDownloadFailure() async {
        objectPolicy = SecureObjectPolicy(origin: SecureObjectPolicy.productionOrigin)
        _ = try! window.insert(sequence: 0, data: Data([1])); _ = window.drain(capacity: 1)
        #expect(await objectDownloadFailed(SecureConnectionError.protocolViolation, end: 1))
        #expect(failure == nil)
        ackTask?.cancel(); ackTask = nil
        #expect(!(await objectDownloadFailed(SecureConnectionError.protocolViolation, end: 2)))
        #expect(failure != nil, "A partially missing object must still fail closed")
    }

    func checkDuplicateDisabledReply() async {
        do {
            let _: SecureObjectGrant = try await withCheckedThrowingContinuation { continuation in
                let requestID = String(repeating: "d", count: 32)
                let batch = try! SecureObjectBatch(sequence: 0, chunks: [Data([1])])
                objectRequests[requestID] = ObjectRequest(batch: batch, continuation: continuation, timeout: Task {},
                    deadline: .now + .seconds(30), nextAttempt: .now)
                let reply: [String: Any] = ["requestId": requestID, "op": "error", "code": "disabled"]
                #expect(throws: Never.self) { try receiveObjectReply(reply) }
                #expect(objectRequests.isEmpty)
                #expect(throws: Never.self) { try receiveObjectReply(reply) }
            }
            Issue.record("A disabled allocation must return its fallback signal")
        } catch { }
    }

    func checkRetiredObject() throws {
        let policy = SecureObjectPolicy(origin: SecureObjectPolicy.productionOrigin)
        objectPolicy = policy; sid = "host"; rid = "device"
        for index in 0..<40 { _ = try window.insert(sequence: UInt64(index), data: Data([1])); _ = window.drain(capacity: 16_384) }
        let id = String(repeating: "b", count: 32)
        let value: [String: Any] = ["id": id, "bytes": 100, "count": 1, "sha256": String(repeating: "c", count: 64),
            "url": policy.origin.absoluteString + policy.path(namespace: host.namespace, sid: sid, rid: rid, direction: "host", id: id) + "?expired"]
        #expect(try receiveObject(sequence: 0, value: value))
        #expect(objectFetches.isEmpty)
    }
}

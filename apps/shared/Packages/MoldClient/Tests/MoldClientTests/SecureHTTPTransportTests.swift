import Foundation
import Testing
@testable import MoldClient

struct SecureHTTPTransportTests {
    @Test func largeUploadsKeepHTTPAuthorityInsideBoundedEncryptedRecords() async throws {
        let connection = RecordFixture()
        let active = SecureHTTPTransport(connection: connection)
        var request = URLRequest(url: URL(string: "https://paired.example/api/reference%20input?tag=a&tag=b")!)
        request.httpMethod = "PATCH"
        request.setValue("private-device-bearer", forHTTPHeaderField: "X-Api-Key")
        let task = Task { try await active.response(for: request, body: .data(Data(repeating: 4, count: 3 * 1024 * 1024))) }
        await connection.waitForData(count: 64)
        let sent = await connection.sent
        let open = try #require(sent.first { $0.kind == .open })
        let payload = try #require(JSONSerialization.jsonObject(with: open.payload) as? [String: Any])
        #expect(payload["path"] as? String == "/api/reference%20input?tag=a&tag=b")
        #expect(payload["length"] as? Int == 3 * 1024 * 1024)
        #expect((payload["headers"] as? [[String]])?.contains(["X-Api-Key", "private-device-bearer"]) == true)
        #expect(sent.filter { $0.kind == .data }.reduce(0) { $0 + $1.payload.count } == 1_044_480)
        await active.close()
        _ = try? await task.value
    }
    @Test func bodyCreditReturnsOnlyWhenConsumerPullsAndCancelIsPerStream() async throws {
        let connection = RecordFixture()
        let transport = SecureHTTPTransport(connection: connection)
        let task = Task { try await transport.response(for: URLRequest(url: URL(string: "https://paired.example/api/events")!), body: .data(Data())) }
        await connection.waitForOpen()
        await connection.offer(SecureHTTPRecord(.headers, 1, Data(#"{"status":200,"headers":[["content-type","text/event-stream"]]}"#.utf8)))
        let (bytes, response) = try await task.value
        #expect(response.statusCode == 200)
        for _ in 0..<33 { await connection.offer(SecureHTTPRecord(.data, 1, Data(repeating: 7, count: 16_320))) }
        #expect(await connection.creditCount == 0)
        var iterator = bytes.makeAsyncIterator()
        #expect(try await iterator.next() == 7)
        #expect(await connection.creditCount == 0)
        for _ in 1...524_287 { _ = try await iterator.next() }
        #expect(await connection.creditCount == 1)
        let credit = try #require(await connection.sent.first(where: { $0.kind == .credit })?.credit)
        #expect(credit >= 524_288 && credit < 524_288 + 16_384)
        await bytes.cancel()
        #expect(await connection.sent.contains(where: { $0.kind == .cancel && $0.streamID == 1 }))
        #expect(await connection.closed == false)
        await transport.close()
    }

    @Test func tinyPausedResponseFramesHaveBoundedAllocationCount() {
        var queue = SecureBodyQueue()
        for _ in 0..<65_536 { queue.append(Data([7])) }
        #expect(queue.chunks.count == 4)
        #expect(queue.chunks.allSatisfy { $0.count == 16_384 })
        #expect(queue.popFirst() == Data(repeating: 7, count: 16_384))
    }

    @Test func disconnectedMutationIsNeverReopened() async throws {
        let connection = RecordFixture()
        let transport = SecureHTTPTransport(connection: connection)
        var request = URLRequest(url: URL(string: "https://paired.example/api/generate")!)
        request.httpMethod = "POST"
        let task = Task { try await transport.response(for: request, body: .data(Data("{}".utf8))) }
        await connection.waitForOpen()
        await connection.close()
        await #expect(throws: SecureConnectionError.closed) { try await task.value }
        #expect(await connection.sent.filter { $0.kind == .open }.count == 1)
    }

    @Test func progressingResponseMayOutliveTheRequestIdleTimeout() async throws {
        let connection = RecordFixture()
        let transport = SecureHTTPTransport(connection: connection)
        var request = URLRequest(url: URL(string: "https://paired.example/api/video")!)
        request.timeoutInterval = 0.15
        let task = Task { try await transport.response(for: request, body: .data(Data())) }
        await connection.waitForOpen()
        await connection.offer(SecureHTTPRecord(.headers, 1, Data(#"{"status":200,"headers":[]}"#.utf8)))
        let (bytes, _) = try await task.value
        let producer = Task {
            for value: UInt8 in 1...16 {
                try await Task.sleep(for: .milliseconds(80))
                await connection.offer(SecureHTTPRecord(.data, 1, Data([value])))
            }
            await connection.offer(SecureHTTPRecord(.end, 1))
        }
        var result: [UInt8] = []
        for try await byte in bytes { result.append(byte) }
        try await producer.value
        #expect(result == Array(1...16))
        await transport.close()
    }

    @Test func pausedMediaConsumerDoesNotExpireBufferedResponse() async throws {
        let connection = RecordFixture()
        let transport = SecureHTTPTransport(connection: connection)
        var request = URLRequest(url: URL(string: "https://paired.example/api/video")!)
        request.timeoutInterval = 0.05
        let task = Task { try await transport.response(for: request, body: .data(Data())) }
        await connection.waitForOpen()
        await connection.offer(SecureHTTPRecord(.headers, 1, Data(#"{"status":200,"headers":[]}"#.utf8)))
        let (bytes, _) = try await task.value
        await connection.offer(SecureHTTPRecord(.data, 1, Data([42])))
        try await Task.sleep(for: .milliseconds(1200))
        await connection.offer(SecureHTTPRecord(.end, 1))
        var result: [UInt8] = []
        for try await byte in bytes { result.append(byte) }
        #expect(result == [42])
        await transport.close()
    }

    @Test func stalledBodyReadTimesOutOnlyItsOwnStream() async throws {
        let connection = RecordFixture()
        let transport = SecureHTTPTransport(connection: connection)
        var request = URLRequest(url: URL(string: "https://paired.example/api/video")!)
        request.timeoutInterval = 0.05
        let task = Task { try await transport.response(for: request, body: .data(Data())) }
        await connection.waitForOpen()
        await connection.offer(SecureHTTPRecord(.headers, 1, Data(#"{"status":200,"headers":[]}"#.utf8)))
        let (bytes, _) = try await task.value
        var iterator = bytes.makeAsyncIterator()
        await #expect(throws: URLError(.timedOut)) { try await iterator.next() }
        #expect(await connection.closed == false)
        #expect(await connection.sent.contains(where: { $0.kind == .cancel && $0.streamID == 1 }))
        await transport.close()
    }
}

private actor RecordFixture: SecureRecordConnection {
    var sent: [SecureHTTPRecord] = []
    var received: [SecureHTTPRecord] = []
    var waiting: CheckedContinuation<SecureHTTPRecord, any Error>?
    var closed = false
    var creditCount: Int { sent.filter { $0.kind == .credit }.count }
    func send(_ record: SecureHTTPRecord) throws {
        guard !closed else { throw SecureConnectionError.closed }
        sent.append(record)
    }
    func receive() async throws -> SecureHTTPRecord {
        guard !closed else { throw SecureConnectionError.closed }
        if !received.isEmpty { return received.removeFirst() }
        return try await withCheckedThrowingContinuation { waiting = $0 }
    }
    func offer(_ record: SecureHTTPRecord) {
        if let waiting { self.waiting = nil; waiting.resume(returning: record) }
        else { received.append(record) }
    }
    func close() {
        closed = true; waiting?.resume(throwing: SecureConnectionError.closed); waiting = nil
    }
    func waitForOpen() async {
        while !sent.contains(where: { $0.kind == .open }) { await Task.yield() }
    }
    func waitForData(count: Int) async {
        while sent.filter({ $0.kind == .data }).count < count { await Task.yield() }
    }
}

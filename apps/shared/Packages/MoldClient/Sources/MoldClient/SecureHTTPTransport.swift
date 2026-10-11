import Foundation

/// One authenticated channel, at most 32 independently cancellable HTTP streams.
/// The receive window is 1 MiB per stream; credit returns in 512 KiB batches
/// only as the consumer pulls data. A closed connection never replays requests.
actor SecureHTTPTransport: MoldHTTPTransport {
    private struct Stream {
        let url: URL
        let idleTimeout: TimeInterval
        var headers: CheckedContinuation<(MoldResponseBytes, HTTPURLResponse), any Error>?
        var reader: CheckedContinuation<Data?, any Error>?
        var writer: CheckedContinuation<Void, any Error>?
        var chunks = SecureBodyQueue()
        var receiveCredit = SecureHTTPRecord.streamWindow
        var sendCredit = SecureHTTPRecord.streamWindow
        var consumedCredit = 0
        var ended = false
        var upload: Task<Void, Never>?
        var timeout: Task<Void, Never>?
        var timeoutEpoch: UInt64 = 0
        var uploaded = false
    }
    private let connection: any SecureRecordConnection
    private var streams: [UInt32: Stream] = [:]
    private var nextID: UInt32 = 1
    private var incoming: Task<Void, Never>?
    private var failure: (any Error)?
    init(connection: any SecureRecordConnection) { self.connection = connection }
    var isUsable: Bool { failure == nil }
    var isIdle: Bool { streams.isEmpty }

    func response(for request: URLRequest, body: MoldRequestBody) async throws -> (MoldResponseBytes, HTTPURLResponse) {
        try Task.checkCancellation()
        if let failure { throw failure }
        guard streams.count < 32, nextID < UInt32.max - 2 else { throw SecureConnectionError.capacity }
        guard let url = request.url, url.user == nil, url.password == nil else { throw SecureConnectionError.protocolViolation }
        let length = try body.length
        guard length >= 0, length <= 64 * 1024 * 1024 else { throw SecureConnectionError.requestTooLarge }
        let id = nextID; nextID += 2
        if incoming == nil { incoming = Task { await receiveLoop() } }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                streams[id] = Stream(url: url, idleTimeout: max(0.01, request.timeoutInterval), headers: continuation)
                streams[id]?.upload = Task {
                    do { try await writeRequest(id: id, request: request, body: body, length: length) }
                    catch { await cancel(id, error: error) }
                }
                refreshDeadline(id)
            }
        } onCancel: { Task { await self.cancel(id, error: CancellationError()) } }
    }

    private func writeRequest(id: UInt32, request: URLRequest, body: MoldRequestBody, length: Int) async throws {
        guard let url = request.url else { throw SecureConnectionError.protocolViolation }
        let headers = (request.allHTTPHeaderFields ?? [:]).filter { !["host", "content-length"].contains($0.key.lowercased()) }
            .sorted { $0.key < $1.key }.map { [$0.key, $0.value] }
        let target = url.path(percentEncoded: true) + (url.query(percentEncoded: true).map { "?" + $0 } ?? "")
        let payload = try JSONSerialization.data(withJSONObject: ["method": request.httpMethod ?? "GET", "path": target, "headers": headers, "length": length])
        try await connection.send(SecureHTTPRecord(.open, id, payload))
        refreshDeadline(id)
        switch body {
        case let .data(data):
            var offset = 0
            while offset < data.count {
                let count = min(16_320, data.count - offset)
                try await writeData(id, Data(data[offset..<(offset + count)])); offset += count
            }
        case let .file(url):
            let file = try FileHandle(forReadingFrom: url)
            defer { try? file.close() }
            var remaining = length
            while remaining > 0 {
                guard let chunk = try file.read(upToCount: min(16_320, remaining)), !chunk.isEmpty else { throw SecureConnectionError.protocolViolation }
                try await writeData(id, chunk); remaining -= chunk.count
            }
            guard try file.read(upToCount: 1)?.isEmpty != false else { throw SecureConnectionError.protocolViolation }
        }
        try Task.checkCancellation()
        try await connection.send(SecureHTTPRecord(.end, id))
        streams[id]?.uploaded = true
        refreshDeadline(id)
    }

    private func writeData(_ id: UInt32, _ data: Data) async throws {
        while true {
            try Task.checkCancellation()
            guard let stream = streams[id] else { throw SecureConnectionError.closed }
            if stream.sendCredit >= data.count { break }
            try await withCheckedThrowingContinuation { streams[id]?.writer = $0 }
        }
        streams[id]?.sendCredit -= data.count
        try await connection.send(SecureHTTPRecord(.data, id, data))
        refreshDeadline(id)
    }

    private func receiveLoop() async {
        do {
            while !Task.isCancelled { try await receive(connection.receive()) }
        } catch { await fail(error) }
    }

    private func receive(_ record: SecureHTTPRecord) async throws {
        guard let stream = streams[record.streamID] else {
            // Late bytes for a locally cancelled, already-used stream are discarded.
            guard record.streamID < nextID, record.streamID % 2 == 1 else { throw SecureConnectionError.protocolViolation }
            return
        }
        let id = record.streamID
        switch record.kind {
        case .headers:
            guard let pending = stream.headers,
                  let value = try JSONSerialization.jsonObject(with: record.payload) as? [String: Any],
                  let status = value["status"] as? Int, (200...599).contains(status),
                  let headers = value["headers"] as? [[String]], headers.count <= 128,
                  headers.allSatisfy({ $0.count == 2 }) else { throw SecureConnectionError.protocolViolation }
            var fields: [String: String] = [:]
            for pair in headers { fields[pair[0]] = pair[1] }
            guard let response = HTTPURLResponse(url: stream.url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: fields)
            else { throw SecureConnectionError.protocolViolation }
            streams[id]?.headers = nil
            refreshDeadline(id)
            pending.resume(returning: (MoldResponseBytes(reader: SecureStreamBody(owner: self, id: id)), response))
        case .data:
            guard stream.headers == nil, !stream.ended, record.payload.count <= stream.receiveCredit,
                  !record.payload.isEmpty, record.payload.count <= 16_384 else { throw SecureConnectionError.protocolViolation }
            streams[id]?.receiveCredit -= record.payload.count
            streams[id]?.chunks.append(record.payload)
            if let reader = stream.reader {
                streams[id]?.reader = nil
                do { reader.resume(returning: try await read(id)) }
                catch { reader.resume(throwing: error) }
            }
            refreshDeadline(id)
        case .end:
            guard stream.headers == nil, !stream.ended else { throw SecureConnectionError.protocolViolation }
            streams[id]?.ended = true
            refreshDeadline(id)
            if let reader = stream.reader {
                streams[id]?.reader = nil
                finish(id); reader.resume(returning: nil)
            }
        case .credit:
            guard let credit = record.credit, credit > 0, credit <= SecureHTTPRecord.streamWindow,
                  stream.sendCredit + Int(credit) <= SecureHTTPRecord.streamWindow else { throw SecureConnectionError.protocolViolation }
            streams[id]?.sendCredit += Int(credit)
            streams[id]?.writer = nil; stream.writer?.resume()
            refreshDeadline(id)
        case .cancel: await cancel(id, error: SecureConnectionError.closed, notify: false)
        case .open: throw SecureConnectionError.protocolViolation
        }
    }

    func read(_ id: UInt32) async throws -> Data? {
        try Task.checkCancellation()
        guard let stream = streams[id] else { throw failure ?? SecureConnectionError.closed }
        if let first = stream.chunks.first {
            _ = streams[id]?.chunks.popFirst()
            let consumed = stream.consumedCredit + first.count
            if consumed >= SecureHTTPRecord.creditBatch {
                streams[id]?.consumedCredit = 0
                streams[id]?.receiveCredit += consumed
                try await connection.send(SecureHTTPRecord(.credit, id, SecureHTTPRecord.u32(UInt32(consumed))))
            } else { streams[id]?.consumedCredit = consumed }
            return first
        }
        if stream.ended { finish(id); return nil }
        guard stream.reader == nil else { throw SecureConnectionError.protocolViolation }
        return try await withCheckedThrowingContinuation {
            streams[id]?.reader = $0
            refreshDeadline(id)
        }
    }

    /// Timeout measures a stalled operation, not the lifetime of a download or
    /// event subscription. A paused consumer holds bounded credit without a
    /// deadline; the next pending pull starts a fresh idle interval.
    private func refreshDeadline(_ id: UInt32) {
        guard var stream = streams[id] else { return }
        stream.timeout?.cancel()
        stream.timeout = nil
        stream.timeoutEpoch &+= 1
        let epoch = stream.timeoutEpoch
        if stream.headers != nil || stream.reader != nil || !stream.uploaded {
            let interval = stream.idleTimeout
            stream.timeout = Task {
                do { try await Task.sleep(for: .seconds(interval)) }
                catch { return }
                guard streams[id]?.timeoutEpoch == epoch else { return }
                await cancel(id, error: URLError(.timedOut))
            }
        }
        streams[id] = stream
    }

    func cancel(_ id: UInt32, error: any Error = CancellationError(), notify: Bool = true) async {
        guard let stream = streams.removeValue(forKey: id) else { return }
        stream.upload?.cancel(); stream.timeout?.cancel()
        stream.headers?.resume(throwing: error); stream.reader?.resume(throwing: error); stream.writer?.resume(throwing: error)
        if notify { try? await connection.send(SecureHTTPRecord(.cancel, id, Data("cancelled".utf8))) }
    }
    private func finish(_ id: UInt32) {
        let stream = streams.removeValue(forKey: id)
        stream?.upload?.cancel(); stream?.timeout?.cancel(); stream?.writer?.resume(throwing: CancellationError())
    }
    private func fail(_ error: any Error) async {
        guard failure == nil else { return }
        failure = error
        for id in Array(streams.keys) { await cancel(id, error: error, notify: false) }
        await connection.close()
    }
    func close() async { incoming?.cancel(); incoming = nil; await fail(SecureConnectionError.closed) }
}

private struct SecureStreamBody: MoldBodyReader {
    let owner: SecureHTTPTransport
    let id: UInt32
    func read() async throws -> Data? { try await owner.read(id) }
    func cancel() async { await owner.cancel(id) }
}

/// Coalesce paused tiny frames without delaying a live stream's waiting reader.
struct SecureBodyQueue {
    private(set) var chunks: [Data] = []
    var first: Data? { chunks.first }
    mutating func append(_ bytes: Data) {
        var offset = 0
        if let last = chunks.indices.last, chunks[last].count < 16_384 {
            let count = min(16_384 - chunks[last].count, bytes.count)
            chunks[last].append(bytes.prefix(count)); offset += count
        }
        while offset < bytes.count {
            let count = min(16_384, bytes.count - offset)
            chunks.append(Data(bytes[offset..<(offset + count)])); offset += count
        }
    }
    mutating func popFirst() -> Data? {
        chunks.isEmpty ? nil : chunks.removeFirst()
    }
}

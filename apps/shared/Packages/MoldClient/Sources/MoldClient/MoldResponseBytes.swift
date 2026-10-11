import Foundation

/// One consumer pulls bounded decrypted chunks. Implementations return body credit only
/// as chunks are pulled, and cancellation releases both the queue and its HTTP stream.
public protocol MoldBodyReader: Sendable {
    func read() async throws -> Data?
    func cancel() async
}

public enum MoldBodyError: Error, Equatable, Sendable { case invalidChunk }

/// A lazy byte sequence shared by encrypted responses and the embedded loopback engine.
/// It holds one chunk locally rather than dispatching to the connection actor per byte.
public struct MoldResponseBytes: AsyncSequence, Sendable {
    public typealias Element = UInt8
    private let source: Source
    fileprivate enum Source: Sendable {
        case loopback(URLSession.AsyncBytes)
        case encrypted(BodyLease)
    }

    init(_ bytes: URLSession.AsyncBytes) { source = .loopback(bytes) }
    public init(reader: any MoldBodyReader) { source = .encrypted(BodyLease(reader)) }

    public func cancel() async {
        if case let .encrypted(lease) = source { await lease.reader.cancel() }
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        private var loopback: URLSession.AsyncBytes.AsyncIterator?
        private let lease: BodyLease?
        private var chunk = Data()
        private var offset = 0
        private var ended = false

        fileprivate init(_ source: Source) {
            switch source {
            case let .loopback(bytes): loopback = bytes.makeAsyncIterator(); lease = nil
            case let .encrypted(lease): self.lease = lease
            }
        }

        public mutating func next() async throws -> UInt8? {
            guard !ended else { return nil }
            if loopback != nil { return try await loopback?.next() }
            guard let lease else { return nil }
            do {
                try Task.checkCancellation()
                if offset == chunk.count {
                    let next = try await withTaskCancellationHandler {
                        try await lease.reader.read()
                    } onCancel: {
                        Task { await lease.reader.cancel() }
                    }
                    guard let next else { ended = true; chunk = Data(); return nil }
                    guard !next.isEmpty, next.count <= 65_536 else { throw MoldBodyError.invalidChunk }
                    chunk = next; offset = 0
                }
                defer { offset += 1 }
                return chunk[chunk.startIndex + offset]
            } catch {
                ended = true; chunk = Data()
                await lease.reader.cancel()
                throw error
            }
        }
    }

    public func makeAsyncIterator() -> AsyncIterator { AsyncIterator(source) }
}

/// The last owner dropping a partially consumed response also cancels the host stream.
fileprivate final class BodyLease: Sendable {
    let reader: any MoldBodyReader
    init(_ reader: any MoldBodyReader) { self.reader = reader }
    deinit {
        let reader = reader
        Task { await reader.cancel() }
    }
}

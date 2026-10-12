import Foundation
import Testing
@testable import MoldClient

struct MoldResponseBytesTests {
    @Test func chunkSequencePreservesBoundedTransportChunksAndPullsOnDemand() async throws {
        let first = Data(repeating: 1, count: 16_384)
        let second = Data([2, 3])
        let source = BodyFixture([first, second])
        var iterator = MoldResponseBytes(reader: source).chunks.makeAsyncIterator()
        #expect(await source.reads == 0)
        #expect(try await iterator.next() == first)
        #expect(await source.reads == 1)
        #expect(try await iterator.next() == second)
        #expect(await source.reads == 2)
        #expect(try await iterator.next() == nil)
    }

    @Test(arguments: [Data(), Data(repeating: 1, count: 65_537)])
    func chunkSequenceRejectsInvalidTransportChunks(chunk: Data) async throws {
        let source = BodyFixture([chunk])
        await #expect(throws: MoldBodyError.invalidChunk) {
            var iterator = MoldResponseBytes(reader: source).chunks.makeAsyncIterator()
            _ = try await iterator.next()
        }
        #expect(await source.cancelled)
    }

    @Test func cancellingChunkReadClosesTheUnderlyingStream() async throws {
        let source = BodyFixture([], suspend: true)
        let task = Task {
            var iterator = MoldResponseBytes(reader: source).chunks.makeAsyncIterator()
            _ = try await iterator.next()
        }
        while await source.reads == 0 { await Task.yield() }
        task.cancel()
        _ = try? await task.value
        #expect(await source.cancelled)
    }

    @Test func doesNotPullAnotherChunkUntilConsumerExhaustsTheCurrentOne() async throws {
        let source = BodyFixture([Data([1, 2, 3]), Data([4])])
        let bytes = MoldResponseBytes(reader: source)
        #expect(await source.reads == 0)
        var iterator = bytes.makeAsyncIterator()
        #expect(try await iterator.next() == 1)
        #expect(await source.reads == 1)
        #expect(try await iterator.next() == 2)
        #expect(try await iterator.next() == 3)
        #expect(await source.reads == 1)
        #expect(try await iterator.next() == 4)
        #expect(await source.reads == 2)
        #expect(try await iterator.next() == nil)
    }

    @Test func cancellationClosesTheUnderlyingStream() async throws {
        let source = BodyFixture([], suspend: true)
        let bytes = MoldResponseBytes(reader: source)
        let task = Task {
            var iterator = bytes.makeAsyncIterator()
            _ = try await iterator.next()
        }
        while await source.reads == 0 { await Task.yield() }
        task.cancel()
        _ = try? await task.value
        #expect(await source.cancelled)
    }

    @Test func rejectsOversizedTransportChunk() async throws {
        let source = BodyFixture([Data(repeating: 1, count: 65_537)])
        let bytes = MoldResponseBytes(reader: source)
        await #expect(throws: MoldBodyError.invalidChunk) {
            var iterator = bytes.makeAsyncIterator()
            _ = try await iterator.next()
        }
        #expect(await source.cancelled)
    }
}

private actor BodyFixture: MoldBodyReader {
    var chunks: [Data]
    var reads = 0
    var cancelled = false
    let suspend: Bool
    init(_ chunks: [Data], suspend: Bool = false) { self.chunks = chunks; self.suspend = suspend }
    func read() async throws -> Data? {
        reads += 1
        if suspend { try await Task.sleep(for: .seconds(10)) }
        return chunks.isEmpty ? nil : chunks.removeFirst()
    }
    func cancel() { cancelled = true }
}

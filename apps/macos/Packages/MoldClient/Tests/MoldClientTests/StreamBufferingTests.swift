import Foundation
import Testing

@testable import MoldClient

// How much the stream pipeline holds, and when it reads.

/// A byte source that records how far it has been read. `Sendable` because
/// `moldLines()` is constrained to a sendable base; the counter is only ever
/// touched from the one iterating task.
private final class CountingBytes: AsyncSequence, @unchecked Sendable {
    typealias Element = UInt8

    let bytes: [UInt8]
    private(set) var pulled = 0

    init(_ text: String) { bytes = Array(text.utf8) }

    struct Iterator: AsyncIteratorProtocol {
        let source: CountingBytes
        var index = 0
        mutating func next() async throws -> UInt8? {
            guard index < source.bytes.count else { return nil }
            defer { index += 1 }
            source.pulled += 1
            return source.bytes[index]
        }
    }

    func makeAsyncIterator() -> Iterator { Iterator(source: self) }
}

/// **Fails today**: `moldLines()` runs its own `Task` pumping an
/// `.unbounded` `AsyncThrowingStream`, so it drains the source as fast as the
/// source will go however little the consumer takes -- and there is no `yield`
/// that blocks, so nothing anywhere exerts backpressure on the socket. During
/// a gallery burst memory tracked the burst rather than what the app could
/// apply.
@Test func theLineReaderReadsNoFurtherThanItsConsumerAsks() async throws {
    let source = CountingBytes("alpha\nbeta\ngamma\n")
    var lines = source.moldLines().makeAsyncIterator()

    #expect(try await lines.next() == "alpha")
    #expect(source.pulled == 6, "\"alpha\" and its newline, and not one byte more")

    #expect(try await lines.next() == "beta")
    #expect(source.pulled == 11)
}

/// The same for the frame parser above it: one `next()` reads one frame's
/// worth of lines, not the whole connection.
@Test func theFrameParserReadsNoFurtherThanItsConsumerAsks() async throws {
    let wire = "event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\n"
    let source = CountingBytes(wire)
    var frames = source.moldLines().serverSentEvents().makeAsyncIterator()

    let first = try await frames.next()
    #expect(first?.name == "a")
    #expect(first?.data == "1")
    #expect(source.pulled == 21, "through the blank line that ended the frame, and no further")
}

/// **Fails today**: all four `AsyncThrowingStream`s in the package default to
/// `.unbounded`. A ceiling is the only lever `AsyncStream` offers -- there is
/// no `yield` that blocks -- so each one has to state its own, and a new
/// stream that forgets to is the same bug again.
@Test func everyStreamStatesItsBufferingPolicy() throws {
    let sources = RepoFixtures.testDirectory
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appending(path: "Sources/MoldClient")
    let files = try FileManager.default
        .contentsOfDirectory(at: sources, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "swift" }

    // A CONSTRUCTION is the name followed by `(` or a trailing closure. A
    // RETURN TYPE is the same name after `->`, whose `{` is the function
    // body -- so the lead is what tells them apart.
    let construction = /(?<lead>->[ ]*)?AsyncThrowingStream(<[^>]*>)?(?<open>[ ]*[({])/
    var unpoliced: [String] = []
    var buffers: [String] = []
    for file in files {
        let source = try String(contentsOf: file, encoding: .utf8)
        for match in source.matches(of: construction) where match.lead == nil {
            buffers.append(file.lastPathComponent)
            let policed = match.open.hasSuffix("(")
                && source[match.range.upperBound...].hasPrefix("bufferingPolicy:")
            guard !policed else { continue }
            unpoliced.append(
                "\(file.lastPathComponent): \(source[match.range.upperBound...].prefix(40))")
        }
    }
    #expect(unpoliced == [])

    // **Fails today**: `stream(_:timeout:)` builds one too, STACKED under
    // each of these -- so `events()`'s declared 512 was 512 on top of another
    // independent 512, and "one buffer in the pipeline" was not true. The
    // four routes are the only places a buffer belongs, because they are the
    // only places that know what losing a frame COSTS.
    #expect(Set(buffers) == [
        "HTTPBackend+Events.swift",     // MoldEvent
        "HTTPBackend+Work.swift",       // DownloadEvent
        "HTTPBackend+Generation.swift", // BatchStatus
        "HTTPBackend+Machines.swift",   // ResourceSnapshot
    ])
}

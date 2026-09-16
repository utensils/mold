import Foundation
import Testing

@testable import MoldClient

/// Splitting a byte stream into lines, INCLUDING the empty ones.
///
/// This exists because `URLSession.AsyncBytes.lines` does not: it silently
/// drops zero-length lines. In server-sent events the blank line IS the frame
/// terminator, so a parser fed by `.lines` sees `event:` and `data:` arrive
/// and is never told the event finished. The stream connects, the server
/// sends, and the client reports nothing -- with no error anywhere.
@Suite struct LineAccumulatorSuite {
    private func lines(_ text: String) -> [String] {
        var accumulator = LineAccumulator()
        var out: [String] = []
        for byte in Array(text.utf8) { out += accumulator.consume(byte) }
        return out
    }

    @Test func ordinaryLinesComeOutWhole() {
        #expect(lines("alpha\nbeta\n") == ["alpha", "beta"])
    }

    /// The whole point.
    @Test func anEmptyLineIsALine() {
        #expect(lines("data: x\n\ndata: y\n\n") == ["data: x", "", "data: y", ""])
    }

    @Test func carriageReturnsAreFraming() {
        #expect(lines("data: x\r\n\r\n") == ["data: x", ""])
    }

    /// Bytes arrive in whatever chunks the network felt like. A line is only
    /// finished when its newline arrives.
    @Test func aLineWithNoNewlineYetIsNotALine() {
        var accumulator = LineAccumulator()
        #expect(Array("par".utf8).flatMap { accumulator.consume($0) }.isEmpty)
        #expect(Array("tial\n".utf8).flatMap { accumulator.consume($0) } == ["partial"])
    }

    @Test func multiByteCharactersSurviveBeingSplitAcrossReads() {
        #expect(lines("caf\u{00E9}\n") == ["caf\u{00E9}"])
    }

    // MARK: - End to end

    /// The exact bytes a live mold sends when a print is favourited from
    /// somewhere else, captured from `GET /api/events`.
    @Test func aRealFrameParsesIntoARealEvent() {
        let wire = """
            event: authority\r\n\
            data: {"instance_id":"ff00bea2"}\r\n\
            \r\n\
            event: event\r\n\
            data: {"type":"gallery_updated","filename":"a.png"}\r\n\
            \r\n
            """
        var accumulator = LineAccumulator()
        var parser = SSEParser()
        var events: [MoldEvent] = []
        for byte in Array(wire.utf8) {
            for line in accumulator.consume(byte) {
                guard let frame = parser.consume(line: line) else { continue }
                if let event = MoldEvent(name: frame.name, data: frame.data) {
                    events.append(event)
                }
            }
        }
        #expect(events == [.authority(instanceID: "ff00bea2"),
                           .gallery(.updated(filename: "a.png", row: nil))])
    }
}

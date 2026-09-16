import Foundation
import Testing

@testable import MoldClient

private func events(_ lines: [String]) -> [ServerSentEvent] {
    var parser = SSEParser()
    return lines.compactMap { parser.consume(line: $0) }
}

@Test func parsesANamedEvent() {
    let parsed = events(["event: generation_batch", #"data: {"id":"x"}"#, ""])
    #expect(parsed.count == 1)
    #expect(parsed[0].name == "generation_batch")
    #expect(parsed[0].data == #"{"id":"x"}"#)
}

@Test func keepAliveCommentsAreNotEvents() {
    // mold sends `: ping` every 15 seconds. Delivering those as empty events
    // would look like a settled batch arriving four times a minute.
    #expect(events([": ping", "", ": ping", ""]).isEmpty)
}

@Test func multipleDataLinesJoinWithNewlines() {
    let parsed = events(["data: one", "data: two", ""])
    #expect(parsed.first?.data == "one\ntwo")
}

@Test func exactlyOneLeadingSpaceIsStripped() {
    // "data:  x" carries a value of " x" -- one space is framing, the rest is
    // content, and eating both would corrupt any payload starting with space.
    #expect(events(["data:  x", ""]).first?.data == " x")
    #expect(events(["data:y", ""]).first?.data == "y")
}

@Test func aBlankLineWithNothingBufferedEmitsNothing() {
    #expect(events(["", "", ""]).isEmpty)
}

@Test func consecutiveEventsDoNotLeakFieldsIntoEachOther() {
    let parsed = events(["event: a", "data: 1", "", "data: 2", ""])
    #expect(parsed.count == 2)
    #expect(parsed[0].name == "a")
    // The second event has no `event:` line, so it must not inherit "a".
    #expect(parsed[1].name == nil)
    #expect(parsed[1].data == "2")
}

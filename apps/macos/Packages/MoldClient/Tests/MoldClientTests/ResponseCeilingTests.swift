import Foundation
import Testing

@testable import MoldClient

/// How much of an answer this app will hold.
///
/// **Fails today**: every route buffers its whole body through
/// `session.data(for:)` and nothing caps the result, so a host answering
/// without limit takes the process down -- there is no ceiling to ask.
@Suite struct ResponseCeilingSuite {

    @Test func anOrdinaryAnswerComesStraightBack() throws {
        let body = Data(repeating: 1, count: 1_024)
        #expect(try ResponseCeiling.checked(body, ceiling: 4_096, what: "that print") == body)
    }

    @Test func anAnswerOverTheCeilingIsRefusedRatherThanHeld() {
        let body = Data(repeating: 1, count: 4_097)
        #expect(throws: ResponseCeiling.Exceeded(bytes: 4_097, ceiling: 4_096,
                                                 what: "that print")) {
            try ResponseCeiling.checked(body, ceiling: 4_096, what: "that print")
        }
    }

    @Test func anAnswerExactlyAtTheCeilingIsKept() throws {
        let body = Data(repeating: 1, count: 4_096)
        #expect(try ResponseCeiling.checked(body, ceiling: 4_096, what: "x").count == 4_096)
    }

    /// The refusal has to be readable: `HostStore.report` turns it into the
    /// machine's own sentence.
    @Test func theRefusalSaysHowBigAndHowMuchIsAllowed() throws {
        let refusal = ResponseCeiling.Exceeded(bytes: 600 * 1_024 * 1_024,
                                               ceiling: ResponseCeiling.media,
                                               what: "that print")
        let sentence = try #require(refusal.errorDescription)
        #expect(sentence.contains("that print"))
        #expect(sentence.contains("MB"))
    }

    /// The media ceiling is the server's own for one member, not a guess.
    @Test func theMediaCeilingMatchesWhatTheServerWillServe() {
        #expect(ResponseCeiling.media == 512 * 1_024 * 1_024)
        #expect(ResponseCeiling.json < ResponseCeiling.media)
    }
}

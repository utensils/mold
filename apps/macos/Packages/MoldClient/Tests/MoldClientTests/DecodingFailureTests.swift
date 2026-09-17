import Foundation
import Testing

@testable import MoldClient

// What a reply this build cannot read says about itself.

private struct Child: Decodable, Equatable {
    let jobId: String
    let revision: Int
}

private struct Batch: Decodable {
    let id: String
    let children: [Child]
}

private func summary(of json: String) -> String? {
    do {
        _ = try MoldJSON.decoder.decode(Batch.self, from: Data(json.utf8))
        return nil
    } catch let error as DecodingError {
        return DecodingFailure.summary(error)
    } catch {
        return "not a decoding error"
    }
}

/// **Fails today**: `get`/`post` collapse every decode failure to
/// `.malformedResponse`, so the one fact that makes such a bug findable --
/// which key, at which path -- is thrown away before anyone can see it.
/// The name is the SWIFT one, because `MoldJSON`'s key strategy has already
/// converted it by the time the key is missing -- which is the more useful
/// half anyway: it names the property to go and look at.
@Test func aMissingKeyNamesTheKeyAndWhereItWasMissing() {
    #expect(summary(of: #"{"id":"b1","children":[{"revision":1}]}"#)
            == "no key jobId at children.0")
}

/// An array index reads as its number, not as Foundation's "Index 0" prose.
@Test func aPathThroughAnArrayReadsAsItsIndex() {
    let wrong = #"{"id":"b1","children":[{"job_id":"j","revision":1},{"job_id":"j","revision":"x"}]}"#
    #expect(summary(of: wrong) == "type mismatch, expected Int at children.1.revision")
}

/// A null where a value belongs is its own case, and says which type was
/// wanted.
@Test func aNullWhereAValueBelongsSaysSo() {
    #expect(summary(of: #"{"id":null,"children":[]}"#) == "no value for String at id")
}

/// A body that is not JSON at all has no path to name, so it says that
/// rather than inventing one.
@Test func aBodyThatIsNotJSONSaysItIsCorruptAtTheRoot() {
    #expect(summary(of: "<html>nope</html>") == "corrupt data at the root")
}

/// The whole point of building the summary from the error's structured parts
/// rather than its `debugDescription`: a value a person owns must never reach
/// the log. Here the offending value is a prompt.
@Test func noValueFromTheBodyEverAppearsInTheSummary() throws {
    let body = #"{"id":"b1","children":[{"job_id":"a cat asleep on a windowsill","revision":"x"}]}"#
    let text = try #require(summary(of: body))
    #expect(!text.contains("cat"))
    #expect(!text.contains("windowsill"))
    #expect(text == "type mismatch, expected Int at children.0.revision")
}

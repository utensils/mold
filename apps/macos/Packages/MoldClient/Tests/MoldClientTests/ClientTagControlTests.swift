import Foundation
import Testing

@testable import MoldClient

// `normalize_tag_name` (`crates/mold-core/src/organization.rs:37-50`) REFUSES
// a tag holding a non-whitespace control character, and a refused tag is a
// 422 on the whole render. Collapsing the length instead of refusing it is a
// stated divergence -- this app validates as you type rather than erroring
// after a render -- and this is the same treatment for the same reason.

/// **Fails today**: `normalize` only collapses whitespace, so a pasted tag
/// carrying an escape byte is sent as it was typed and the machine refuses
/// the whole render.
@Test func aPastedControlCharacterIsStrippedRatherThanSent() {
    #expect(ClientTags.normalize(["ow\u{001B}ls"]) == ["owls"])
    #expect(ClientTags.normalize(["\u{0000}owls\u{007F}"]) == ["owls"])
    // C1 is `char::is_control()` on the Rust side too.
    #expect(ClientTags.normalize(["ow\u{0090}ls"]) == ["owls"])
}

/// A WHITESPACE control is collapsed, not removed -- `normalize_tag_name`
/// says so in as many words ("indistinguishable from a space once
/// collapsed"), and removing it would join two words the person separated.
@Test func aWhitespaceControlStillCollapsesToOneSpace() {
    #expect(ClientTags.normalize(["a\tb"]) == ["a b"])
    #expect(ClientTags.normalize(["a\nb"]) == ["a b"])
    #expect(ClientTags.normalize(["a\u{000B}b"]) == ["a b"])
    #expect(ClientTags.normalize(["a \u{00A0} b"]) == ["a b"])
}

/// A tag that was NOTHING but control characters is empty, and an empty tag
/// is dropped -- the same rule as a tag that was only spaces.
@Test func aTagThatWasOnlyControlCharactersIsDropped() {
    #expect(ClientTags.normalize(["\u{0001}\u{0002}", "owls"]) == ["owls"])
}

/// Stripping happens BEFORE the duplicate fold and the length cap, so the
/// tag that is compared and counted is the one that will be sent.
@Test func strippingHappensBeforeTheFoldAndTheCap() {
    #expect(ClientTags.normalize(["owls", "ow\u{001B}ls"]) == ["owls"])
    let long = String(repeating: "a", count: ClientTags.maxTagChars) + "\u{001B}b"
    #expect(ClientTags.normalize([long])
            == [String(repeating: "a", count: ClientTags.maxTagChars)])
}

/// Everything a person might legitimately type is untouched -- emoji, an
/// accent, a `#`, a slash.
@Test func ordinaryCharactersAreLeftAlone() {
    #expect(ClientTags.normalize(["café ☕", "#owls", "a/b"]) == ["café ☕", "#owls", "a/b"])
}

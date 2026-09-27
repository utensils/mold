import Foundation
import Testing

@testable import MoldClient

// Each pinned against the Rust it ports: `compose_client_tags`
// (`crates/mold-core/src/organization.rs:107-148`) and `title_slug`
// (`crates/mold-core/src/print_title.rs:33`).

@Test func aTitleBecomesTheSlugItsFilenameCarries() {
    #expect(ClientTags.titleSlug("Smurf Village at dusk") == "smurf-village-at-dusk")
    #expect(ClientTags.titleSlug("!!! ???") == nil)
}

@Test func theSlugIsCutAtFortyBytesWithNoDanglingDash() throws {
    let long = "a very long title that keeps going and going and going and going"
    let slug = try #require(ClientTags.titleSlug(long))
    #expect(slug.utf8.count <= ClientTags.titleSlugMaxBytes)
    #expect(!slug.hasSuffix("-"))
}

@Test func aTagAlreadyTypedIsNotAddedTwiceByTheTitle() {
    let composed = ClientTags.compose(
        explicit: ["Smurf-Village"], title: "Smurf Village", autoTagTitle: true)
    #expect(composed.tags == ["Smurf-Village"])
    #expect(composed.autoTagged == nil)
}

@Test func duplicateTagsCollapseAndTheFirstSpellingWins() {
    #expect(ClientTags.normalize(["Owls", "owls", "OWLS"]) == ["Owls"])
}

@Test func whitespaceInsideATagCollapsesToOneSpace() {
    #expect(ClientTags.normalize(["smurf   village"]) == ["smurf village"])
}

@Test func theAutoTagCannotPushTheListPastTwenty() {
    let full = (0 ..< ClientTags.maxTags).map { "t\($0)" }
    let composed = ClientTags.compose(explicit: full, title: "Smurf Village", autoTagTitle: true)
    #expect(composed.tags.count == ClientTags.maxTags)
    #expect(composed.autoTagged == nil)
}

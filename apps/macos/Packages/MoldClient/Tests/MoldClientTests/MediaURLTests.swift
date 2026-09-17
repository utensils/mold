import Foundation
import Testing

@testable import MoldClient

// A media ticket used to be signed over `"/api/gallery/image/\(filename)"`
// raw, while the URL a player actually requests is percent-encoded. A
// filename with a space therefore signed one string and requested another,
// and the host's ticket check -- which compares against the encoded request
// path -- refused it. `mediaPath` is the one rule that produces both.

private let urls = MediaURL(baseURL: URL(string: "http://plato:7680")!)

@Test func aFilenameWithASpaceEncodesTheSameWayForSigningAndForFetching() {
    #expect(urls.mediaPath("a b.png") == "/api/gallery/image/a%20b.png")
    #expect(urls.mediaPath("a b.png") == urls.media("a b.png").path(percentEncoded: true))
}

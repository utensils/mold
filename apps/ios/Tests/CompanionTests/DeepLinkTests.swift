import Foundation
import Testing

@testable import MoldCompanion

/// `moldstudio://` round-trips, and the Tauri app's `mold://` is never read.
struct DeepLinkTests {
    @Test(arguments: [
        DeepLink.print(host: UUID(), filename: "mold-flux-1718.png"),
        DeepLink.print(host: UUID(), filename: "a lighthouse at dusk.png"),
        DeepLink.queue(job: nil), DeepLink.queue(job: "j-42"),
        DeepLink.generate(inbox: nil), DeepLink.generate(inbox: "ABC"),
    ])
    func everyLinkRoundTrips(_ link: DeepLink) {
        #expect(DeepLink(link.url) == link)
        #expect(link.url.scheme == "moldstudio")
    }

    @Test func theTauriSchemeAndJunkAreRefused() {
        #expect(DeepLink(URL(string: "mold://pair?token=x")!) == nil)
        #expect(DeepLink(URL(string: "moldstudio://print/not-a-uuid/a.png")!) == nil)
        #expect(DeepLink(URL(string: "moldstudio://elsewhere")!) == nil)
    }
}

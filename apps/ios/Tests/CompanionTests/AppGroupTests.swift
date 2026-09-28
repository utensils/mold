import Foundation
import Testing

@testable import MoldCompanion

/// The widget and the Share extension only ever talk to the app through this
/// container. A signing change that drops the entitlement (ad-hoc simulator
/// signing included) makes it `nil`, and every hand-off would fail silently.
struct AppGroupTests {
    @Test func theSharedContainerResolves() throws {
        let container = try #require(AppGroup.container)
        #expect(FileManager.default.fileExists(atPath: container.path(percentEncoded: false)))
    }
}

import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

/// Notifications: the words per kind, a link that opens the right place, the
/// Settings switches honoured, and one per batch.
@MainActor
struct NotifierTests {
    private func notifier() -> (Notifier, UserDefaults) {
        let defaults = UserDefaults(suiteName: "notifier-\(UUID())")!
        return (Notifier(defaults: defaults, center: nil), defaults)
    }

    private let batch = ActiveBatch(id: "b1", clientBatchId: "c1", host: UUID(), prompt: "a lighthouse at dusk",
                                    startedAt: .now)

    @Test func aFinishedRenderLinksToItsPrintAndThreadsByMachine() throws {
        let (notifier, _) = notifier()
        let print = PrintID(host: batch.host, filename: "a.png")
        let content = try #require(notifier.content(.finished(count: 1), batch: batch, machine: "workstation", print: print))
        #expect(content.title == "Finished on workstation")
        #expect(content.body == "a lighthouse at dusk")
        #expect(content.threadIdentifier == batch.host.uuidString)
        #expect(content.userInfo["link"] as? String == DeepLink.print(host: batch.host, filename: "a.png").url.absoluteString)
    }

    @Test func aKindSwitchedOffInSettingsSaysNothing() {
        let (notifier, defaults) = notifier()
        defaults.set(false, forKey: Preference.notifyHeld)
        #expect(notifier.content(.held("Model missing"), batch: batch, machine: "m", print: nil) == nil)
        #expect(notifier.content(.failed("Out of memory"), batch: batch, machine: "m", print: nil) != nil)
    }

    @Test func aBatchIsAnnouncedOnce() {
        let (notifier, _) = notifier()
        notifier.post(.finished(count: 1), batch: batch, machine: "m", print: nil)
        #expect(notifier.content(.finished(count: 1), batch: batch, machine: "m", print: nil) == nil)
    }
}

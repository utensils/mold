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
        #expect(content.title == "Render complete")
        #expect(content.body == "Your print is ready on workstation.")
        #expect(content.threadIdentifier == batch.host.uuidString)
        #expect(content.userInfo["link"] as? String == DeepLink.print(host: batch.host, filename: "a.png").url.absoluteString)
    }

    @Test func aBatchUsesConciseCopyWithoutThePrompt() throws {
        let (notifier, _) = notifier()
        let content = try #require(notifier.content(.finished(count: 4), batch: batch, machine: "workstation", print: nil))
        #expect(content.title == "Render complete")
        #expect(content.body == "4 prints are ready on workstation.")
        #expect(content.attachments.isEmpty)
        #expect(content.userInfo["link"] as? String == DeepLink.queue(job: nil).url.absoluteString)
    }

    @Test func theAppBundleProvidesTheSystemNotificationIcon() throws {
        let bundle = Bundle(for: Notifier.self)
        let icons = try #require(bundle.object(forInfoDictionaryKey: "CFBundleIcons") as? [String: Any])
        let primary = try #require(icons["CFBundlePrimaryIcon"] as? [String: Any])
        #expect(primary["CFBundleIconName"] as? String == "AppIcon")
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

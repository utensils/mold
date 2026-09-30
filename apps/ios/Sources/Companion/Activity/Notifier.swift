import Foundation
import MoldClient
@preconcurrency import UserNotifications

/// Local notifications for renders that settle while the app is away
/// (DESIGN.md §5.10). The server has no push: these come from the app as it
/// follows a render into the background, and from background refresh. Never
/// shown in the foreground -- the inline result covers it -- and posted once
/// per batch, threaded per machine.
@Observable
final class Notifier: NSObject {
    enum Kind: Equatable {
        case finished(count: Int)
        case failed(String)
        case held(String)
    }

    /// Where a tapped notification wants to go; the window's `RootView`
    /// takes it.
    var link: DeepLink?
    @ObservationIgnored var favourite: ((PrintID) async -> Void)?
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let center: UNUserNotificationCenter?

    static let finishedCategory = "finished"
    static let heldCategory = "held"
    static let failedCategory = "failed"
    nonisolated static let viewAction = "view"
    static let favouriteAction = "favourite"

    init(defaults: UserDefaults = .standard, center: UNUserNotificationCenter? = .current()) {
        self.defaults = defaults
        self.center = center
        super.init()
        center?.delegate = self
        center?.setNotificationCategories([
            UNNotificationCategory(identifier: Self.finishedCategory, actions: [
                UNNotificationAction(identifier: Self.viewAction, title: String(localized: "View"), options: [.foreground]),
                UNNotificationAction(identifier: Self.favouriteAction, title: String(localized: "Favourite")),
            ], intentIdentifiers: []),
            UNNotificationCategory(identifier: Self.heldCategory, actions: [
                UNNotificationAction(identifier: Self.viewAction, title: String(localized: "View in Queue"), options: [.foreground]),
            ], intentIdentifiers: []),
            UNNotificationCategory(identifier: Self.failedCategory, actions: [
                UNNotificationAction(identifier: Self.viewAction, title: String(localized: "View"), options: [.foreground]),
            ], intentIdentifiers: []),
        ])
    }

    /// Asked once, the first time a render is sent: the moment the reason
    /// for asking is obvious.
    func requestAuthorization() async {
        _ = try? await center?.requestAuthorization(options: [.alert, .sound, .badge])
    }

    /// The words and the link for one settled batch, or `nil` when that kind
    /// is switched off in Settings or this batch was already announced.
    func content(_ kind: Kind, batch: ActiveBatch, machine: String, print: PrintID?) -> UNMutableNotificationContent? {
        let (key, category, title, body, link): (String, String, String, String, DeepLink) = switch kind {
        case let .finished(count):
            (Preference.notifyFinished, Self.finishedCategory,
             String(localized: "Render complete"),
             count > 1 ? String(localized: "\(count) prints are ready on \(machine).")
                 : String(localized: "Your print is ready on \(machine)."), print.map { .print(host: $0.host, filename: $0.filename) } ?? .queue(job: nil))
        case let .failed(reason):
            (Preference.notifyFailed, Self.failedCategory, String(localized: "Didn't finish on \(machine)"), reason,
             .generate(inbox: nil))
        case let .held(reason):
            (Preference.notifyHeld, Self.heldCategory, String(localized: "Waiting on \(machine)"), reason, .queue(job: nil))
        }
        guard Preference.isOn(key, defaults: defaults), !announced.contains(batch.clientBatchId) else { return nil }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.categoryIdentifier = category
        content.threadIdentifier = batch.host.uuidString
        content.userInfo = ["link": link.url.absoluteString]
        return content
    }

    func post(_ kind: Kind, batch: ActiveBatch, machine: String, print: PrintID?) {
        guard let content = content(kind, batch: batch, machine: machine, print: print) else { return }
        remember(batch.clientBatchId)
        // iOS supplies the app icon and native layout. No prompt or media
        // attachment: keep the completion banner compact and private.
        center?.add(UNNotificationRequest(identifier: batch.clientBatchId, content: content, trigger: nil))
    }

    // MARK: - Once per batch

    private static let announcedKey = "notify.announced"
    private var announced: [String] { defaults.stringArray(forKey: Self.announcedKey) ?? [] }
    private func remember(_ id: String) {
        defaults.set(Array((announced + [id]).suffix(64)), forKey: Self.announcedKey)
    }

    fileprivate func handle(action: String, link: String?) async {
        guard let link = link.flatMap(URL.init(string:)).flatMap(DeepLink.init) else { return }
        if action == Self.favouriteAction, case let .print(host, filename) = link {
            await favourite?(PrintID(host: host, filename: filename))
        } else {
            self.link = link
        }
    }
}

extension Notifier: UNUserNotificationCenterDelegate {
    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification)
        async -> UNNotificationPresentationOptions {
        // In the foreground the result is already on screen.
        []
    }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter,
                                            didReceive response: UNNotificationResponse) async {
        let action = response.actionIdentifier == UNNotificationDefaultActionIdentifier ? Self.viewAction : response.actionIdentifier
        let link = response.notification.request.content.userInfo["link"] as? String
        await handle(action: action, link: link)
    }
}

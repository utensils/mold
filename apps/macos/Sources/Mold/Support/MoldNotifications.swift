import Foundation
import MoldClient
import UserNotifications

/// What `MoldNotifications` needs from `UNUserNotificationCenter`, kept
/// abstract so a test never touches the real center -- see the type doc
/// below for why touching it at all is dangerous outside a bundle.
protocol NotificationCenterProtocol: AnyObject {
    func requestAuthorization(
        options: UNAuthorizationOptions, completionHandler: @escaping @Sendable (Bool, (any Error)?) -> Void)
    func add(
        _ request: UNNotificationRequest,
        withCompletionHandler completionHandler: (@Sendable ((any Error)?) -> Void)?)
}

extension UNUserNotificationCenter: NotificationCenterProtocol {}

/// Notifications for work that finished or failed while you were elsewhere.
///
/// **Every entry point is guarded by `isInsideBundle`.** Touching
/// `UNUserNotificationCenter` from a binary that is not inside a real `.app`
/// raises an UNCATCHABLE `NSInternalInconsistencyException` and takes the
/// whole process down (`desktop/src-tauri/src/notifications.rs:159-183`,
/// which hit `tauri dev`). The guard is a PATH check on the executable, not
/// `Bundle.main.bundleIdentifier` -- a bundle identifier can be swizzled
/// process-wide by a dependency, so an identifier-based guard passes and
/// aborts a second time; no dependency can swizzle `executablePath`.
/// `make run` and `make test` both go through `Mold Studio.app`, so this is defence
/// against a stray `swift run` during a slice, which is exactly when it
/// would bite.
///
/// Authorization is requested on FIRST NEED -- the first time there is
/// something to say and the preference is on -- never at launch: a
/// permission prompt before the app has done anything is a prompt about
/// nothing.
@MainActor
@Observable
final class MoldNotifications {
    private let landedPrints: LandedPrints
    private let queue: QueueStore
    private let hosts: HostStore
    private let library: LibraryStore
    /// Not `private`: `+Delivery.swift` hands it every request, and
    /// `private` does not cross a file boundary even within one type.
    let center: any NotificationCenterProtocol
    private let defaults: UserDefaults
    private let coalesceDelay: Duration
    private let isBundled: Bool

    /// One burst per machine -- `flushFinished` empties it.
    private var pendingFinished: [MoldHost.ID: [LandedPrints.Landing]] = [:]
    private var coalescers: [MoldHost.ID: Task<Void, Never>] = [:]
    /// The one authorization request, once there is something to say.
    /// Written from `+Delivery.swift`, same cross-file reason as `center`.
    var authorization: Task<Void, Never>?
    /// Every notification this object has handed the centre, chained: each
    /// waits for the one before it, and the first waits for authorization to
    /// be ANSWERED. Not `private(set)` for the app's sake -- nothing reads it
    /// -- but for the tests', which await this instead of polling. Written
    /// from `+Delivery.swift`, so `internal` rather than `private(set)`.
    var deliveries: Task<Void, Never>?

    /// Read live, the same reason `LandedPrints.enabled` is: this is a plain
    /// object, and the preference can change under it at any time.
    var enabled: Bool {
        defaults.object(forKey: "notifyRenders") == nil ? true : defaults.bool(forKey: "notifyRenders")
    }

    /// True only inside a `…/Foo.app/Contents/MacOS/…` executable. A default
    /// argument rather than reading `Bundle.main` inline so a test can hand
    /// it a synthetic path with no app bundle at all.
    static func isInsideBundle(
        _ path: String = Bundle.main.executablePath ?? CommandLine.arguments.first ?? ""
    ) -> Bool {
        path.contains(".app/Contents/MacOS/")
    }

    init(
        landedPrints: LandedPrints, queue: QueueStore, hosts: HostStore, library: LibraryStore,
        center: any NotificationCenterProtocol = UNUserNotificationCenter.current(),
        defaults: UserDefaults = AppStorageSuite.defaults,
        coalesceDelay: Duration = .seconds(2), executablePath: String = Bundle.main.executablePath ?? ""
    ) {
        self.landedPrints = landedPrints
        self.queue = queue
        self.hosts = hosts
        self.library = library
        self.center = center
        self.defaults = defaults
        self.coalesceDelay = coalesceDelay
        isBundled = Self.isInsideBundle(executablePath)
        landedPrints.onLanding = { [weak self] landing in self?.noteFinished(landing) }
        queue.onOutcome = { [weak self] host, entry, sentence in self?.noteFailed(host, entry, sentence) }
    }

    /// Only while Mold is in the background -- the General tab's own words
    /// for the toggle. `LandedPrints` already gates its own arrivals this
    /// way; a failure notification follows suit so the one toggle means one
    /// thing for both.
    private var shouldNotify: Bool { isBundled && enabled && !landedPrints.isActive }

    private func noteFinished(_ landing: LandedPrints.Landing) {
        guard shouldNotify else { return }
        pendingFinished[landing.host, default: []].append(landing)
        guard coalescers[landing.host] == nil else { return }
        coalescers[landing.host] = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: coalesceDelay)
            guard !Task.isCancelled else { return }
            flushFinished(landing.host)
        }
    }

    /// One notification per burst: the newest print's prompt as the body
    /// when `LibraryStore` already has the row, else its bare filename --
    /// never invented.
    private func flushFinished(_ host: MoldHost.ID) {
        coalescers[host] = nil
        guard let landings = pendingFinished.removeValue(forKey: host), let newest = landings.last
        else { return }
        let machine = hosts.host(host)?.name ?? "that machine"
        let title = landings.count == 1
            ? "Finished on \(machine)" : "\(landings.count) prints finished on \(machine)"
        let id = PrintID(host: host, filename: newest.filename)
        let body = library.items.first { $0.id == id }?.print.metadata.prompt ?? newest.filename
        post(title: title, body: body, userInfo: ["kind": "print", "host": host.uuidString, "filename": id.filename])
    }

    /// `QueueStore.onOutcome`'s one listener -- a child moving to `failed`,
    /// or to a hold the host says trying again would not help (decision 24).
    /// Never coalesced: each is a distinct job with its own sentence, unlike
    /// a burst of otherwise-identical "finished" frames.
    private func noteFailed(_ host: MoldHost.ID, _ entry: QueueEntry, _ sentence: String) {
        guard shouldNotify else { return }
        let machine = hosts.host(host)?.name ?? "that machine"
        post(title: "Failed on \(machine)", body: sentence, userInfo: ["kind": "failure", "host": host.uuidString])
    }
}

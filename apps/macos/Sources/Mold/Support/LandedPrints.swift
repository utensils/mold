import Foundation
import MoldClient

/// Prints that landed on ANY machine while Mold was in the background.
///
/// Distinct filenames, whoever asked for them -- never a count of this app's
/// own queue. The desktop tried that first and wrote down why it stopped
/// (`desktop/src/stores/landedPrints.ts`): a long clip left a number nobody
/// could clear, and work another machine did never showed at all.
///
/// Best-effort and in-memory by design: it resets on relaunch, because a
/// badge is about what happened while you were away, not a log.
@MainActor
@Observable
final class LandedPrints {
    /// One arrival: which machine reported it, and when -- what S5b's
    /// per-machine coalescer reads to build its finished notification.
    struct Landing: Hashable {
        let host: MoldHost.ID
        let filename: String
        let at: Date
    }

    private let defaults: UserDefaults
    private(set) var recent: [Landing] = []

    /// Fired for each new arrival, after it is appended -- `MoldNotifications`
    /// coalesces from this rather than polling `recent` itself.
    var onLanding: ((Landing) -> Void)?

    /// Distinct filenames across every machine -- decision 21: the SAME print
    /// echoed by two machines (a remote render auto-saved here too) is one
    /// arrival, not two.
    var count: Int { Set(recent.map(\.filename)).count }

    /// `NSApp.isActive` in production, mirrored by `MoldAppDelegate`; a test
    /// drives it directly. Becoming active is what a person coming back to
    /// the app looks like, so it clears what accumulated while they were
    /// away.
    var isActive: Bool = false {
        didSet {
            guard isActive, !oldValue else { return }
            clear()
        }
    }

    /// Read live off `UserDefaults` rather than through `@AppStorage`: this
    /// is a plain object, not a view, and the preference can change under it
    /// at any time.
    var enabled: Bool {
        defaults.object(forKey: "badgeLandedPrints") == nil
            ? true
            : defaults.bool(forKey: "badgeLandedPrints")
    }

    init(hosts: HostStore, defaults: UserDefaults = AppStorageSuite.defaults) {
        self.defaults = defaults
        // For the life of the app -- `LibraryStore.swift`'s shape. See
        // `HostStore+Events`.
        hosts.onEvent { [weak self] host, event in self?.apply(event, from: host) }
        // Turning the preference off must not merely stop counting -- a stale
        // number nobody can clear is the exact failure the desktop wrote down.
        // KVO rather than `GeneralSettings`' own `onChange`: this is a plain
        // object, and the toggle can flip while no Settings window is even
        // open.
        NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification, object: defaults, queue: nil
        ) { [weak self] _ in
            Task { @MainActor in self?.reconcileEnabled() }
        }
    }

    private func reconcileEnabled() {
        guard !enabled, !recent.isEmpty else { return }
        clear()
    }

    private func apply(_ event: MoldEvent, from host: MoldHost.ID) {
        guard case let .gallery(.added(filename, _)) = event else { return }
        guard !isActive, enabled else { return }
        guard !recent.contains(where: { $0.host == host && $0.filename == filename }) else { return }
        let landing = Landing(host: host, filename: filename, at: Date())
        recent.append(landing)
        onLanding?(landing)
    }

    /// Coming back to the app, or turning the preference off -- either way
    /// the badge stops showing a number nobody asked for.
    func clear() {
        recent.removeAll()
    }
}

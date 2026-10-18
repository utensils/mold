import AppKit
import MoldClient

/// A file URL stays valid until the clipboard changes or its source is revoked.
@MainActor
final class SourceMediaClipboard {
    static let shared = SourceMediaClipboard()
    private(set) var file: SourceMediaFile?
    private var changeCount = 0
    private var monitor: Task<Void, Never>?
    private var terminationObserver: (any NSObjectProtocol)?
    init(center: NotificationCenter = .default) {
        terminationObserver = center.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.clear() }
        }
    }
    func copy(_ file: SourceMediaFile, pasteboard: NSPasteboard = .general,
              isCurrent: @escaping @MainActor () -> Bool) {
        clear()
        guard isCurrent() else { return }
        pasteboard.clearContents()
        guard pasteboard.writeObjects([file.url as NSURL]) else { return }
        self.file = file; changeCount = pasteboard.changeCount
        monitor = Task { [weak self] in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .milliseconds(250)) } catch { return }
                guard let self else { return }
                if pasteboard.changeCount != self.changeCount || !isCurrent() { self.clear(); return }
            }
        }
    }
    func clear() { monitor?.cancel(); monitor = nil; file = nil }
}

#if DEBUG
import AppKit

/// `MOLD_NATIVE_UAT_SCRIPT=<path>` drives the app from a text file, one step
/// per line, and writes what happened to `<path>.log`.
///
/// It exists so a UAT can run on a headless Mac over SSH. Everything there
/// that a person would reach for is gated behind a permission nobody is
/// present to grant -- Screen Recording for `screencapture`, Accessibility
/// for an AX press -- so the app does both jobs itself: a menu item is
/// performed through its own `NSMenu`, and a window renders itself to a PNG.
/// Nothing here takes focus or moves a pointer, so it is also safe on a Mac
/// somebody is using.
///
///     wait 2
///     menu View > Library
///     snapshot /tmp/library.png
///     quit
@MainActor
enum UATScript {
    static func runIfRequested(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        responses: NotificationResponses? = nil
    ) {
        guard let path = NativeUAT.script.value(in: environment),
              let script = try? String(contentsOfFile: path, encoding: .utf8)
        else { return }
        Task { await run(steps(in: script), log: URL(fileURLWithPath: path + ".log"), responses: responses) }
    }

    /// Blank lines and `#` comments are not steps.
    static func steps(in script: String) -> [String] {
        script.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }

    private static func run(_ steps: [String], log: URL, responses: NotificationResponses?) async {
        var lines: [String] = []
        for step in steps {
            let (verb, argument) = split(step)
            let outcome: String
            switch verb {
            case "wait":
                try? await Task.sleep(for: .seconds(Double(argument) ?? 1))
                outcome = "ok"
            case "menu": outcome = perform(menuPath: argument)
            case "snapshot": outcome = snapshot(to: argument)
            case "windows": outcome = NSApp.windows.filter(\.isVisible).map(\.title).joined(separator: " | ")
            case "notify": outcome = await UATNotification.post(argument)
            case "notification-response": outcome = UATNotification.deliver(argument, to: responses)
            case "quit":
                write(lines + ["quit: ok"], to: log)
                NSApp.terminate(nil)
                return
            default: outcome = "unknown step"
            }
            lines.append("\(step): \(outcome)")
            write(lines, to: log)
        }
    }

    private static func split(_ step: String) -> (String, String) {
        let parts = step.split(separator: " ", maxSplits: 1).map(String.init)
        return (parts[0], parts.count > 1 ? parts[1] : "")
    }

    /// `A > B > C` walks the main menu by title. `update()` runs at every
    /// level so the answer is the enablement a person would see.
    private static func perform(menuPath: String) -> String {
        guard let menu = NSApp.mainMenu else { return "no main menu" }
        return perform(menuPath.components(separatedBy: " > ")[...], in: menu)
    }

    private static func perform(_ titles: ArraySlice<String>, in menu: NSMenu) -> String {
        menu.update()
        guard let title = titles.first, let item = menu.items.first(where: { $0.title == title })
        else { return "no item “\(titles.first ?? "")”" }
        if titles.count == 1 {
            guard item.isEnabled else { return "disabled" }
            menu.performActionForItem(at: menu.index(of: item))
            return "ok"
        }
        guard let submenu = item.submenu else { return "“\(title)” has no submenu" }
        return perform(titles.dropFirst(), in: submenu)
    }

    /// Every visible window, frame included, so a toolbar is in the picture.
    /// The first goes to `path`; the rest to `path` with `.1`, `.2`… before
    /// the extension.
    private static func snapshot(to path: String) -> String {
        let base = URL(fileURLWithPath: path)
        let windows = NSApp.windows.filter { $0.isVisible && $0.contentView != nil }
        for (index, window) in windows.enumerated() {
            guard let view = window.contentView?.superview ?? window.contentView,
                  let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds)
            else { continue }
            view.cacheDisplay(in: view.bounds, to: bitmap)
            let name = index == 0 ? base : base.deletingPathExtension()
                .appendingPathExtension("\(index).png")
            try? bitmap.representation(using: .png, properties: [:])?.write(to: name)
        }
        return "\(windows.count) window(s)"
    }

    private static func write(_ lines: [String], to log: URL) {
        try? lines.joined(separator: "\n").appending("\n").write(to: log, atomically: true, encoding: .utf8)
    }
}
#endif

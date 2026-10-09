import AppKit

/// Quitting while the engine is still draining.
///
/// `applicationShouldTerminate` answers `.terminateLater` and macOS then waits
/// with no indication of what for. The engine's budget is the server's own
/// (`EngineShutdownBudget`), which is 45 s rather than the 8 s the app used to
/// allow — long enough that a silent wait reads as a hang. This is the one
/// sentence saying why, and the one way out that does not lose the drain by
/// accident (review 05-M7).
@MainActor
final class EngineQuit {
    private var panel: NSPanel?
    private var replied = false

    /// Puts the panel up. Deliberately not modal: the drain runs on the main
    /// runloop's Tasks, and a modal session would starve them.
    func present(seconds: UInt64) {
        guard panel == nil else { return }
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 360, height: 118),
            styleMask: [.titled, .utilityWindow],
            backing: .buffered, defer: false)
        panel.title = "Quitting Mold"
        panel.isFloatingPanel = true
        panel.hidesOnDeactivate = false
        let content = Self.content(seconds: seconds, quitNow: { [weak self] in
            self?.reply()
        })
        panel.contentView = content
        panel.setContentSize(content.fittingSize)
        panel.center()
        panel.makeKeyAndOrderFront(nil)
        self.panel = panel
    }

    /// Replies to macOS at most once: the drain finishing and Quit Now can
    /// both arrive, and a second reply is a double answer to one question.
    func reply() {
        guard !replied else { return }
        replied = true
        panel?.orderOut(nil)
        panel = nil
        NSApplication.shared.reply(toApplicationShouldTerminate: true)
    }

    private static func content(seconds: UInt64, quitNow: @escaping () -> Void) -> NSView {
        let spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.startAnimation(nil)

        let label = NSTextField(wrappingLabelWithString:
            "Finishing this Mac's renders and closing the library cleanly. "
            + "Up to \(seconds) seconds.")
        label.font = .preferredFont(forTextStyle: .body)

        let button = NSButton(title: "Quit Now", target: QuitNowTarget.shared, action: nil)
        button.bezelStyle = .rounded
        QuitNowTarget.shared.attach(button, quitNow)
        button.keyEquivalent = "\u{1b}"

        let row = NSStackView(views: [spinner, label])
        row.alignment = .top
        row.spacing = 8
        let stack = NSStackView(views: [row, button])
        stack.orientation = .vertical
        stack.alignment = .trailing
        stack.spacing = 12
        // A stack used directly as a window's content view does not preserve
        // its edge insets. Pin the stack inside an ordinary container instead.
        let content = NSView()
        content.addSubview(stack)
        stack.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            content.widthAnchor.constraint(equalToConstant: 360),
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 16),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -16),
            row.widthAnchor.constraint(equalTo: stack.widthAnchor),
        ])
        return content
    }
}

/// `NSButton` wants an Objective-C target; this is the smallest one that can
/// hold a Swift closure.
@MainActor
private final class QuitNowTarget: NSObject {
    static let shared = QuitNowTarget()
    private var action: (() -> Void)?

    func attach(_ button: NSButton, _ action: @escaping () -> Void) {
        self.action = action
        button.target = self
        button.action = #selector(fire)
    }

    @objc private func fire() { action?() }
}

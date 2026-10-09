import AppKit
import Testing
@testable import Mold

@MainActor struct EngineQuitLayoutTests {
    @Test func quitNowHasNativeButtonPaddingAndFitsBelowTheMessage() async throws {
        let quit = EngineQuit()
        quit.present(seconds: 45)
        let panel = try #require(NSApplication.shared.windows.first { $0.title == "Quitting Mold" })
        defer { panel.close() }
        try await Task.sleep(for: .milliseconds(200))
        let content = try #require(panel.contentView)
        content.wantsLayer = true
        content.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        content.layoutSubtreeIfNeeded()
        let button = try #require(descendant(NSButton.self, in: content))
        let label = try #require(descendant(NSTextField.self, in: content))
        let buttonFrame = button.convert(button.bounds, to: content)
        // AppKit lays out text fields by their optical alignment rectangle;
        // their raw bounds include the text cell's transparent edge padding.
        let labelFrame = label.convert(label.alignmentRect(forFrame: label.bounds), to: content)
        #expect(button.bezelStyle == .rounded, "Quit Now needs native horizontal and vertical button padding")
        #expect(buttonFrame.minX >= 20 && content.bounds.maxX - buttonFrame.maxX >= 20)
        #expect(buttonFrame.minY >= 16, "The action must retain the panel's bottom margin")
        #expect(labelFrame.minY - buttonFrame.maxY >= 12, "The wrapped message must stay clear of Quit Now")
        #expect(content.bounds.maxY - labelFrame.maxY >= 16)
        #expect(labelFrame.minX >= 20 && labelFrame.maxX <= content.bounds.maxX - 20)
        let bitmap = try #require(content.bitmapImageRepForCachingDisplay(in: content.bounds))
        content.cacheDisplay(in: content.bounds, to: bitmap)
        try #require(bitmap.representation(using: .png, properties: [:]))
            .write(to: URL(fileURLWithPath: "/tmp/mac-engine-quit-layout.png"))
    }

    private func descendant<V: NSView>(_ type: V.Type, in view: NSView) -> V? {
        if let match = view as? V { return match }
        return view.subviews.lazy.compactMap { descendant(type, in: $0) }.first
    }
}

import AppKit
import MoldClient
import SwiftUI
import Testing

@testable import Mold

@MainActor
struct PromptEditorPresentationTests {
    @Test(arguments: [CGFloat(580), CGFloat(760)])
    func nativeEditorKeepsLongTextAndNewlinesLive(width: CGFloat) async throws {
        let host = MoldHost(name: "Editor fixture", baseURL: URL(string: "http://fixture.invalid")!)
        let backend = FakeBackend(host: host)
        backend.historyRows = []
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.draft.prompt = (0..<30).map {
            "Line \($0): A quiet coastal landscape, soft morning light and detailed textures."
        }.joined(separator: "\n")
        let initial = controller.draft.prompt
        let editor = PromptEditorSheet(
            draft: Binding(get: { controller.draft }, set: { controller.draft = $0 }),
            recipe: FakeFixtures.recipe(), host: host, destination: .constant(.generate)
        )
        .environment(hosts).environment(controller)
        .environment(PromptHistoryStore(hosts: hosts)).environment(ExpandStore())
        let view = NSHostingView(rootView: editor)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: width, height: 640),
            styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFront(nil)
        defer { window.close() }
        for _ in 0..<8 { await Task.yield() }
        view.layoutSubtreeIfNeeded()
        let text = try #require(textViews(in: view).first)
        #expect(text.string == initial)
        #expect(text.frame.height > 300)
        text.setSelectedRange(NSRange(location: (initial as NSString).length, length: 0))
        text.insertText("\nAdditional line", replacementRange: text.selectedRange())
        for _ in 0..<8 { await Task.yield() }
        #expect(controller.draft.prompt == initial + "\nAdditional line")
        #expect(backend.callCount("generate") == 0)
        if let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
            view.cacheDisplay(in: view.bounds, to: bitmap)
            try bitmap.representation(using: .png, properties: [:])?.write(
                to: URL(fileURLWithPath: "/tmp/mold-prompt-editor-\(Int(width)).png"))
        }
    }

    @Test(arguments: [false, true])
    func actualNativeSheetHasReadableLightAndDarkPresentation(dark: Bool) async throws {
        let host = MoldHost(name: "Editor fixture", baseURL: URL(string: "http://fixture.invalid")!)
        let backend = FakeBackend(host: host)
        backend.historyRows = []
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.draft.prompt =
            "A quiet coastal landscape, soft morning light and detailed textures.\n\nKeep the horizon low, leave room for the open sky, and use a calm palette."
        let editor = PromptEditorSheet(
            draft: Binding(get: { controller.draft }, set: { controller.draft = $0 }),
            recipe: FakeFixtures.recipe(), host: host, destination: .constant(.generate)
        )
        .environment(hosts).environment(controller)
        .environment(PromptHistoryStore(hosts: hosts)).environment(ExpandStore())
        let presentation = Presentation()
        let root = Button("Edit prompt") { presentation.open = true }
            .frame(width: 900, height: 740)
            .sheet(isPresented: Binding(get: { presentation.open }, set: { presentation.open = $0 })) { editor }
            .preferredColorScheme(dark ? .dark : .light)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 740),
            styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentView = NSHostingView(rootView: root)
        window.orderFront(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(250))
        let sheet = try #require(window.sheets.first)
        let view = try #require(sheet.contentView?.superview)
        view.layoutSubtreeIfNeeded()
        #expect(textViews(in: view).first?.string == controller.draft.prompt)
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        try bitmap.representation(using: .png, properties: [:])?.write(
            to: URL(fileURLWithPath: "/tmp/mold-prompt-native-sheet-\(dark ? "dark" : "light").png")
        )
        window.sheets.first?.orderOut(nil)
    }

    @Observable final class Presentation {
        var open = true
    }

    private func textViews(in view: NSView) -> [NSTextView] {
        (view as? NSTextView).map { [$0] } ?? view.subviews.flatMap { textViews(in: $0) }
    }
}

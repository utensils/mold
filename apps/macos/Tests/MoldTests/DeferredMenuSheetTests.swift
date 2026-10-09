import AppKit
import MoldClient
import Observation
import SwiftUI
import Testing
@testable import Mold

@MainActor
@Suite(.serialized)
struct DeferredMenuSheetTests {
    @Test func observedRequestPresentsDismissesAndReopensNativeSheet() async throws {
        let state = SheetRequest()
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 400, height: 240),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: RequestFixture(state: state))
        window.makeKeyAndOrderFront(nil)
        defer { window.close() }
        state.requested = true
        try await waitUntil { window.attachedSheet != nil }
        state.requested = false
        try await waitUntil { window.attachedSheet == nil }
        state.requested = true
        try await waitUntil { window.attachedSheet != nil }
        state.requested = false
        try await waitUntil { window.attachedSheet == nil }
    }

    @Test(arguments: ["return", "escape", "click"])
    func reportActionsDismissWithoutReopening(action: String) async throws {
        let hosts = HostStore(hosts: []) { FakeBackend(host: $0) }
        let library = LibraryStore(hosts: hosts)
        library.localSaveReport = "All prints are already saved on This Mac."
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 700, height: 400),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: ReportFixture().environment(library))
        window.makeKeyAndOrderFront(nil)
        defer { window.close() }
        library.localSaveAlertPresented = true
        try await waitUntil { window.attachedSheet != nil }
        let sheet = try #require(window.attachedSheet)
        try await Task.sleep(for: .milliseconds(700))
        let content = try #require(sheet.contentView)
        content.layoutSubtreeIfNeeded()
        if action == "click" {
            let bitmap = try #require(content.bitmapImageRepForCachingDisplay(in: content.bounds))
            content.cacheDisplay(in: content.bounds, to: bitmap)
            try #require(bitmap.representation(using: .png, properties: [:]))
                .write(to: URL(fileURLWithPath: "/tmp/mac-sync-details-presented.png"))
            // The fixture's rendered trailing Done button; retain the image
            // above so the actual mouse target can be verified visually.
            let point = NSPoint(x: content.bounds.maxX - 50, y: 45)
            for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
                let event = try #require(NSEvent.mouseEvent(with: type, location: point,
                    modifierFlags: [], timestamp: 0, windowNumber: sheet.windowNumber,
                    context: nil, eventNumber: 0, clickCount: 1, pressure: 1))
                sheet.sendEvent(event)
            }
        } else {
            let characters = action == "return" ? "\r" : "\u{1b}"
            let event = try #require(NSEvent.keyEvent(with: .keyDown, location: .zero,
                modifierFlags: [], timestamp: 0, windowNumber: sheet.windowNumber,
                context: nil, characters: characters, charactersIgnoringModifiers: characters,
                isARepeat: false, keyCode: action == "return" ? 36 : 53))
            sheet.sendEvent(event)
        }
        try await waitUntil { window.attachedSheet == nil }
        #expect(!library.localSaveAlertPresented)
        library.localSaveReport = "Next sync completed."
        try await Task.sleep(for: .milliseconds(100))
        #expect(window.attachedSheet == nil)
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<100 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(condition(), "Native sheet presentation must follow its observed request")
    }
}

@MainActor @Observable private final class SheetRequest {
    var requested = false
}

private struct RequestFixture: View {
    let state: SheetRequest
    var body: some View {
        @Bindable var state = state
        Text("Library").frame(width: 400, height: 240)
            .deferredMenuSheet(isPresented: $state.requested) {
                Text("Details").frame(width: 300, height: 180)
            }
    }
}

private struct ReportFixture: View {
    @Environment(LibraryStore.self) private var library
    var body: some View {
        @Bindable var library = library
        Text("Library").frame(width: 700, height: 400)
            .deferredMenuSheet(isPresented: $library.localSaveAlertPresented) {
                LibrarySyncReportSheet()
            }
            // Match LibraryPane's sibling sheet presenters.
            .sheet(isPresented: .constant(false)) { EmptyView() }
            .sheet(isPresented: .constant(false)) { EmptyView() }
            .sheet(isPresented: .constant(false)) { EmptyView() }
            .sheet(isPresented: .constant(false)) { EmptyView() }
    }
}

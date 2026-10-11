import AppKit
import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor
struct DraftNavigationPersistenceTests {
    @Test func leavingGenerateReturnsBeforeWritingAndPersistsLatestDraft() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        let persistence = DraftPersistence(store: store, delay: .seconds(60))
        var draft = RenderDraft(); draft.prompt = "latest prompt"
        let descriptor = DraftDescriptor(draft, model: "fixture", family: nil, recipeID: nil)
        persistence.saveOnNavigation(descriptor)
        // No suspension yet: navigation must return before any filesystem write.
        #expect(store.load() == nil)
        for _ in 0..<200 {
            if store.load()?.prompt == draft.prompt { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("Navigation save did not complete without the typing debounce")
    }

    @Test func immediateQuitSupersedesQueuedNavigationSave() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        let persistence = DraftPersistence(store: store, delay: .seconds(60))
        var draft = RenderDraft(); draft.prompt = "navigation snapshot"
        persistence.saveOnNavigation(DraftDescriptor(draft, model: "fixture", family: nil, recipeID: nil))
        draft.prompt = "final quit snapshot"
        persistence.flush(DraftDescriptor(draft, model: "fixture", family: nil, recipeID: nil))
        #expect(store.load()?.prompt == draft.prompt)
        try await Task.sleep(for: .milliseconds(100))
        #expect(store.load()?.prompt == draft.prompt)
    }

    @Test func terminationAfterNavigationFlushesWithoutAMountedPane() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        let notifications = NotificationCenter()
        let persistence = DraftPersistence(store: store, delay: .seconds(60), notifications: notifications)
        var draft = RenderDraft(); draft.prompt = "saved after leaving Generate"
        var media = DraftMedia(); media.sourceImage = "AQID"
        let inputs = DraftInputSnapshot(media)
        persistence.saveOnNavigation(DraftDescriptor(draft, model: "fixture", family: nil, recipeID: nil), inputs: inputs)
        #expect(store.load() == nil)
        // No view/subscriber exists and the queued task has not run yet.
        notifications.post(name: NSApplication.willTerminateNotification, object: nil)
        let saved = try #require(store.load())
        #expect(saved.prompt == draft.prompt)
        #expect(try store.loadInputs(for: saved) == inputs)
    }

    @Test func laterPaneFlushCannotBeOverwrittenByLifetimeObserver() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        let notifications = NotificationCenter()
        let persistence = DraftPersistence(store: store, notifications: notifications)
        var draft = RenderDraft(); draft.prompt = "older pending draft"
        persistence.schedule(DraftDescriptor(draft, model: "fixture", family: nil, recipeID: nil))
        draft.prompt = "latest live draft"
        persistence.flush(DraftDescriptor(draft, model: "fixture", family: nil, recipeID: nil))
        notifications.post(name: NSApplication.willTerminateNotification, object: nil)
        #expect(store.load()?.prompt == draft.prompt)
    }

}

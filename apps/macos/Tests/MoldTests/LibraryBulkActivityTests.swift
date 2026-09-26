import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct LibraryBulkActivityTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    @Test func optimisticEditReportsItsTargetCountUntilTheOutboxSettles() async throws {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let edit = PrintEdit(change: .favorite(true),
                             targets: [workstation.id: ["one.png", "two.png"]])

        library.mutations.send(edit, in: library)

        #expect(library.mutations.progress == "Updating 0 of 2 print changes…")
        try await waitUntil { library.mutations.progress == nil }
        #expect(fake.callCount("mutate") == 1)
    }

    @Test func optimisticEditNeverRetargetsAnEditedHostAfterItsReply() async throws {
        let workstation = machine()
        let oldBackend = FakeBackend(host: workstation)
        oldBackend.delays["mutate"] = .milliseconds(100)
        let replacement = MoldHost(id: workstation.id, name: "replacement",
                                   baseURL: URL(string: "http://replacement")!)
        let replacementBackend = FakeBackend(host: replacement)
        replacementBackend.prints = [FakeFixtures.print("one.png")]
        let hosts = HostStore(hosts: [workstation]) { host in
            host.name == "replacement" ? replacementBackend : oldBackend
        }
        let library = LibraryStore(hosts: hosts)
        library.perHost[workstation.id] = [
            LibraryEntry(host: workstation, print: FakeFixtures.print("one.png"))
        ]

        library.setFavorite(true, on: library.perHost[workstation.id] ?? [])
        await oldBackend.entered("mutate")
        hosts.update(replacement)
        try await waitUntil { library.mutations.progress == nil }

        #expect(oldBackend.callCount("mutate") == 1)
        #expect(replacementBackend.callCount("mutate") == 0)
        #expect(replacementBackend.callCount("gallery") == 1)
        #expect(library.perHost[workstation.id]?.first?.print.isFavorite == false)
    }

    @Test func independentActivitiesDoNotHideEachOther() {
        let workstation = machine()
        let hosts = HostStore(hosts: [workstation]) { FakeBackend(host: $0) }
        let library = LibraryStore(hosts: hosts)

        let first = library.beginBulkActivity("Importing…")
        let second = library.beginBulkActivity("Saving…")
        library.endBulkActivity(first)

        #expect(library.bulkActivities == [second: "Saving…"])
    }

    @Test func importStaysVisibleAndDiscardsAReplyAfterItsHostIsRemoved() async throws {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.delays["importPrint"] = .milliseconds(100)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let folder = FileManager.default.temporaryDirectory
            .appending(path: "mold-import-status-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appending(path: "one.png")
        try Data("bytes".utf8).write(to: file)

        let sending = Task { await PrintImport(hosts: hosts, library: library)
            .send([file], to: workstation) }
        await fake.entered("importPrint")

        #expect(library.bulkActivities.values.contains {
            $0 == "Importing 1 of 1 files to workstation…"
        })
        hosts.remove(workstation)
        await sending.value
        #expect(library.bulkActivities.isEmpty)
        #expect(hosts.failures.isEmpty)
    }

    @Test func globalTagProgressRejectsAnOldHostsReplyAndReload() async throws {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.delays["renameTag"] = .milliseconds(100)
        fake.tagRows = [TagCount(name: "new", count: 99)]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        library.tags.perHost[workstation.id] = [TagCount(name: "old", count: 1)]

        library.tags.rename("old", to: "new", in: library)
        await fake.entered("renameTag")

        #expect(library.bulkActivities.values.contains {
            $0 == "Renaming tag on 1 of 1 machines…"
        })
        hosts.update(MoldHost(id: workstation.id, name: "replacement",
                              baseURL: URL(string: "http://replacement")!))
        try await waitUntil { library.bulkActivities.isEmpty }
        #expect(fake.callCount("tags") == 0)
        #expect(library.tags.perHost[workstation.id] == [TagCount(name: "old", count: 1)])
    }

    @Test func bulkSaveReportsEachPrintUntilDiskCopiesFinish() async throws {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.mediaAnswer = Data(repeating: 7, count: 1_024)
        fake.delays["media"] = .milliseconds(100)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let cache = FileManager.default.temporaryDirectory
            .appending(path: "mold-save-cache-\(UUID().uuidString)")
        let destination = FileManager.default.temporaryDirectory
            .appending(path: "mold-save-status-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: cache)
            try? FileManager.default.removeItem(at: destination)
        }
        let actions = LibraryActions(hosts: hosts, library: library,
                                     materializer: PrintMaterializer(root: cache))
        let entries = ["one.png", "two.png"].map {
            LibraryEntry(host: workstation, print: FakeFixtures.print($0))
        }

        let saving = Task { await actions.saveAll(entries, into: destination) }
        await fake.entered("media")

        #expect(library.bulkActivities.values.contains { $0 == "Saving 1 of 2 prints…" })
        await saving.value
        #expect(library.bulkActivities.isEmpty)
        #expect(Set(try FileManager.default.contentsOfDirectory(atPath: destination.path))
            == ["one.png", "two.png"])
    }

    @Test func fleetShelfProgressDoesNotRetargetAnEditedHost() async throws {
        let workstation = machine()
        let oldBackend = FakeBackend(host: workstation)
        oldBackend.delays["deleteCollection"] = .milliseconds(100)
        let replacement = MoldHost(id: workstation.id, name: "replacement",
                                   baseURL: URL(string: "http://replacement")!)
        let replacementBackend = FakeBackend(host: replacement)
        replacementBackend.collectionRows = []
        let hosts = HostStore(hosts: [workstation]) { host in
            host.name == "replacement" ? replacementBackend : oldBackend
        }
        let library = LibraryStore(hosts: hosts)
        let shelf = try #require(CollectionShelf.merge([
            workstation.id: [Collection(id: "old-id", name: "Drafts", slug: "drafts")]
        ]).first)

        let deleting = Task { await library.deleteShelf(shelf) }
        await oldBackend.entered("deleteCollection")

        #expect(library.bulkActivities.values.contains {
            $0 == "Deleting collection on 1 of 1 machines…"
        })
        hosts.update(replacement)
        await deleting.value
        #expect(library.bulkActivities.isEmpty)
        #expect(oldBackend.callCount("deleteCollection") == 1)
        #expect(oldBackend.callCount("collections") == 0)
        #expect(replacementBackend.callCount("deleteCollection") == 0)
        #expect(replacementBackend.callCount("collections") == 1)
    }
}
